
from __future__ import annotations

import logging
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch.nn.functional as F
from isaaclab.utils.math import quat_apply, quat_mul, quat_rotate_inverse

from isaaclab.utils import DelayBuffer, LinearInterpolation
from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators import IdealPDActuator


if TYPE_CHECKING:
    from .fourbar_pd_cfg import (
        fourbarPDActuatorCfg,
        fourbarDCMotorCfg,
        PacefourbarDCMotorCfg
    )

# import logger
logger = logging.getLogger(__name__)

class FourbarPDActuator(IdealPDActuator):
    cfg: fourbarPDActuatorCfg

    def __init__(self, cfg: fourbarPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # parse configuration
        if self.cfg.constraints is None:
            raise ValueError("The fourbar constraints must be provided for the fourbar actuator model.")
        self._constraints = self.cfg.constraints
        # prepare motor level tensors
        self._motor_pos, self._motor_vel = torch.zeros_like(self.computed_effort), torch.zeros_like(self.computed_effort)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # calculate current motor-joint jacobian and motor positions with IK
        _motor_jac, self._motor_pos = self._fourbar_logic(joint_pos, self._constraints)
        # Jacobian mapping to calculate motor velocities
        self._motor_vel = torch.bmm(_motor_jac, joint_vel.unsqueeze(dim=-1)).squeeze(dim=-1)
        # Jacobian mapping to calculate ff motor efforts
        motor_efforts = torch.linalg.solve(_motor_jac.transpose(1,2), control_action.joint_efforts.unsqueeze(dim=-1)).squeeze(dim=-1)
        # motor_efforts = torch.bmm(motor_jac.inverse().transpose(1,2), control_action.joint_efforts.unsqueeze(dim=-1)).squeeze(dim=-1)
        # motor_efforts = control_action.joint_efforts

        # Assume control action is in motor space
        error_pos = control_action.joint_positions - self._motor_pos
        error_vel = control_action.joint_velocities - self._motor_vel
        # calculate the desired motor torques
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + motor_efforts
        # clip the torques based on the motor limits
        motor_effort = self._clip_effort(self.computed_effort)
        # Map motor level effort back to joint level
        self.applied_effort = torch.bmm(_motor_jac.transpose(1,2), motor_effort.unsqueeze(dim=-1)).squeeze(dim=-1)
        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action
    
    def _fourbar_logic(self, dof_pos_clipped, constraints):
        # IK computation, dof_pos -> motor_pos
        device = dof_pos_clipped.device
        dof_pos = dof_pos_clipped.clone()
        B = dof_pos.shape[0]
        r_c_ = torch.zeros((B, 2, 3), device=device) # [B,2,3]        
        r_a_init_ = torch.tensor(constraints['r_a_init_'], device=device).unsqueeze(dim=0).repeat(B,1,1)  # [B,2,3]
        r_b_init_ = torch.tensor(constraints['r_b_init_'], device=device).unsqueeze(dim=0).repeat(B,1,1)  # [B,2,3]
        r_c_init_ = torch.tensor(constraints['r_c_init_'], device=device).unsqueeze(dim=0).repeat(B,1,1)  # [B,2,3]
        r_c_offset_local_ = torch.tensor(constraints['r_c_offset_local_'], device=device).unsqueeze(dim=0).repeat(B,1,1)   # [B,2,3]
        base_to_p1_offset = torch.tensor(constraints['base_to_p1_offset'], device=device).unsqueeze(dim=0).repeat(B,1)  # [B,3]
        base_to_p1_axis = torch.tensor(constraints['base_to_p1_axis'], device=device).unsqueeze(dim=0).repeat(B,1)    # [B,3]
        p1_to_p2_offset = torch.tensor(constraints['p1_to_p2_offset'], device=device).unsqueeze(dim=0).repeat(B,1)    # [B,3]
        p1_to_p2_axis = torch.tensor(constraints['p1_to_p2_axis'], device=device).unsqueeze(dim=0).repeat(B,1)      # [B,3]
        is_elbow_up = constraints['is_elbow_up_'] # bool
        motor_limit_l_ = torch.tensor(constraints.get('motor_angles_min_', [-1.57, -1.57]), device=device).unsqueeze(dim=0).repeat(B,1)  
        motor_limit_h_ = torch.tensor(constraints.get('motor_angles_max_', [1.57, 1.57]), device=device).unsqueeze(dim=0).repeat(B,1)  
        motor_angles_temp = torch.zeros((B,2), device=device)

        l_bar_ = torch.norm(r_b_init_ - r_a_init_, dim=-1) # [B,2]
        l_rod_ = torch.norm(r_c_init_ - r_b_init_, dim=-1)  # [B,2]
        r_b_offset_local_ = r_b_init_ - r_a_init_  # [B,2,3]
        b_vec_ = r_b_init_ - r_a_init_  # [B,2,3]

        rot_base_to_p1 = axis_angle_to_quat(base_to_p1_axis, dof_pos[:, 0])  # [B,4]
        rot_p1_to_p2 = axis_angle_to_quat(p1_to_p2_axis, dof_pos[:, 1])  # [B,4]
        base_to_p1_offset_expanded = base_to_p1_offset.unsqueeze(1).repeat(1,2,1)  # [B,2,3]
        p1_to_p2_offset_expanded = p1_to_p2_offset.unsqueeze(1).repeat(1,2,1)  # [B,2,3]
        base_to_p1_axis_expanded = base_to_p1_axis.unsqueeze(1).repeat(1,2,1)  # [B,2,3]
        p1_to_p2_axis_expanded = p1_to_p2_axis.unsqueeze(1).repeat(1,2,1)  # [B,2,3]
        r_c_ = base_to_p1_offset.unsqueeze(1).repeat(1,2,1)  # [B,2,3]
        rot_base_to_p1_expanded = rot_base_to_p1.unsqueeze(1).repeat(1,2,1)  # [B,2,4]
        rot_p1_to_p2_expanded = rot_p1_to_p2.unsqueeze(1).repeat(1,2,1)  # [B,2,4]
        r_c_ = base_to_p1_offset_expanded + quat_apply(rot_base_to_p1_expanded, p1_to_p2_offset_expanded) + \
            quat_apply(quat_mul(rot_base_to_p1_expanded, rot_p1_to_p2_expanded), r_c_offset_local_)  # [B,2,3]
        
        a_vec_ = r_c_ - r_a_init_
        d = -(l_rod_.square()-l_bar_.square()-a_vec_.square().sum(dim=-1)) / 2 # [B,2]
        e = d - a_vec_[..., 1] * b_vec_[..., 1] # [B,2]
        A = (a_vec_.square()[..., 0] + a_vec_.square()[..., 2]) * (b_vec_.square()[..., 0] + b_vec_.square()[..., 2])  # [B,2]
        B_ = (a_vec_[..., 0] * b_vec_[..., 2] - a_vec_[..., 2] * b_vec_[..., 0]) * e  # [B,2]
        C = e.square() - (a_vec_.square()[..., 0] * b_vec_.square()[..., 0] + a_vec_.square()[..., 2] * b_vec_.square()[..., 2] + 2 * a_vec_[..., 0] * a_vec_[..., 2] * b_vec_[..., 0] * b_vec_[..., 2])  # [B,2]
        value_pos_sign = torch.clamp(
            (B_ + torch.sqrt(torch.clamp(B_ * B_ - A * C, min=0.0))) / A, -1.0, 1.0)
        value_neg_sign = torch.clamp(
            (B_ - torch.sqrt(torch.clamp(B_ * B_ - A * C, min=0.0))) / A, -1.0, 1.0)
        
        motor_angle_pos = torch.asin(value_pos_sign)  # [B,2]
        motor_angle_neg = torch.asin(value_neg_sign)  # [B,2]   
        motor_angle_candidates = torch.zeros((6,B,2), device=device)  # [6,B,2], initialized to large negative value meaning invalid

        env_1 = (motor_angle_pos >= motor_limit_l_) &\
            (motor_angle_pos <= motor_limit_h_) # [B,2]
        env_2 = (motor_angle_neg >= motor_limit_l_) &\
            (motor_angle_neg <= motor_limit_h_ )
        env_3 = (torch.pi - motor_angle_pos >= motor_limit_l_) &\
            (torch.pi - motor_angle_pos <= motor_limit_h_)
        env_4 = (torch.pi - motor_angle_neg >= motor_limit_l_) &\
            (torch.pi - motor_angle_neg <= motor_limit_h_)
        env_5 = (-torch.pi - motor_angle_pos >= motor_limit_l_ )&\
            (-torch.pi - motor_angle_pos <= motor_limit_h_)
        env_6 = (-torch.pi - motor_angle_neg >= motor_limit_l_) &\
            (-torch.pi - motor_angle_neg <= motor_limit_h_)
        env_masks = [env_1, env_2, env_3, env_4, env_5, env_6] # list of [B,2] masks
        motor_angle_candidates[0][env_1] = motor_angle_pos[env_1] 
        motor_angle_candidates[1][env_2] = motor_angle_neg[env_2]
        motor_angle_candidates[2][env_3] = torch.pi - motor_angle_pos[env_3]
        motor_angle_candidates[3][env_4] = torch.pi - motor_angle_neg[env_4]
        motor_angle_candidates[4][env_5] = -torch.pi - motor_angle_pos[env_5]
        motor_angle_candidates[5][env_6] = -torch.pi - motor_angle_neg[env_6]

        r_a_init_expanded = r_a_init_ # [B,2,3]
        r_b_offset_local_expanded = r_b_offset_local_  # [B,2,3]
        y_axis = torch.tensor([0.,1.,0.], device=device).unsqueeze(0)
        for j in range(6):
            env_mask = env_masks[j] # [B,2]
            q_motor_flattened = motor_angle_candidates[j, env_mask] # [2N]
            r_a_init_flattened = r_a_init_expanded[env_mask] # [2N,3]
            r_b_offset_local_flattened = r_b_offset_local_expanded[env_mask] # [2N,3]
            r_c_flattened = r_c_[env_mask]  # [2N,3]
            r_b_ = r_a_init_flattened + quat_apply(axis_angle_to_quat(y_axis, q_motor_flattened), r_b_offset_local_flattened)  # [2N,3])

            bar = r_b_ - r_a_init_flattened  # [2N,3]
            rod = r_c_flattened - r_b_  # [2N,3]
            l_rod_condition = torch.abs(rod.norm(dim=-1) - l_rod_[env_mask]) < 1e-4  # [2N]
            elbow_direction = torch.cross(bar, rod, dim=1)[:, 1] > 0 # [2N]
            elbow_direction_condition = (elbow_direction == is_elbow_up) # [2N]

            valid_condition = l_rod_condition & elbow_direction_condition  # [2N]
            motor_angles_temp[env_mask] = q_motor_flattened * valid_condition.float() + motor_angles_temp[env_mask] * (~valid_condition).float()

        motor_angles_temp = torch.clip(motor_angles_temp, motor_limit_l_, motor_limit_h_)
        
        # Jacobian computation, \dot{m} = J\dot{q}
        r_b_ = r_a_init_ + quat_apply(axis_angle_to_quat(y_axis, motor_angles_temp), r_b_offset_local_) # [B,2,3]
        r_bar_ = r_b_ - r_a_init_ # [B,2,3]
        r_rod_ = r_c_ - r_b_ # [B,2,3]
        J_x = torch.zeros((B,2,6), dtype=torch.float32, device=device)
        J_x[..., :3] = r_rod_
        J_x[..., 3:] = torch.cross(r_c_, r_rod_)
        J_theta = torch.zeros(B, 2, 2, device=device, dtype=torch.float32)
        cross_bar_rod = torch.cross(r_bar_, r_rod_, dim=-1)   # (B,2,3)
        J_theta[:, 0, 0] = cross_bar_rod[:, 0, 1]
        J_theta[:, 1, 1] = cross_bar_rod[:, 1, 1]

        d0 = J_theta[:, 0, 0].clone()
        d1 = J_theta[:, 1, 1].clone()
        eps = 1e-8
        mask0 = torch.abs(d0) < eps
        mask1 = torch.abs(d1) < eps

        d0[mask0] = torch.where(d0[mask0] >= 0, torch.full_like(d0[mask0], eps),
                                                torch.full_like(d0[mask0], -eps))
        d1[mask1] = torch.where(d1[mask1] >= 0, torch.full_like(d1[mask1], eps),
                                                torch.full_like(d1[mask1], -eps))
        J_=J_x.clone()
        J_[:,0] /= d0.unsqueeze(dim=1)
        J_[:,1] /= d1.unsqueeze(dim=1)

        axis1_base = F.normalize(torch.tensor(constraints['base_to_p1_axis'], device=device), dim=-1)   # (3,)
        axis2_local = F.normalize(torch.tensor(constraints['p1_to_p2_axis'], device=device), dim=-1)    # (3,)
        axis1_base = axis1_base.unsqueeze(0).expand(B, 3)                  # (B,3)
        axis2_local = axis2_local.unsqueeze(0).expand(B, 3)                # (B,3)

        # rotations we already computed earlier in IK
        rot_base_to_p1 = axis_angle_to_quat(
            torch.tensor(constraints['base_to_p1_axis'], device=device).unsqueeze(0).expand(B, 3),
            dof_pos[:, 0]
        )  # (B,4)

        # joint 2 axis in base frame
        axis2_base = quat_apply(rot_base_to_p1, axis2_local)               # (B,3)

        # joint origins and p2 origin in base
        base_to_p1_offset = torch.tensor(constraints['base_to_p1_offset'], device=device).unsqueeze(0).expand(B, 3)
        p1_to_p2_offset   = torch.tensor(constraints['p1_to_p2_offset'], device=device).unsqueeze(0).expand(B, 3)

        o1 = base_to_p1_offset                                             # (B,3)
        p2_origin = base_to_p1_offset + quat_apply(rot_base_to_p1, p1_to_p2_offset)  # (B,3)
        o2 = p2_origin                                                     # joint 2 axis through p2 origin

        # angular part
        Jw1 = axis1_base                                                  # (B,3)
        Jw2 = axis2_base                                                  # (B,3)

        # linear part: Jv1 = ω1 × (p2 - o1), Jv2 = ω2 × (p2 - o2) = 0
        Jv1 = torch.cross(Jw1, (p2_origin - o1), dim=-1)                  # (B,3)
        Jv2 = torch.zeros_like(Jv1)                                       # (B,3)

        # Build jac_joint with [linear; angular] order, already swapped like in C++
        jac_joint = torch.zeros(B, 6, 2, device=device, dtype=torch.float32)
        jac_joint[:, 0:3, 0] = Jv1
        jac_joint[:, 3:6, 0] = Jw1
        jac_joint[:, 0:3, 1] = Jv2
        jac_joint[:, 3:6, 1] = Jw2

        return torch.bmm(J_, jac_joint), motor_angles_temp
    
class FourbarPDActuatorReverse(FourbarPDActuator):
    '''
    Variant of the fourbar PD actuator, but swaps the joint space order
    '''
    def compute(self, control_action, joint_pos, joint_vel):
        joint_pos_, joint_vel_ = joint_pos[...,[1,0]], joint_vel[..., [1,0]]
        control_action_ = super().compute(control_action, joint_pos_, joint_vel_).clone()
        control_action.joint_efforts = control_action_.joint_efforts[...,[1,0]]
        return control_action

class FourbarDCMotor(FourbarPDActuator):
    cfg: fourbarDCMotorCfg

    def __init__(self, cfg: fourbarDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # parse configuration
        if self.cfg.saturation_effort is None:
            raise ValueError("The saturation_effort must be provided for the DC motor actuator model.")
        self._saturation_effort = self.cfg.saturation_effort
        # find the velocity on the torque-speed curve that intersects effort_limit in the second and fourth quadrant
        self._vel_at_effort_lim = self.velocity_limit * (1 + self.effort_limit / self._saturation_effort)
        # prepare motor level tensors
        self._motor_pos, self._motor_vel = torch.zeros_like(self.computed_effort), torch.zeros_like(self.computed_effort)
        # create buffer for zeros effort
        self._zeros_effort = torch.zeros_like(self.computed_effort)
        # check that quantities are provided
        if self.cfg.velocity_limit is None:
            raise ValueError("The velocity limit must be provided for the DC motor actuator model.")

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # save current joint vel
        self._motor_vel[:] = torch.clip(self._motor_vel, min=-self._vel_at_effort_lim, max=self._vel_at_effort_lim)
        # compute torque limits
        torque_speed_top = self._saturation_effort * (1.0 - self._motor_vel / self.velocity_limit)
        torque_speed_bottom = self._saturation_effort * (-1.0 - self._motor_vel / self.velocity_limit)
        # -- max limit
        max_effort = torch.clip(torque_speed_top, max=self.effort_limit)
        # -- min limit
        min_effort = torch.clip(torque_speed_bottom, min=-self.effort_limit)
        # clip the torques based on the motor limits
        clamped = torch.clip(effort, min=min_effort, max=max_effort)
        return clamped
    
class FourbarDCMotorReverse(FourbarDCMotor):
    '''
    Variant of the fourbar DC Motor, but swaps the joint space order
    '''
    def compute(self, control_action, joint_pos, joint_vel):
        joint_pos_, joint_vel_ = joint_pos[...,[1,0]], joint_vel[..., [1,0]]
        control_action_ = super().compute(control_action, joint_pos_, joint_vel_).clone()
        control_action.joint_efforts = control_action_.joint_efforts[...,[1,0]]
        return control_action

class PaceFourbarDCMotor(FourbarDCMotor):
    cfg: PacefourbarDCMotorCfg

    def __init__(self, cfg: PacefourbarDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if isinstance(cfg.encoder_bias, (list, tuple)):
            if len(cfg.encoder_bias) != self.num_joints:
                raise ValueError(
                    f"encoder_bias must have {self.num_joints} elements (one per joint), "
                    f"but got {len(cfg.encoder_bias)}: {cfg.encoder_bias}"
                )
        self.encoder_bias = torch.tensor(cfg.encoder_bias, device=self._device).unsqueeze(0).repeat(self._num_envs, 1)

        self.torques_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(cfg.max_delay, torch.arange(self._num_envs, device=self._device))

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # reset buffers
        self.torques_delay_buffer.reset(env_ids)

    def update_encoder_bias(self, encoder_bias: torch.Tensor):
        self.encoder_bias = encoder_bias

    def update_time_lags(self, delay: int | torch.Tensor, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(delay, env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # compute actuator model with encoder bias added to joint positions (joint position in encoder frame, not simulation frame)
        control_action_sim = super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
        control_action_sim.joint_efforts = self.torques_delay_buffer.compute(control_action_sim.joint_efforts)
        return control_action_sim

class PaceFourbarDCMotorReverse(PaceFourbarDCMotor):
    '''
    Variant of the Pace fourbar DC Motor, but swaps the joint space order
    '''
    def compute(self, control_action, joint_pos, joint_vel):
        joint_pos_, joint_vel_ = joint_pos[...,[1,0]], joint_vel[..., [1,0]]
        control_action_sim = super().compute(control_action, joint_pos_, joint_vel_)
        je_ = control_action_sim.joint_efforts.clone()
        control_action_sim.joint_efforts = je_[...,[1,0]]
        return control_action_sim

################HELPER#########################
def axis_angle_to_quat(axis_vecs: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Args:
        axis_vecs: (..., 3) tensor of rotation axes (need not be unit length).
        angles:    (...) tensor of rotation angles in radians; broadcast-compatible
                   with axis_vecs[..., 0].
    Returns:
        (..., 4) tensor of quaternions in [x, y, z, w] order.
    """
    if axis_vecs.shape[-1] != 3:
        raise ValueError(f"axis_vecs must end with dim 3, got {axis_vecs.shape}")

    # Move angles onto the same device/dtype
    angles = angles.to(device=axis_vecs.device, dtype=axis_vecs.dtype)

    # Determine broadcasted batch shape
    batch_shape = torch.broadcast_shapes(axis_vecs.shape[:-1], angles.shape)

    # Broadcast tensors
    axis = axis_vecs.expand(*batch_shape, 3)
    ang = angles.expand(*batch_shape)

    axis = F.normalize(axis, dim=-1)
    half = 0.5 * ang
    sin_half = torch.sin(half)
    cos_half = torch.cos(half)

    quat = torch.empty(*batch_shape, 4, dtype=axis.dtype, device=axis.device)
    quat[..., :3] = axis * sin_half.unsqueeze(-1)
    quat[..., 3] = cos_half
    return quat