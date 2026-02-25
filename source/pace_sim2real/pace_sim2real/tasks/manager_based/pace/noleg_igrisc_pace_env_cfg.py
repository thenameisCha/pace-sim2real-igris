# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from isaaclab.utils import configclass

from pace_sim2real.assets.robots.robros import IGRIS_C_WAIST_CFG
from isaaclab.assets import ArticulationCfg
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real.actuators import PacefourbarDCMotorCfg, PacefourbarDCMotorReverseCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch

MYACTUATOR_PACE_ACTUATOR = {
        "X12-150": PaceDCMotorCfg(
            joint_names_expr=[".*Hip_Pitch.*", 
                              ],
            velocity_limit=10.,
            saturation_effort=150.,
            effort_limit=150.,
            stiffness={
                ".*Hip_Pitch.*": 150.0,
            }, 
            damping={
                ".*Hip_Pitch.*": 15.0,
            },
            encoder_bias=[0.0] * 2,  # encoder bias in radians
            max_delay=10,  # max delay in simulation steps
            motor_constant=[0.0] * 2,  # motor constant in log scale
        ),
        "X8-120": PaceDCMotorCfg(
            joint_names_expr=[".*Hip_Roll.*"],
            velocity_limit=16.54,
            saturation_effort=120.,
            effort_limit=120.,
            stiffness=150.0,
            damping=17.,
            encoder_bias=[0.0] * 2,  # encoder bias in radians
            max_delay=10,  # max delay in simulation steps
            motor_constant=[0.0] * 2,  # motor constant in log scale
        ),
        "X8-60": PaceDCMotorCfg(
            joint_names_expr=[
                # ".*Hip_Yaw.*", 
                ".*Waist_Yaw.*"],
            velocity_limit=16.0,
            saturation_effort=60.,
            effort_limit=60.,
            stiffness={
                ".*Waist_Yaw.*": 100.,
            }, 
            damping={
                ".*Waist_Yaw.*": 14.,
            },
            encoder_bias=[0.0] ,  # encoder bias in radians
            max_delay=10,  # max delay in simulation steps
            motor_constant=[0.0] ,  # motor constant in log scale
        ),
        "waist_fourbar": PacefourbarDCMotorReverseCfg(
            joint_names_expr=[".*Waist_Roll.*", ".*Waist_Pitch.*"],
            velocity_limit=16.0,
            effort_limit=60,
            saturation_effort=60.0,
            stiffness={".*": 100.},  # P gain in Nm/rad
            damping={
                '.*Roll.*': 18.,
                '.*Pitch.*': 18.,
            },
            encoder_bias=[0.0] * 2,  # encoder bias in radians
            max_delay=10,  # max delay in simulation steps
            motor_constant=[0.0] * 2,  # motor constant in log scale
            constraints={
            'r_a_init_': [
                [0.0, 0.0905, -0.04],
                [0.0, -0.0905, -0.04]
            ],
            'r_b_init_': [
                [-0.05167, 0.09050, -0.04587],
                [-0.05167, -0.09050, -0.04587]
            ],
            'r_c_init_': [
                [-0.05, 0.0940, 0.014],
                [-0.05, -0.0940, 0.014]
            ],
            'r_c_offset_local_': [
                [-0.05, 0.094, 0.014],
                [-0.05, -0.094, 0.014]
            ],
            
            'base_to_p1_offset': [0.0, 0.0, -0.04],
            'base_to_p1_axis': [0.0, -1.0, 0.0],
            'p1_to_p2_offset': [0.0, 0.0, 0.04],
            'p1_to_p2_axis': [-1.0, 0.0, 0.0],
            'motor_angles_min_': [-0.75, -0.75],
            'motor_angles_max_': [1.5, 1.5],
            'is_elbow_up_': True
            }
        ),
    }

@configclass
class IgrisCPaceCfg(PaceCfg):
    """Pace configuration for Igris-C robot."""
    robot_name: str = "igris_c_sim"
    data_dir: str = "igris_c_sim/waist/0203/chirp_data.pt"  # located in pace_sim2real/data/igris_c_sim/chirp_data.pt
    bounds_params: torch.Tensor = torch.zeros((32, 2))  # 7 + 7 + 7 + 7 + 3 + 1 = 65 parameters to optimize
    joint_order: list[str] = [
        'Joint_Waist_Yaw', 
        'Joint_Waist_Roll', 
        'Joint_Waist_Pitch', 
        'Joint_Hip_Pitch_Left', 
        'Joint_Hip_Roll_Left', 
        'Joint_Hip_Pitch_Right', 
        'Joint_Hip_Roll_Right', 
    ]
    drive_id: list[str] = [ # Actuator models in the real robot
        'X8-60',
        'X8-120',
        'X12-150',
    ]
    drive_dict: dict[str] = {
        'X8-60': {
            'id': 0
        }, 
        'X8-120': {
            'id': 1
        }, 
        'X12-150': {
            'id': 2
        }, 
        'waist_fourbar': {
            'id': 0
        }, 
    }

    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[0, 0] = 0.019
        self.bounds_params[1, 0] = 0.1242938
        self.bounds_params[2, 0] = 0.0472087
        self.bounds_params[[3, 5], 0] = 0.117
        self.bounds_params[[4,6], 0] = 0.055
        self.bounds_params[0, 1] = 0.019*1.5
        self.bounds_params[1, 1] = 0.1242938*1.5
        self.bounds_params[2, 1] = 0.0472087*1.5
        self.bounds_params[[3, 5], 1] = 0.117*1.5
        self.bounds_params[[4,6], 1] = 0.055*1.5

        # self.bounds_params[:15, 0] = 1e-5
        # self.bounds_params[:15, 1] = 1.0  # armature between 1e-5 - 1.0 [kgm2]
        self.bounds_params[7:14, 1] = 5.0  # dof_damping between 0.0 - 7.0 [Nm s/rad]
        self.bounds_params[8, 1] = 20.0  # More damping in waist pitch and roll
        self.bounds_params[9, 1] = 10.0  # More damping in waist pitch and roll
        self.bounds_params[14:21, 1] = 5.  # friction between 0.0 - 0.5
        self.bounds_params[21:28, 0] = -0.2
        self.bounds_params[21:28, 1] = 0.2  # bias between -0.1 - 0.1 [rad]
        self.bounds_params[28:31, 0] = -0. # Motor constants in log scale, [60, 90, 120, 150] Nm
        self.bounds_params[28:31, 1] = 0.
        self.bounds_params[31, 0] = 1.0  # delay between 0.0 - 10.0 [sim steps]
        self.bounds_params[31, 1] = 5.0  # delay between 0.0 - 10.0 [sim steps]


@configclass
class IgrisCPaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for IGRIS-C robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = IGRIS_C_WAIST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)),
                                                  actuators=MYACTUATOR_PACE_ACTUATOR)


@configclass
class IgrisCPaceEnvCfg(PaceSim2realEnvCfg):

    scene: IgrisCPaceSceneCfg = IgrisCPaceSceneCfg()
    sim2real: PaceCfg = IgrisCPaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.sim.dt = 1/300.  # 900Hz simulation
        self.decimation = 1  # 300Hz control1