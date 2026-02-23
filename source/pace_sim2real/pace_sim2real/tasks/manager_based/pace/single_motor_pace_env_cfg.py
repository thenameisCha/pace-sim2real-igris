# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from isaaclab.utils import configclass

from pace_sim2real.assets.robots.single_motor import SINGLE_MOTOR_CFG
from isaaclab.assets import ArticulationCfg
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real.actuators import PacefourbarDCMotorCfg, PacefourbarDCMotorReverseCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch

MYACTUATOR_PACE_ACTUATOR = {
        "motor_joint": PaceDCMotorCfg(
            joint_names_expr=["motor_joint"],
            velocity_limit=17.5,
            saturation_effort=1000.,
            effort_limit=90.,
            stiffness={
                ".*.*": 17.765301,
            }, 
            damping={
                ".*.*": 1.885000,
            },
            encoder_bias=[0.0],  # encoder bias in radians
            max_delay=10,  # max delay in simulation steps
            motor_constant=[0.0],  # motor constant in log scale
        ),
    }
@configclass
class MotorPaceCfg(PaceCfg):
    """Pace configuration for single motor."""
    robot_name: str = "single_motor_sim"
    data_dir: str = "single_motor/0213/BIONIC-X8-90_Kp17.765301_Kd1.885000_amplitude1.000000_PACE_log/chirp_data.pt"  # located in pace_sim2real/data/igris_c_sim/chirp_data.pt
    bounds_params: torch.Tensor = torch.zeros((6, 2))  # parameters to optimize
    joint_order: list[str] = [
        'motor_joint'
    ]
    drive_id: list[str] = [ # Actuator models in the real robot
        'motor_joint'
    ]
    drive_dict: dict[str] = {
        'motor_joint': {
            'id': 0
        }
    }

    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[0, 0] = 1e-5
        self.bounds_params[0, 1] = 1.0  # armature between 1e-5 - 1.0 [kgm2]
        self.bounds_params[1, 1] = 10.0  # dof_damping between 0.0 - 7.0 [Nm s/rad]
        self.bounds_params[2, 1] = 5.  # friction between 0.0 - 0.5
        self.bounds_params[3, 0] = -0.2
        self.bounds_params[3, 1] = 0.2  # bias between -0.1 - 0.1 [rad]
        self.bounds_params[4, 0] = -0. # Motor constants in log scale, [60, 90, 120, 150] Nm
        self.bounds_params[4, 1] = 0.
        self.bounds_params[5, 0] = 1.0  # delay between 0.0 - 10.0 [sim steps]
        self.bounds_params[5, 1] = 5.0  # delay between 0.0 - 10.0 [sim steps]


@configclass
class MotorPaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for IGRIS-C robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = SINGLE_MOTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)),
                                                  actuators=MYACTUATOR_PACE_ACTUATOR)


@configclass
class MotorPaceEnvCfg(PaceSim2realEnvCfg):

    scene: MotorPaceSceneCfg = MotorPaceSceneCfg()
    sim2real: PaceCfg = MotorPaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.sim.dt = 1/500.  # 500Hz simulation
        self.decimation = 1  # 500Hz control1