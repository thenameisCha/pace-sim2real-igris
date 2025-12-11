# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from isaaclab.utils import configclass

from pace_sim2real.assets.robots.robros import IGRIS_C_WAIST_CFG
from isaaclab.assets import ArticulationCfg
from pace_sim2real.utils import PaceDCMotorCfg
from actuators import PacefourbarDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch
M_PI = 3.141592
MYACTUATOR_PACE_ACTUATOR = {
        "legs": PaceDCMotorCfg(
            joint_names_expr=[".*Hip.*", ".*Knee.*"],
            velocity_limit_sim=8.5,
            effort_limit_sim={
                ".*Hip_Pitch.*": 150,
                ".*Hip_Roll.*": 120,
                ".*Hip_Yaw.*": 60,
                ".*Knee.*": 150,
            },
            stiffness={
                ".*Hip_Pitch.*": 150.,
                ".*Hip_Roll.*": 150.,
                ".*Hip_Yaw.*": 100.,
                ".*Knee.*": 150.,
            }, 
            damping={
                ".*Hip_Pitch.*": 2.,
                ".*Hip_Roll.*": 3.,
                ".*Hip_Yaw.*": 1.5,
                ".*Knee.*": 1.,
            },
            encoder_bias=[0.0] * 8,  # encoder bias in radians
            max_delay=6,  # max delay in simulation steps
        ),
        "Lankle": PacefourbarDCMotorCfg(
            joint_names_expr=["Left_Ankle.*"],
            velocity_limit_sim=8.5,
            effort_limit_sim={".*": 90},
            stiffness={".*": 50.0},  # P gain in Nm/rad
            damping={
                ".*": 3.,
            },
            encoder_bias=[0.0] * 8,  # encoder bias in radians
            max_delay=6,  # max delay in simulation steps
            constraints={
            'r_a_init_': [
                [0.0, 0.03775, 0.26],
                [0.0, -0.03775, 0.152]
            ],
            'r_b_init_': [
                [-0.03750, 0.03750, 0.25989],
                [-0.03750, -0.03750, 0.15181]
            ],
            'r_c_init_': [
                [-0.03400, 0.03100, 0.0],
                [-0.03400, -0.03100, 0.0]
            ],
            'r_c_offset_local_': [
                [-0.034, 0.031, 0.0],
                [-0.034, -0.031, 0.0]
            ],
            
            'base_to_p1_offset': [0.0, 0.0, -0.0],
            'base_to_p1_axis': [0.0, 1.0, 0.0],
            'p1_to_p2_offset': [0.0, 0.0, -0.0],
            'p1_to_p2_axis': [1.0, 0.0, 0.0],
            'motor_angles_min_': [-36.1 *M_PI/180, -35.4 *M_PI/180],
            'motor_angles_max_': [34.9 *M_PI/180, 30 *M_PI/180],
            'is_elbow_up_': False
            }
        ),
        "Rankle": PacefourbarDCMotorCfg(
            joint_names_expr=["Right_Ankle.*"],
            effort_limit_sim=90,
            velocity_limit_sim=8.5,
            effort_limit_sim={".*": 90},
            stiffness={".*": 50.0},  # P gain in Nm/rad
            damping={
                ".*": 3.,
            },
            encoder_bias=[0.0] * 8,  # encoder bias in radians
            max_delay=6,  # max delay in simulation steps
            constraints={
            'r_a_init_': [
                [0.0, -0.03775, 0.26],
                [0.0, 0.03775, 0.152]
            ],
            'r_b_init_': [
                [-0.03750, -0.03750, 0.25989],
                [-0.03750, 0.03750, 0.15181]
            ],
            'r_c_init_': [
                [-0.03400, -0.03100, 0.0],
                [-0.03400, 0.03100, 0.0]
            ],
            'r_c_offset_local_': [
                [-0.034, -0.031, 0.0],
                [-0.034, 0.031, 0.0]
            ],
            
            'base_to_p1_offset': [0.0, 0.0, -0.0],
            'base_to_p1_axis': [0.0, 1.0, 0.0],
            'p1_to_p2_offset': [0.0, 0.0, -0.0],
            'p1_to_p2_axis': [1.0, 0.0, 0.0],
            'motor_angles_min_': [-36.1 *M_PI/180, -35.4 *M_PI/180],
            'motor_angles_max_': [34.9 *M_PI/180, 30 *M_PI/180],
            'is_elbow_up_': False
            }
        ),
        "waist": PacefourbarDCMotorCfg(
            joint_names_expr=[".*Waist.*"],
            effort_limit_sim=60,
            velocity_limit_sim=100.0,
            stiffness={".*": 70.0},  # P gain in Nm/rad
            damping={
                '.*Roll.*': 1.8,
                '.*Pitch.*': 1.8,
            },
            encoder_bias=[0.0] * 8,  # encoder bias in radians
            max_delay=6,  # max delay in simulation steps
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
    },



@configclass
class IgrisCPaceCfg(PaceCfg):
    """Pace configuration for Igris-C robot."""
    robot_name: str = "igris_c_sim"
    data_dir: str = "igris_c_sim/chirp_data.pt"  # located in pace_sim2real/data/igris_c_sim/chirp_data.pt
    bounds_params: torch.Tensor = torch.zeros((49, 2))  # 14 + 14 + 14 + 14 + 1 = 49 parameters to optimize
    joint_order: list[str] = [
        "Waist_Roll",
        "Waist_Pitch",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Knee_Pitch",
        "Right_Ankle_Pitch",
        "Right_Ankle_Roll",
    ]

    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[:14, 0] = 1e-5
        self.bounds_params[:14, 1] = 5.0  # armature between 1e-5 - 1.0 [kgm2]
        self.bounds_params[14:28, 1] = 15.0  # dof_damping between 0.0 - 7.0 [Nm s/rad]
        self.bounds_params[28:42, 1] = 2.  # friction between 0.0 - 0.5
        self.bounds_params[42:56, 0] = -0.1
        self.bounds_params[56:70, 1] = 0.2  # bias between -0.1 - 0.1 [rad]
        self.bounds_params[70, 1] = 10.0  # delay between 0.0 - 10.0 [sim steps]


@configclass
class IgrisCPaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for Anymal-D robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = IGRIS_C_WAIST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
                                                  actuators=MYACTUATOR_PACE_ACTUATOR)


@configclass
class IgrisCPaceEnvCfg(PaceSim2realEnvCfg):

    scene: IgrisCPaceSceneCfg = IgrisCPaceSceneCfg()
    sim2real: PaceCfg = IgrisCPaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.003333  # 300Hz simulation
        self.decimation = 1  # 300Hz control
