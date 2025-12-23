# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ROBROS robots.
"""

from isaaclab_assets.sensors.velodyne import VELODYNE_VLP_16_RAYCASTER_CFG

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from pace_sim2real.assets import PACE_ASSETS_EXT_DIR, PACE_ASSETS_DATA_DIR
from pace_sim2real.actuators import FourbarDCMotor, fourbarDCMotorCfg, fourbarDCMotorReverseCfg

import torch
##
# Configuration - Actuators.
##
MYACTUATOR_ACTUATOR = {
        "X12-150": DCMotorCfg(
            joint_names_expr=[".*Hip_Pitch.*", ".*Knee.*"],
            velocity_limit=10.,
            saturation_effort=150.,
            effort_limit=150.,
            stiffness={
                ".*Hip_Pitch.*": 150.,
                ".*Knee.*": 150.,
            }, 
            damping={
                ".*Hip_Pitch.*": 2.,
                ".*Knee.*": 1.,
            },
        ),
        "X8-120": DCMotorCfg(
            joint_names_expr=[".*Hip_Roll.*"],
            velocity_limit=16.54,
            saturation_effort=120.,
            effort_limit=120.,
            stiffness=150.,
            damping=3.,
        ),
        "X8-60": DCMotorCfg(
            joint_names_expr=[".*Hip_Yaw.*", ".*Waist_Yaw.*"],
            velocity_limit=16.0,
            saturation_effort=60.,
            effort_limit=60.,
            stiffness={
                ".*Hip_Yaw.*": 100.,
                ".*Waist_Yaw.*": 70.,
            }, 
            damping={
                ".*Hip_Yaw.*": 1.5,
                ".*Waist_Yaw.*": 1.8,
            },
        ),
        "Lankle": fourbarDCMotorCfg(
            joint_names_expr=['Joint_Ankle_Pitch_Left', 'Joint_Ankle_Roll_Left'],
            velocity_limit=13.61,
            saturation_effort=90,
            effort_limit={".*": 90},
            stiffness={".*": 50.0},  # P gain in Nm/rad
            damping={
                ".*": 3.,
            },
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
            'motor_angles_min_': [-36.1 *torch.pi/180, -35.4 *torch.pi/180],
            'motor_angles_max_': [34.9 *torch.pi/180, 30 *torch.pi/180],
            'is_elbow_up_': False
            }
        ),
        "Rankle": fourbarDCMotorCfg(
            joint_names_expr=['Joint_Ankle_Pitch_Right', 'Joint_Ankle_Roll_Right'],
            velocity_limit=13.61,
            saturation_effort=90,
            effort_limit={".*": 90},
            stiffness={".*": 50.0},  # P gain in Nm/rad
            damping={
                ".*": 3.,
            },
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
            'motor_angles_min_': [-36.1 *torch.pi/180, -35.4 *torch.pi/180],
            'motor_angles_max_': [34.9 *torch.pi/180, 30 *torch.pi/180],
            'is_elbow_up_': False
            }
        ),
        "waist": fourbarDCMotorReverseCfg(
            joint_names_expr=[".*Waist.*"],
            velocity_limit=16.0,
            effort_limit=60,
            saturation_effort=60.0,
            stiffness={".*": 70.0},  # P gain in Nm/rad
            damping={
                '.*Roll.*': 1.8,
                '.*Pitch.*': 1.8,
            },
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
"""Configuration for MyActuator actuators with DC actuator model."""


##
# Configuration - Articulation.
##

IGRIS_C_WAIST_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PACE_ASSETS_DATA_DIR}/Robots/ROBROS/igris_c/igris_c_v2/igris_c_v2_waist_flat.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.),
        joint_pos={
            ".*Hip_Pitch.*": -0.2,
            ".*Hip_Roll.*": 0.0,
            ".*Hip_Yaw.*": 0.0,
            ".*Knee.*": 0.3,
            ".*Ankle_Pitch.*": -0.15,
            ".*Ankle_Roll.*": 0.0,
            ".*Waist_Yaw.*": 0.0,
            ".*Waist_Pitch.*": 0.0,
            ".*Waist_Roll.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators=MYACTUATOR_ACTUATOR,
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for IGRIS-C, with only 15 joints allowed."""



##
# Configuration - Sensors.
##

