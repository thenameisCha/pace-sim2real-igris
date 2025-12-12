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

M_PI = 3.141592

##
# Configuration - Actuators.
##

BIONIC_150_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*Hip_Pitch", ".*Knee_Pitch"],
    saturation_effort=500.0,
    effort_limit=150.0,
    velocity_limit=7.5,
    stiffness={".*": 0.0},
    damping={".*": 0.0},
)
BIONIC_90_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*Ankle_Pitch", ".*Ankle_Roll"],
    saturation_effort=500.0,
    effort_limit=90.0,
    velocity_limit=7.5,
    stiffness={".*": 0.0},
    damping={".*": 0.0},
)
v4_120_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*Hip_Roll"],
    saturation_effort=500.0,
    effort_limit=120.0,
    velocity_limit=7.5,
    stiffness={".*": 0.0},
    damping={".*": 0.0},
)
v4_60_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*Hip_Yaw"],
    saturation_effort=500.0,
    effort_limit=60.0,
    velocity_limit=7.5,
    stiffness={".*": 0.0},
    damping={".*": 0.0},
)
"""Configuration for MyActuator actuators with DC actuator model."""


##
# Configuration - Articulation.
##

IGRIS_C_WAIST_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PACE_ASSETS_DATA_DIR}/Robots/ROBROS/igris_c/igris_c_v2/igris_c_v2_waist_fix.usd",
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
            ".*Waist_Pitch.*": 0.0,
            ".*Waist_Roll.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*Hip.*", ".*Knee.*"],
            velocity_limit=100.0,
            effort_limit={
                ".*Hip_Pitch.*": 150,
                ".*Hip_Roll.*": 120,
                ".*Hip_Yaw.*": 60,
                ".*Knee.*": 150,
            },
            armature={
                ".*Hip_Pitch.*": 0.0521,
                ".*Hip_Roll.*": 0.0786,
                ".*Hip_Yaw.*": 0.0307,
                ".*Knee.*": 0.0521,
            },
            friction={
                ".*Hip_Pitch.*": 0.,
                ".*Hip_Roll.*": 0.,
                ".*Hip_Yaw.*": 0.,
                ".*Knee.*": 0.,
            },
            dynamic_friction={
                ".*Hip_Pitch.*": 0.,
                ".*Hip_Roll.*": 0.,
                ".*Hip_Yaw.*": 0.,
                ".*Knee.*": 0.,
            },
            viscous_friction={
                ".*Hip_Pitch.*": 0.,
                ".*Hip_Roll.*": 0.,
                ".*Hip_Yaw.*": 0.,
                ".*Knee.*": 0.,
            },
            stiffness={".*": 0.0},  # For IGRIS-C, PD loop is done externally
            damping={
                ".*Hip_Pitch.*": 0.,
                ".*Hip_Roll.*": 0.,
                ".*Hip_Yaw.*": 0.,
                ".*Knee.*": 0.,
            },
        ),
        "Lankle": fourbarDCMotorCfg(
            joint_names_expr=['Joint_Ankle_Pitch_Left', 'Joint_Ankle_Roll_Left'],
            effort_limit=90,
            velocity_limit=100.0,
            armature=0.0307,
            friction={".*": 0.0},
            dynamic_friction={".*": 0.0},
            viscous_friction={".*": 0.0},
            stiffness={".*": 50.0},  # P gain in Nm/rad
            damping={
                '.*Roll.*': 3.,
                '.*Pitch.*': 3.,
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
            'motor_angles_min_': [-36.1 *M_PI/180, -35.4 *M_PI/180],
            'motor_angles_max_': [34.9 *M_PI/180, 30 *M_PI/180],
            'is_elbow_up_': False
            }
        ),
        "Rankle": fourbarDCMotorCfg(
            joint_names_expr=['Joint_Ankle_Pitch_Right', 'Joint_Ankle_Roll_Right'],
            effort_limit=90,
            velocity_limit=100.0,
            armature=0.0307,
            friction={".*": 0.0},
            dynamic_friction={".*": 0.0},
            viscous_friction={".*": 0.0},
            stiffness={".*": 50.0},  # P gain in Nm/rad
            damping={
                '.*Roll.*': 3.,
                '.*Pitch.*': 3.,
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
            'motor_angles_min_': [-36.1 *M_PI/180, -35.4 *M_PI/180],
            'motor_angles_max_': [34.9 *M_PI/180, 30 *M_PI/180],
            'is_elbow_up_': False
            }
        ),
        "waist": fourbarDCMotorReverseCfg(
            joint_names_expr=[".*Waist.*"],
            effort_limit=60,
            velocity_limit=100.0,
            armature=0.0307,
            friction=0.,
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
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for IGRIS-C, with only 14 joints allowed."""



##
# Configuration - Sensors.
##

