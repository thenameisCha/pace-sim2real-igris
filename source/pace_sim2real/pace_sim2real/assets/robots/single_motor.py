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

"""Configuration for MyActuator actuators with DC actuator model."""
MYACTUATOR_ACTUATOR = {
        "joint": DCMotorCfg(
            joint_names_expr=["motor_joint"],
            velocity_limit=17.5,
            saturation_effort=60.,
            effort_limit=60.,
            stiffness={
                ".*.*": 59.475101,
            }, 
            damping={
                ".*.*": 4.732900,
            },
        ),
    }


##
# Configuration - Articulation.
##

SINGLE_MOTOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PACE_ASSETS_DATA_DIR}/Robots/ROBROS/single_motor/usd/single_motor_flatten8.usd",
        activate_contact_sensors=False,
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
            ".*.*": 0.0,
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

