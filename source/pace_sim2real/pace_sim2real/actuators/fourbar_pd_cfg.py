
from dataclasses import MISSING

from isaaclab.utils import configclass
import torch

from . import fourbar_pd
from isaaclab.actuators import (
    DCMotorCfg,
    DelayedPDActuatorCfg,
    IdealPDActuatorCfg,
    ImplicitActuatorCfg,
    RemotizedPDActuatorCfg,
) 

@configclass
class fourbarPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for fourbar linkage actuator model."""

    class_type: type = fourbar_pd.FourbarPDActuator

    constraints: dict = MISSING
    """Kinematic and Motor constraints on the actuator"""

@configclass
class fourbarDCMotorCfg(fourbarPDActuatorCfg):
    """Configuration for fourbar linkage DC motor model."""
    class_type: type = fourbar_pd.FourbarDCMotor

    constraints: dict = MISSING
    """Kinematic and Motor constraints on the actuator"""
    saturation_effort: float = MISSING
    """Peak motor force/torque of the electric DC motor (in N-m)."""

@configclass
class PacefourbarDCMotorCfg(fourbarDCMotorCfg):
    """Configuration for Pace fourbar DC Motor actuator model.

    This class extends the base fourbar DCMotorCfg with Pace-specific parameters.
    """
    class_type: type = fourbar_pd.PaceFourbarDCMotor
    encoder_bias: list[float] | float | None = 0.0
    max_delay: int | None = 0
@configclass
class fourbarPDActuatorReverseCfg(fourbarPDActuatorCfg):

    class_type: type = fourbar_pd.FourbarPDActuatorReverse

@configclass
class fourbarDCMotorReverseCfg(fourbarDCMotorCfg):
    """Configuration for fourbar linkage DC motor model."""
    class_type: type = fourbar_pd.FourbarDCMotorReverse

@configclass
class PacefourbarDCMotorReverseCfg(PacefourbarDCMotorCfg):
    """Configuration for Pace fourbar DC Motor actuator model.

    This class extends the base fourbar DCMotorCfg with Pace-specific parameters.
    """
    class_type: type = fourbar_pd.PaceFourbarDCMotorReverse
    
