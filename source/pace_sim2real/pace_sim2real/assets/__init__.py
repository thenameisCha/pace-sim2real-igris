# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Package containing asset and sensor configurations for PACE."""

import os

# Conveniences to other module directories via relative paths
PACE_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

PACE_ASSETS_DATA_DIR = os.path.join(PACE_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

from .robots import *
from .sensors import *
