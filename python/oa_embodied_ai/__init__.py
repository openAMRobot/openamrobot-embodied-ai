# SPDX-License-Identifier: MIT
"""OpenAMRobot embodied AI tooling."""

# On Windows, PyTorch C++ runtime must initialize before pyarrow C++ libraries to prevent DLL collision
try:
    import torch
except ImportError:
    pass
