# SPDX-License-Identifier: MIT
"""Pytest configuration for oa_embodied_ai test suite."""

# On Windows, torch C++ runtime must initialize before pyarrow to prevent DLL initialization collision
try:
    import torch
except ImportError:
    pass

