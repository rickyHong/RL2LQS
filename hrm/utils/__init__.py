"""
HRM Utilities Package
====================

Contains utility functions and configuration management.
"""

from .config import HRMConfig, get_default_config

__all__ = [
    "HRMConfig",
    "get_default_config",
]