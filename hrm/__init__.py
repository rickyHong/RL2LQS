"""
Hierarchical Reasoning Model (HRM)
===================================

A hierarchical reasoning model for complex sequential decision-making tasks.
Inspired by the multi-level, multi-timescale structure of information processing
in the human brain.

This implementation provides:
- Hierarchical architecture with high-level planning and low-level execution
- Adaptive computation time for dynamic reasoning depth
- Efficient training with minimal data requirements
- Support for various reasoning tasks (Sudoku, maze solving, ARC-AGI, etc.)
"""

__version__ = "1.0.0"
__author__ = "HRM Development Team"

from .models.hrm_model import HierarchicalReasoningModel
from .models.components import HighLevelModule, LowLevelModule
from .training.trainer import HRMTrainer
from .evaluation.evaluator import HRMEvaluator
from .utils.config import HRMConfig

__all__ = [
    "HierarchicalReasoningModel",
    "HighLevelModule", 
    "LowLevelModule",
    "HRMTrainer",
    "HRMEvaluator",
    "HRMConfig",
]