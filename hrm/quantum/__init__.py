"""
Quantum Computing Extensions for Hierarchical Reasoning Model (HRM)

This package extends the HRM to quantum computing applications, specifically:
- Quantum state learning with Heisenberg scaling accuracy
- Reinforcement learning for quantum state optimization
- Quantum-enhanced hierarchical reasoning
- Quantum metrology and sensing applications

Key Features:
- QuantumHRM: Quantum-enhanced hierarchical reasoning model
- QuantumEnvironment: Quantum state learning environment
- HeisenbergScaling: Precision scaling mechanisms
- QuantumStateRepresentation: Quantum state encoding/decoding
- QuantumFidelityMetrics: Quantum-specific evaluation metrics
"""

from .quantum_hrm import QuantumHRM, QuantumHighLevelModule, QuantumLowLevelModule
from .quantum_environment import QuantumEnvironment, QuantumStateSpace
from .quantum_states import (
    QuantumStateRepresentation, 
    QuantumStateEncoder, 
    QuantumStateDecoder,
    HeisenbergScalingEstimator
)
from .quantum_rl import QuantumRLAgent, QuantumPolicyNetwork, QuantumValueNetwork
from .quantum_training import QuantumHRMTrainer, QuantumLoss, QuantumOptimizer
from .quantum_metrics import (
    QuantumFidelityMetric, 
    HeisenbergScalingMetric,
    QuantumEntanglementMetric
)
from .quantum_datasets import (
    QuantumStateDataset,
    QuantumTomographyDataset, 
    QuantumMetrologyDataset
)

__all__ = [
    'QuantumHRM',
    'QuantumHighLevelModule', 
    'QuantumLowLevelModule',
    'QuantumEnvironment',
    'QuantumStateSpace',
    'QuantumStateRepresentation',
    'QuantumStateEncoder',
    'QuantumStateDecoder', 
    'HeisenbergScalingEstimator',
    'QuantumRLAgent',
    'QuantumPolicyNetwork',
    'QuantumValueNetwork',
    'QuantumHRMTrainer',
    'QuantumLoss',
    'QuantumOptimizer',
    'QuantumFidelityMetric',
    'HeisenbergScalingMetric', 
    'QuantumEntanglementMetric',
    'QuantumStateDataset',
    'QuantumTomographyDataset',
    'QuantumMetrologyDataset'
]

__version__ = "1.0.0"
__author__ = "HRM Quantum Team"