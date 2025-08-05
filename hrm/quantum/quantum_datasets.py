"""
Quantum Datasets for HRM

This module implements various quantum datasets for training and evaluating
quantum-enhanced hierarchical reasoning models.

Key Components:
- QuantumStateDataset: General quantum state dataset
- QuantumTomographyDataset: Quantum state tomography dataset
- QuantumMetrologyDataset: Quantum metrology and sensing dataset
- QuantumBenchmarkDataset: Benchmark datasets for evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import random
import pickle
import json

from .quantum_states import (
    QuantumStateRepresentation,
    QuantumStateConfig,
    create_ghz_state,
    create_spin_squeezed_state,
    quantum_fidelity,
    von_neumann_entropy
)

@dataclass
class QuantumDatasetConfig:
    """Configuration for quantum datasets"""
    n_qubits: int = 4
    dataset_size: int = 10000
    validation_split: float = 0.2
    test_split: float = 0.1
    noise_level: float = 0.01
    measurement_shots: int = 1024
    target_states: List[str] = None
    difficulty_levels: List[str] = None
    include_mixed_states: bool = True
    random_seed: int = 42

class QuantumStateDataset(Dataset):
    """
    General quantum state dataset for training quantum models
    
    This dataset generates various quantum states including:
    - Pure states (coherent superpositions)
    - Mixed states (statistical mixtures)
    - Entangled states (GHZ, Bell, spin-squeezed)
    - Random states (Haar random)
    """
    
    def __init__(self, config: QuantumDatasetConfig):
        self.config = config
        self.n_qubits = config.n_qubits
        self.dim = 2 ** self.n_qubits
        
        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Generate dataset
        self.states, self.labels, self.metadata = self._generate_dataset()
        
        # Split dataset
        self.train_indices, self.val_indices, self.test_indices = self._split_dataset()
        
    def _generate_dataset(self) -> Tuple[List[torch.Tensor], List[int], List[Dict]]:
        """Generate quantum states dataset"""
        states = []
        labels = []
        metadata = []
        
        # Define state types and their generators
        state_generators = {
            0: self._generate_product_state,      # Product states
            1: self._generate_ghz_state,          # GHZ states
            2: self._generate_bell_state,         # Bell states
            3: self._generate_spin_squeezed_state, # Spin-squeezed states
            4: self._generate_random_state,       # Random states
            5: self._generate_coherent_state,     # Coherent states
            6: self._generate_mixed_state,        # Mixed states
            7: self._generate_cat_state           # Cat states
        }
        
        # Generate balanced dataset
        states_per_class = self.config.dataset_size // len(state_generators)
        
        for label, generator in state_generators.items():
            for i in range(states_per_class):
                state, meta = generator()
                states.append(state)
                labels.append(label)
                metadata.append(meta)
        
        # Add remaining states randomly
        remaining = self.config.dataset_size - len(states)
        for _ in range(remaining):
            label = random.choice(list(state_generators.keys()))
            state, meta = state_generators[label]()
            states.append(state)
            labels.append(label)
            metadata.append(meta)
        
        # Shuffle dataset
        indices = list(range(len(states)))
        random.shuffle(indices)
        
        states = [states[i] for i in indices]
        labels = [labels[i] for i in indices]
        metadata = [metadata[i] for i in indices]
        
        return states, labels, metadata
    
    def _generate_product_state(self) -> Tuple[torch.Tensor, Dict]:
        """Generate product state |ψ⟩ = |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ ..."""
        state = torch.tensor([1.0 + 0j], dtype=torch.complex64)
        
        for qubit in range(self.n_qubits):
            # Random single-qubit state
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            
            qubit_state = torch.tensor([
                np.cos(theta/2),
                np.exp(1j * phi) * np.sin(theta/2)
            ], dtype=torch.complex64)
            
            state = torch.kron(state, qubit_state)
        
        metadata = {
            'state_type': 'product',
            'entanglement': 0.0,
            'purity': 1.0
        }
        
        return state, metadata
    
    def _generate_ghz_state(self) -> Tuple[torch.Tensor, Dict]:
        """Generate GHZ state"""
        state = create_ghz_state(self.n_qubits)
        
        # Add noise if specified
        if self.config.noise_level > 0:
            noise = torch.randn_like(state) * self.config.noise_level
            state = state + noise
            state = F.normalize(state, p=2, dim=0)
        
        metadata = {
            'state_type': 'ghz',
            'entanglement': math.log2(2),  # Maximum entanglement for GHZ
            'purity': 1.0
        }
        
        return state, metadata
    
    def _generate_bell_state(self) -> Tuple[torch.Tensor, Dict]:
        """Generate Bell state (extended to n qubits)"""
        state = torch.zeros(self.dim, dtype=torch.complex64)
        
        if self.n_qubits >= 2:
            # Bell state for first two qubits, others in |0⟩
            state[0] = 1.0 / math.sqrt(2)  # |00...0⟩
            state[3] = 1.0 / math.sqrt(2)  # |11...0⟩ (for first two qubits)
        else:
            # Single qubit superposition
            state[0] = 1.0 / math.sqrt(2)
            state[1] = 1.0 / math.sqrt(2)
        
        metadata = {
            'state_type': 'bell',
            'entanglement': 1.0 if self.n_qubits >= 2 else 0.0,
            'purity': 1.0
        }
        
        return state, metadata
    
    def _generate_spin_squeezed_state(self) -> Tuple[torch.Tensor, Dict]:
        """Generate spin-squeezed state"""
        squeezing_param = np.random.uniform(0.1, 1.0)
        state = create_spin_squeezed_state(self.n_qubits, squeezing_param)
        
        metadata = {
            'state_type': 'spin_squeezed',
            'squeezing_parameter': squeezing_param,
            'entanglement': self._compute_entanglement(state),
            'purity': 1.0
        }
        
        return state, metadata
    
    def _generate_random_state(self) -> Tuple[torch.Tensor, Dict]:
        """Generate Haar random state"""
        # Generate random complex amplitudes
        real_part = torch.randn(self.dim)
        imag_part = torch.randn(self.dim)
        state = torch.complex(real_part, imag_part)
        
        # Normalize
        state = F.normalize(state, p=2, dim=0)
        
        metadata = {
            'state_type': 'random',
            'entanglement': self._compute_entanglement(state),
            'purity': 1.0
        }
        
        return state, metadata
    
    def _generate_coherent_state(self) -> Tuple[torch.Tensor, Dict]:
        """Generate coherent spin state"""
        # Coherent state: all spins aligned
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        
        # Single-qubit coherent state
        single_state = torch.tensor([
            np.cos(theta/2),
            np.exp(1j * phi) * np.sin(theta/2)
        ], dtype=torch.complex64)
        
        # Tensor product for all qubits
        state = single_state
        for _ in range(self.n_qubits - 1):
            state = torch.kron(state, single_state)
        
        metadata = {
            'state_type': 'coherent',
            'theta': theta,
            'phi': phi,
            'entanglement': 0.0,
            'purity': 1.0
        }
        
        return state, metadata
    
    def _generate_mixed_state(self) -> Tuple[torch.Tensor, Dict]:
        """Generate mixed state (as purification of larger system)"""
        if not self.config.include_mixed_states:
            return self._generate_random_state()
        
        # Generate random pure state in larger Hilbert space
        extended_dim = self.dim * 2
        real_part = torch.randn(extended_dim)
        imag_part = torch.randn(extended_dim)
        pure_state = torch.complex(real_part, imag_part)
        pure_state = F.normalize(pure_state, p=2, dim=0)
        
        # Create density matrix
        rho_extended = torch.outer(pure_state, torch.conj(pure_state))
        
        # Partial trace to get mixed state
        rho_mixed = torch.zeros(self.dim, self.dim, dtype=torch.complex64)
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(2):
                    idx1 = i * 2 + k
                    idx2 = j * 2 + k
                    rho_mixed[i, j] += rho_extended[idx1, idx2]
        
        # Convert back to state vector (approximate)
        eigenvals, eigenvecs = torch.linalg.eigh(rho_mixed)
        eigenvals = torch.real(eigenvals)
        
        # Take the dominant eigenstate as approximation
        max_idx = torch.argmax(eigenvals)
        state = eigenvecs[:, max_idx]
        
        # Ensure proper normalization and type
        state = state.to(torch.complex64)
        state = F.normalize(state, p=2, dim=0)
        
        purity = torch.real(torch.trace(torch.matmul(rho_mixed, rho_mixed))).item()
        
        metadata = {
            'state_type': 'mixed',
            'entanglement': self._compute_entanglement(state),
            'purity': purity
        }
        
        return state, metadata
    
    def _generate_cat_state(self) -> Tuple[torch.Tensor, Dict]:
        """Generate cat state (superposition of coherent states)"""
        state = torch.zeros(self.dim, dtype=torch.complex64)
        
        # Cat state: superposition of all |0⟩ and all |1⟩
        state[0] = 1.0 / math.sqrt(2)      # |00...0⟩
        state[-1] = 1.0 / math.sqrt(2)     # |11...1⟩
        
        # Add random phase
        phase = np.random.uniform(0, 2*np.pi)
        state[-1] *= np.exp(1j * phase)
        
        metadata = {
            'state_type': 'cat',
            'phase': phase,
            'entanglement': math.log2(2),
            'purity': 1.0
        }
        
        return state, metadata
    
    def _compute_entanglement(self, state: torch.Tensor) -> float:
        """Compute entanglement measure (simplified)"""
        if self.n_qubits == 1:
            return 0.0
        
        try:
            # Create density matrix
            rho = torch.outer(state, torch.conj(state))
            
            # Partial trace for first qubit
            dim_subsystem = 2
            dim_environment = self.dim // 2
            
            rho_reduced = torch.zeros(dim_subsystem, dim_subsystem, dtype=torch.complex64)
            
            for i in range(dim_subsystem):
                for j in range(dim_subsystem):
                    for k in range(dim_environment):
                        idx1 = i * dim_environment + k
                        idx2 = j * dim_environment + k
                        rho_reduced[i, j] += rho[idx1, idx2]
            
            # Von Neumann entropy
            eigenvals = torch.real(torch.linalg.eigvals(rho_reduced))
            eigenvals = eigenvals[eigenvals > 1e-12]
            entropy = -torch.sum(eigenvals * torch.log2(eigenvals + 1e-12))
            
            return entropy.item()
        except:
            return 0.0
    
    def _split_dataset(self) -> Tuple[List[int], List[int], List[int]]:
        """Split dataset into train/validation/test"""
        n_samples = len(self.states)
        indices = list(range(n_samples))
        
        # Calculate split sizes
        test_size = int(n_samples * self.config.test_split)
        val_size = int(n_samples * self.config.validation_split)
        train_size = n_samples - test_size - val_size
        
        # Split indices
        test_indices = indices[:test_size]
        val_indices = indices[test_size:test_size + val_size]
        train_indices = indices[test_size + val_size:]
        
        return train_indices, val_indices, test_indices
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item"""
        state = self.states[idx]
        label = self.labels[idx]
        metadata = self.metadata[idx]
        
        # Convert to real representation for neural networks
        real_part = torch.real(state)
        imag_part = torch.imag(state)
        state_real = torch.cat([real_part, imag_part])
        
        return {
            'state': state,
            'state_real': state_real,
            'label': torch.tensor(label, dtype=torch.long),
            'metadata': metadata
        }
    
    def get_subset(self, subset: str) -> 'QuantumStateDataset':
        """Get train/validation/test subset"""
        if subset == 'train':
            indices = self.train_indices
        elif subset == 'validation':
            indices = self.val_indices
        elif subset == 'test':
            indices = self.test_indices
        else:
            raise ValueError(f"Unknown subset: {subset}")
        
        # Create subset
        subset_dataset = QuantumStateDataset.__new__(QuantumStateDataset)
        subset_dataset.config = self.config
        subset_dataset.n_qubits = self.n_qubits
        subset_dataset.dim = self.dim
        
        subset_dataset.states = [self.states[i] for i in indices]
        subset_dataset.labels = [self.labels[i] for i in indices]
        subset_dataset.metadata = [self.metadata[i] for i in indices]
        
        return subset_dataset

class QuantumTomographyDataset(Dataset):
    """
    Quantum state tomography dataset
    
    This dataset provides quantum states along with measurement outcomes
    for training quantum state reconstruction models.
    """
    
    def __init__(self, config: QuantumDatasetConfig):
        self.config = config
        self.n_qubits = config.n_qubits
        self.dim = 2 ** self.n_qubits
        
        # Base quantum state dataset
        self.base_dataset = QuantumStateDataset(config)
        
        # Measurement operators (Pauli measurements)
        self.measurement_operators = self._create_measurement_operators()
        
        # Generate measurement data
        self.measurement_data = self._generate_measurement_data()
        
    def _create_measurement_operators(self) -> List[torch.Tensor]:
        """Create measurement operators for tomography"""
        operators = []
        
        # Single-qubit Pauli operators
        pauli_x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex64)
        pauli_y = torch.tensor([[0., -1j], [1j, 0.]], dtype=torch.complex64)
        pauli_z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.complex64)
        identity = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.complex64)
        
        single_operators = [identity, pauli_x, pauli_y, pauli_z]
        
        # Generate all combinations for multi-qubit measurements
        from itertools import product
        
        for combination in product(range(4), repeat=self.n_qubits):
            # Build tensor product
            operator = single_operators[combination[0]]
            for i in range(1, self.n_qubits):
                operator = torch.kron(operator, single_operators[combination[i]])
            
            operators.append(operator)
        
        return operators
    
    def _generate_measurement_data(self) -> List[Dict]:
        """Generate measurement outcomes for each state"""
        measurement_data = []
        
        for i, state in enumerate(self.base_dataset.states):
            measurements = {}
            
            for j, operator in enumerate(self.measurement_operators):
                # Compute expectation value
                expectation = torch.real(
                    torch.conj(state) @ operator @ state
                ).item()
                
                # Simulate measurement shots
                outcomes = []
                for _ in range(self.config.measurement_shots):
                    # Bernoulli trial based on expectation value
                    prob = (expectation + 1) / 2  # Map [-1,1] to [0,1]
                    outcome = 1 if np.random.random() < prob else -1
                    outcomes.append(outcome)
                
                measurements[f'operator_{j}'] = {
                    'expectation': expectation,
                    'outcomes': outcomes,
                    'mean_outcome': np.mean(outcomes),
                    'variance': np.var(outcomes)
                }
            
            measurement_data.append(measurements)
        
        return measurement_data
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tomography dataset item"""
        base_item = self.base_dataset[idx]
        measurements = self.measurement_data[idx]
        
        # Convert measurements to tensor format
        expectation_values = []
        measurement_outcomes = []
        
        for key, data in measurements.items():
            expectation_values.append(data['expectation'])
            measurement_outcomes.extend(data['outcomes'])
        
        base_item.update({
            'expectation_values': torch.tensor(expectation_values, dtype=torch.float32),
            'measurement_outcomes': torch.tensor(measurement_outcomes, dtype=torch.float32),
            'n_measurements': len(self.measurement_operators),
            'shots_per_measurement': self.config.measurement_shots
        })
        
        return base_item

class QuantumMetrologyDataset(Dataset):
    """
    Quantum metrology dataset for Heisenberg scaling
    
    This dataset provides quantum states optimized for parameter estimation
    with Heisenberg scaling accuracy.
    """
    
    def __init__(self, config: QuantumDatasetConfig):
        self.config = config
        self.n_qubits = config.n_qubits
        self.dim = 2 ** self.n_qubits
        
        # Generate metrology-specific states
        self.states, self.parameters, self.fisher_info = self._generate_metrology_data()
        
    def _generate_metrology_data(self) -> Tuple[List[torch.Tensor], List[float], List[float]]:
        """Generate quantum states for metrology"""
        states = []
        parameters = []
        fisher_info = []
        
        for _ in range(self.config.dataset_size):
            # Random parameter to estimate
            param = np.random.uniform(-np.pi, np.pi)
            
            # Generate state optimized for parameter estimation
            state_type = np.random.choice(['ghz', 'spin_squeezed', 'noon'])
            
            if state_type == 'ghz':
                state = self._create_ghz_probe_state(param)
                qfi = self.n_qubits ** 2  # Heisenberg scaling
            elif state_type == 'spin_squeezed':
                squeezing = np.random.uniform(0.1, 1.0)
                state = self._create_squeezed_probe_state(param, squeezing)
                qfi = self.n_qubits ** 1.5  # Between standard and Heisenberg
            else:  # noon
                state = self._create_noon_state(param)
                qfi = self.n_qubits ** 2  # Heisenberg scaling
            
            states.append(state)
            parameters.append(param)
            fisher_info.append(qfi)
        
        return states, parameters, fisher_info
    
    def _create_ghz_probe_state(self, param: float) -> torch.Tensor:
        """Create GHZ state for parameter estimation"""
        state = torch.zeros(self.dim, dtype=torch.complex64)
        state[0] = 1.0 / math.sqrt(2)  # |00...0⟩
        state[-1] = torch.exp(1j * param * self.n_qubits) / math.sqrt(2)  # |11...1⟩
        return state
    
    def _create_squeezed_probe_state(self, param: float, squeezing: float) -> torch.Tensor:
        """Create spin-squeezed state for parameter estimation"""
        # Start with coherent state
        coherent_state = torch.ones(self.dim, dtype=torch.complex64) / math.sqrt(self.dim)
        
        # Apply parameter-dependent evolution
        for i in range(self.dim):
            # Count number of |1⟩s in binary representation
            n_ones = bin(i).count('1')
            phase = param * n_ones
            coherent_state[i] *= torch.exp(1j * phase)
        
        # Apply squeezing (simplified)
        squeezing_matrix = torch.eye(self.dim, dtype=torch.complex64)
        for i in range(self.dim):
            variance = (i - self.dim/2)**2
            squeezing_matrix[i, i] = torch.exp(1j * squeezing * variance / self.dim)
        
        squeezed_state = squeezing_matrix @ coherent_state
        return F.normalize(squeezed_state, p=2, dim=0)
    
    def _create_noon_state(self, param: float) -> torch.Tensor:
        """Create NOON state for parameter estimation"""
        state = torch.zeros(self.dim, dtype=torch.complex64)
        
        # NOON state: superposition of all particles in mode A or mode B
        # Simplified as superposition of |00...0⟩ and |11...1⟩
        state[0] = 1.0 / math.sqrt(2)  # All in mode A
        state[-1] = torch.exp(1j * param * self.n_qubits) / math.sqrt(2)  # All in mode B
        
        return state
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get metrology dataset item"""
        state = self.states[idx]
        param = self.parameters[idx]
        qfi = self.fisher_info[idx]
        
        # Convert to real representation
        real_part = torch.real(state)
        imag_part = torch.imag(state)
        state_real = torch.cat([real_part, imag_part])
        
        # Compute Heisenberg scaling factor
        n_particles = self.n_qubits
        heisenberg_scaling = 1.0 / n_particles  # N^-1
        sql_scaling = 1.0 / math.sqrt(n_particles)  # N^-1/2
        scaling_advantage = sql_scaling / heisenberg_scaling  # √N
        
        return {
            'state': state,
            'state_real': state_real,
            'parameter': torch.tensor(param, dtype=torch.float32),
            'fisher_info': torch.tensor(qfi, dtype=torch.float32),
            'heisenberg_scaling': torch.tensor(heisenberg_scaling, dtype=torch.float32),
            'scaling_advantage': torch.tensor(scaling_advantage, dtype=torch.float32),
            'n_particles': torch.tensor(n_particles, dtype=torch.long)
        }

# Factory functions for creating datasets
def create_quantum_dataset(dataset_type: str, 
                          config: Optional[QuantumDatasetConfig] = None,
                          **kwargs) -> Dataset:
    """Create quantum dataset of specified type"""
    
    if config is None:
        config = QuantumDatasetConfig(**kwargs)
    
    if dataset_type == 'states':
        return QuantumStateDataset(config)
    elif dataset_type == 'tomography':
        return QuantumTomographyDataset(config)
    elif dataset_type == 'metrology':
        return QuantumMetrologyDataset(config)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def create_quantum_dataloaders(dataset: Dataset,
                             batch_size: int = 32,
                             train_split: float = 0.8,
                             val_split: float = 0.1,
                             test_split: float = 0.1,
                             num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/validation/test dataloaders"""
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # Calculate split sizes
    train_size = int(dataset_size * train_split)
    val_size = int(dataset_size * val_split)
    test_size = dataset_size - train_size - val_size
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

# Benchmark datasets for evaluation
class QuantumBenchmarkDataset:
    """Collection of benchmark quantum datasets"""
    
    @staticmethod
    def get_benchmark_states(n_qubits: int) -> Dict[str, torch.Tensor]:
        """Get standard benchmark quantum states"""
        benchmarks = {}
        
        # GHZ state
        benchmarks['ghz'] = create_ghz_state(n_qubits)
        
        # Bell state (extended)
        bell_state = torch.zeros(2**n_qubits, dtype=torch.complex64)
        bell_state[0] = 1.0 / math.sqrt(2)
        if n_qubits >= 2:
            bell_state[3] = 1.0 / math.sqrt(2)
        else:
            bell_state[1] = 1.0 / math.sqrt(2)
        benchmarks['bell'] = bell_state
        
        # W state
        w_state = torch.zeros(2**n_qubits, dtype=torch.complex64)
        for i in range(n_qubits):
            idx = 1 << i  # Single |1⟩ in position i
            w_state[idx] = 1.0 / math.sqrt(n_qubits)
        benchmarks['w'] = w_state
        
        # Uniform superposition
        benchmarks['uniform'] = torch.ones(2**n_qubits, dtype=torch.complex64) / math.sqrt(2**n_qubits)
        
        # Spin-squeezed states with different parameters
        for squeezing in [0.1, 0.5, 1.0]:
            key = f'spin_squeezed_{squeezing}'
            benchmarks[key] = create_spin_squeezed_state(n_qubits, squeezing)
        
        return benchmarks
    
    @staticmethod
    def evaluate_state_fidelity(predicted_state: torch.Tensor, 
                               target_state: torch.Tensor) -> float:
        """Evaluate fidelity between predicted and target states"""
        return quantum_fidelity(
            predicted_state.unsqueeze(0), 
            target_state.unsqueeze(0)
        ).item()
    
    @staticmethod
    def evaluate_entanglement_measure(state: torch.Tensor, n_qubits: int) -> float:
        """Evaluate entanglement measure of quantum state"""
        if n_qubits == 1:
            return 0.0
        
        try:
            # Create density matrix
            rho = torch.outer(state, torch.conj(state))
            
            # Partial trace for first qubit
            dim_subsystem = 2
            dim_environment = len(state) // 2
            
            rho_reduced = torch.zeros(dim_subsystem, dim_subsystem, dtype=torch.complex64)
            
            for i in range(dim_subsystem):
                for j in range(dim_subsystem):
                    for k in range(dim_environment):
                        idx1 = i * dim_environment + k
                        idx2 = j * dim_environment + k
                        rho_reduced[i, j] += rho[idx1, idx2]
            
            # Von Neumann entropy
            eigenvals = torch.real(torch.linalg.eigvals(rho_reduced))
            eigenvals = eigenvals[eigenvals > 1e-12]
            entropy = -torch.sum(eigenvals * torch.log2(eigenvals + 1e-12))
            
            return entropy.item()
        except:
            return 0.0

# Utility functions for dataset analysis
def analyze_dataset_statistics(dataset: Dataset) -> Dict[str, Any]:
    """Analyze statistics of quantum dataset"""
    stats = {
        'dataset_size': len(dataset),
        'state_types': {},
        'entanglement_distribution': [],
        'purity_distribution': [],
        'fidelity_with_benchmarks': {}
    }
    
    # Analyze each sample
    for i in range(len(dataset)):
        item = dataset[i]
        metadata = item['metadata']
        
        # Count state types
        state_type = metadata['state_type']
        stats['state_types'][state_type] = stats['state_types'].get(state_type, 0) + 1
        
        # Collect entanglement and purity
        stats['entanglement_distribution'].append(metadata.get('entanglement', 0.0))
        stats['purity_distribution'].append(metadata.get('purity', 1.0))
    
    # Compute summary statistics
    stats['avg_entanglement'] = np.mean(stats['entanglement_distribution'])
    stats['avg_purity'] = np.mean(stats['purity_distribution'])
    stats['entanglement_std'] = np.std(stats['entanglement_distribution'])
    stats['purity_std'] = np.std(stats['purity_distribution'])
    
    return stats

def save_dataset(dataset: Dataset, filepath: str):
    """Save quantum dataset to file"""
    if hasattr(dataset, 'states'):
        # For custom quantum datasets
        data = {
            'states': dataset.states,
            'labels': dataset.labels if hasattr(dataset, 'labels') else None,
            'metadata': dataset.metadata if hasattr(dataset, 'metadata') else None,
            'config': dataset.config if hasattr(dataset, 'config') else None
        }
    else:
        # For general datasets
        data = {
            'dataset': dataset
        }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_dataset(filepath: str) -> Dataset:
    """Load quantum dataset from file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if 'states' in data:
        # Reconstruct custom quantum dataset
        if 'config' in data and data['config'] is not None:
            dataset = QuantumStateDataset.__new__(QuantumStateDataset)
            dataset.config = data['config']
            dataset.n_qubits = data['config'].n_qubits
            dataset.dim = 2 ** dataset.n_qubits
            dataset.states = data['states']
            dataset.labels = data['labels']
            dataset.metadata = data['metadata']
            return dataset
        else:
            raise ValueError("Cannot reconstruct dataset without config")
    else:
        return data['dataset']