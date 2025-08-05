"""
Quantum Reinforcement Learning Environment for HRM

This module implements quantum environments for reinforcement learning
with quantum state learning and Heisenberg scaling accuracy.

Key Components:
- QuantumEnvironment: Main RL environment for quantum state learning
- QuantumStateSpace: Quantum state space representation
- QuantumActionSpace: Quantum operations and measurements
- QuantumRewardFunction: Reward functions for quantum learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import random

from .quantum_states import (
    QuantumStateRepresentation,
    QuantumStateConfig,
    create_ghz_state,
    create_spin_squeezed_state,
    quantum_fidelity,
    von_neumann_entropy
)

@dataclass
class QuantumEnvironmentConfig:
    """Configuration for quantum RL environment"""
    n_qubits: int = 4
    max_episodes: int = 1000
    max_steps_per_episode: int = 50
    target_fidelity: float = 0.99
    heisenberg_scaling_weight: float = 1.0
    entanglement_weight: float = 0.5
    measurement_noise: float = 0.01
    decoherence_rate: float = 0.001
    reward_shaping: bool = True
    curriculum_learning: bool = True
    
class QuantumStateSpace:
    """Quantum state space for RL environment"""
    
    def __init__(self, config: QuantumEnvironmentConfig):
        self.config = config
        self.n_qubits = config.n_qubits
        self.dim = 2 ** self.n_qubits
        
        # State space bounds (amplitude magnitudes and phases)
        self.amplitude_bounds = (0.0, 1.0)
        self.phase_bounds = (-np.pi, np.pi)
        
        # Gym space representation
        # Real and imaginary parts of quantum amplitudes
        low = np.full(self.dim * 2, -1.0)
        high = np.full(self.dim * 2, 1.0)
        self.space = spaces.Box(low=low, high=high, dtype=np.float32)
        
    def sample(self) -> torch.Tensor:
        """Sample a random quantum state"""
        # Generate random amplitudes
        amplitudes = torch.randn(self.dim, dtype=torch.complex64)
        
        # Normalize to valid quantum state
        amplitudes = F.normalize(amplitudes, p=2, dim=0)
        
        return amplitudes
    
    def encode_state(self, quantum_state: torch.Tensor) -> np.ndarray:
        """Encode quantum state for RL agent"""
        # Convert complex amplitudes to real vector
        real_part = torch.real(quantum_state).numpy()
        imag_part = torch.imag(quantum_state).numpy()
        
        return np.concatenate([real_part, imag_part])
    
    def decode_state(self, encoded_state: np.ndarray) -> torch.Tensor:
        """Decode RL state to quantum state"""
        mid = len(encoded_state) // 2
        real_part = torch.tensor(encoded_state[:mid], dtype=torch.float32)
        imag_part = torch.tensor(encoded_state[mid:], dtype=torch.float32)
        
        # Create complex quantum state
        quantum_state = torch.complex(real_part, imag_part)
        
        # Normalize
        quantum_state = F.normalize(quantum_state, p=2, dim=0)
        
        return quantum_state

class QuantumActionSpace:
    """Quantum action space for RL environment"""
    
    def __init__(self, config: QuantumEnvironmentConfig):
        self.config = config
        self.n_qubits = config.n_qubits
        
        # Available quantum operations
        self.operations = [
            'pauli_x', 'pauli_y', 'pauli_z',
            'hadamard', 'rotation_x', 'rotation_y', 'rotation_z',
            'cnot', 'phase', 'measurement'
        ]
        
        # Action space: [operation_type, target_qubit, control_qubit, angle]
        # operation_type: discrete choice of operations
        # target_qubit: which qubit to apply operation
        # control_qubit: control qubit for two-qubit gates
        # angle: rotation angle for parametric gates
        
        self.space = spaces.Dict({
            'operation': spaces.Discrete(len(self.operations)),
            'target_qubit': spaces.Discrete(self.n_qubits),
            'control_qubit': spaces.Discrete(self.n_qubits),
            'angle': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        })
    
    def sample(self) -> Dict[str, Union[int, float]]:
        """Sample a random quantum action"""
        return {
            'operation': random.randint(0, len(self.operations) - 1),
            'target_qubit': random.randint(0, self.n_qubits - 1),
            'control_qubit': random.randint(0, self.n_qubits - 1),
            'angle': np.random.uniform(-np.pi, np.pi)
        }
    
    def apply_action(self, quantum_state: torch.Tensor, action: Dict[str, Union[int, float]]) -> torch.Tensor:
        """Apply quantum action to state"""
        operation = self.operations[action['operation']]
        target = action['target_qubit']
        control = action['control_qubit']
        angle = action['angle']
        
        new_state = quantum_state.clone()
        
        if operation == 'pauli_x':
            new_state = self._apply_pauli_x(new_state, target)
        elif operation == 'pauli_y':
            new_state = self._apply_pauli_y(new_state, target)
        elif operation == 'pauli_z':
            new_state = self._apply_pauli_z(new_state, target)
        elif operation == 'hadamard':
            new_state = self._apply_hadamard(new_state, target)
        elif operation == 'rotation_x':
            new_state = self._apply_rotation_x(new_state, target, angle)
        elif operation == 'rotation_y':
            new_state = self._apply_rotation_y(new_state, target, angle)
        elif operation == 'rotation_z':
            new_state = self._apply_rotation_z(new_state, target, angle)
        elif operation == 'cnot':
            new_state = self._apply_cnot(new_state, control, target)
        elif operation == 'phase':
            new_state = self._apply_phase(new_state, target, angle)
        elif operation == 'measurement':
            # For measurement, we perform a projective measurement
            new_state = self._apply_measurement(new_state, target)
        
        return new_state
    
    def _apply_pauli_x(self, state: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply Pauli-X gate"""
        new_state = state.clone()
        dim = len(state)
        
        for i in range(dim):
            if (i >> qubit) & 1 == 0:  # qubit is |0⟩
                j = i | (1 << qubit)   # flip to |1⟩
                new_state[i], new_state[j] = state[j], state[i]
        
        return new_state
    
    def _apply_pauli_y(self, state: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply Pauli-Y gate"""
        new_state = state.clone()
        dim = len(state)
        
        for i in range(dim):
            if (i >> qubit) & 1 == 0:  # qubit is |0⟩
                j = i | (1 << qubit)   # flip to |1⟩
                new_state[i] = -1j * state[j]
                new_state[j] = 1j * state[i]
        
        return new_state
    
    def _apply_pauli_z(self, state: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply Pauli-Z gate"""
        new_state = state.clone()
        dim = len(state)
        
        for i in range(dim):
            if (i >> qubit) & 1 == 1:  # qubit is |1⟩
                new_state[i] = -state[i]
        
        return new_state
    
    def _apply_hadamard(self, state: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply Hadamard gate"""
        new_state = state.clone()
        dim = len(state)
        sqrt2_inv = 1.0 / math.sqrt(2)
        
        for i in range(dim):
            if (i >> qubit) & 1 == 0:  # qubit is |0⟩
                j = i | (1 << qubit)   # corresponding |1⟩
                new_state[i] = sqrt2_inv * (state[i] + state[j])
                new_state[j] = sqrt2_inv * (state[i] - state[j])
        
        return new_state
    
    def _apply_rotation_x(self, state: torch.Tensor, qubit: int, angle: float) -> torch.Tensor:
        """Apply rotation around X-axis"""
        cos_half = math.cos(angle / 2)
        sin_half = math.sin(angle / 2)
        
        new_state = state.clone()
        dim = len(state)
        
        for i in range(dim):
            if (i >> qubit) & 1 == 0:  # qubit is |0⟩
                j = i | (1 << qubit)   # corresponding |1⟩
                new_state[i] = cos_half * state[i] - 1j * sin_half * state[j]
                new_state[j] = cos_half * state[j] - 1j * sin_half * state[i]
        
        return new_state
    
    def _apply_rotation_y(self, state: torch.Tensor, qubit: int, angle: float) -> torch.Tensor:
        """Apply rotation around Y-axis"""
        cos_half = math.cos(angle / 2)
        sin_half = math.sin(angle / 2)
        
        new_state = state.clone()
        dim = len(state)
        
        for i in range(dim):
            if (i >> qubit) & 1 == 0:  # qubit is |0⟩
                j = i | (1 << qubit)   # corresponding |1⟩
                new_state[i] = cos_half * state[i] - sin_half * state[j]
                new_state[j] = cos_half * state[j] + sin_half * state[i]
        
        return new_state
    
    def _apply_rotation_z(self, state: torch.Tensor, qubit: int, angle: float) -> torch.Tensor:
        """Apply rotation around Z-axis"""
        new_state = state.clone()
        dim = len(state)
        
        for i in range(dim):
            if (i >> qubit) & 1 == 0:  # qubit is |0⟩
                new_state[i] = state[i] * torch.exp(-1j * angle / 2)
            else:  # qubit is |1⟩
                new_state[i] = state[i] * torch.exp(1j * angle / 2)
        
        return new_state
    
    def _apply_cnot(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        """Apply CNOT gate"""
        if control == target:
            return state  # No-op if control == target
        
        new_state = state.clone()
        dim = len(state)
        
        for i in range(dim):
            if (i >> control) & 1:  # control qubit is |1⟩
                j = i ^ (1 << target)  # flip target qubit
                new_state[i] = state[j]
                new_state[j] = state[i]
        
        return new_state
    
    def _apply_phase(self, state: torch.Tensor, qubit: int, angle: float) -> torch.Tensor:
        """Apply phase gate"""
        new_state = state.clone()
        dim = len(state)
        
        for i in range(dim):
            if (i >> qubit) & 1:  # qubit is |1⟩
                new_state[i] = state[i] * torch.exp(1j * angle)
        
        return new_state
    
    def _apply_measurement(self, state: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply projective measurement (simplified)"""
        # For RL, we simulate measurement by partial collapse
        # This is a simplified version - in reality measurement is probabilistic
        
        new_state = state.clone()
        dim = len(state)
        
        # Calculate probabilities for |0⟩ and |1⟩
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i in range(dim):
            if (i >> qubit) & 1 == 0:
                prob_0 += abs(state[i]) ** 2
            else:
                prob_1 += abs(state[i]) ** 2
        
        # Collapse to most probable state (deterministic for RL)
        if prob_0 > prob_1:
            # Collapse to |0⟩
            for i in range(dim):
                if (i >> qubit) & 1 == 1:
                    new_state[i] = 0.0
        else:
            # Collapse to |1⟩
            for i in range(dim):
                if (i >> qubit) & 1 == 0:
                    new_state[i] = 0.0
        
        # Renormalize
        new_state = F.normalize(new_state, p=2, dim=0)
        
        return new_state

class QuantumRewardFunction:
    """Reward function for quantum RL environment"""
    
    def __init__(self, config: QuantumEnvironmentConfig):
        self.config = config
        self.target_states = self._initialize_target_states()
        
    def _initialize_target_states(self) -> Dict[str, torch.Tensor]:
        """Initialize target quantum states"""
        n_qubits = self.config.n_qubits
        
        targets = {
            'ghz': create_ghz_state(n_qubits),
            'spin_squeezed': create_spin_squeezed_state(n_qubits, 0.5),
            'bell': self._create_bell_state(),
            'uniform_superposition': self._create_uniform_superposition()
        }
        
        return targets
    
    def _create_bell_state(self) -> torch.Tensor:
        """Create Bell state (for 2+ qubits)"""
        dim = 2 ** self.config.n_qubits
        bell_state = torch.zeros(dim, dtype=torch.complex64)
        bell_state[0] = 1.0 / math.sqrt(2)  # |00...0⟩
        bell_state[3] = 1.0 / math.sqrt(2)  # |11...1⟩ (simplified)
        return bell_state
    
    def _create_uniform_superposition(self) -> torch.Tensor:
        """Create uniform superposition state"""
        dim = 2 ** self.config.n_qubits
        return torch.ones(dim, dtype=torch.complex64) / math.sqrt(dim)
    
    def compute_reward(self, current_state: torch.Tensor, action: Dict, target_type: str = 'ghz') -> float:
        """Compute reward for current state and action"""
        target_state = self.target_states[target_type]
        
        # Primary reward: fidelity with target state
        fidelity = quantum_fidelity(current_state.unsqueeze(0), target_state.unsqueeze(0)).item()
        fidelity_reward = fidelity * 10.0  # Scale up
        
        # Entanglement reward
        entanglement = self._compute_entanglement(current_state)
        entanglement_reward = entanglement * self.config.entanglement_weight
        
        # Heisenberg scaling reward (based on quantum Fisher information)
        qfi_reward = self._compute_qfi_reward(current_state)
        heisenberg_reward = qfi_reward * self.config.heisenberg_scaling_weight
        
        # Action efficiency penalty (discourage too many gates)
        action_penalty = -0.01  # Small penalty per action
        
        # Decoherence penalty
        decoherence_penalty = -self.config.decoherence_rate * self._compute_decoherence(current_state)
        
        # Total reward
        total_reward = (fidelity_reward + 
                       entanglement_reward + 
                       heisenberg_reward + 
                       action_penalty + 
                       decoherence_penalty)
        
        return total_reward
    
    def _compute_entanglement(self, state: torch.Tensor) -> float:
        """Compute entanglement measure"""
        # Simplified entanglement measure using von Neumann entropy
        try:
            # Create density matrix
            rho = torch.outer(state, torch.conj(state))
            
            # Compute partial trace for first qubit (simplified)
            n_qubits = self.config.n_qubits
            if n_qubits > 1:
                dim_subsystem = 2
                dim_environment = 2 ** (n_qubits - 1)
                
                rho_reduced = torch.zeros(dim_subsystem, dim_subsystem, dtype=torch.complex64)
                
                for i in range(dim_subsystem):
                    for j in range(dim_subsystem):
                        for k in range(dim_environment):
                            idx1 = i * dim_environment + k
                            idx2 = j * dim_environment + k
                            rho_reduced[i, j] += rho[idx1, idx2]
                
                # Compute von Neumann entropy
                eigenvals = torch.real(torch.linalg.eigvals(rho_reduced))
                eigenvals = eigenvals[eigenvals > 1e-12]
                entropy = -torch.sum(eigenvals * torch.log2(eigenvals + 1e-12))
                
                return entropy.item()
            else:
                return 0.0
        except:
            return 0.0
    
    def _compute_qfi_reward(self, state: torch.Tensor) -> float:
        """Compute reward based on quantum Fisher information"""
        # Simplified QFI estimation
        # In practice, this would involve computing derivatives of the state
        
        # Use state norm and phase properties as proxy
        amplitudes = torch.abs(state)
        phases = torch.angle(state)
        
        # Reward states with good amplitude distribution for parameter estimation
        amplitude_variance = torch.var(amplitudes).item()
        phase_variance = torch.var(phases).item()
        
        # Higher variance in amplitudes and phases generally leads to higher QFI
        qfi_proxy = amplitude_variance + phase_variance
        
        return qfi_proxy
    
    def _compute_decoherence(self, state: torch.Tensor) -> float:
        """Compute decoherence measure"""
        # Simple measure: deviation from pure state
        purity = torch.sum(torch.abs(state) ** 4).item()
        max_purity = 1.0  # For pure states
        
        decoherence = max_purity - purity
        return decoherence

class QuantumEnvironment(gym.Env):
    """
    Quantum Reinforcement Learning Environment
    
    This environment allows RL agents to learn quantum state preparation
    and manipulation for achieving Heisenberg scaling accuracy.
    """
    
    def __init__(self, config: QuantumEnvironmentConfig):
        super().__init__()
        self.config = config
        
        # Initialize spaces
        self.state_space = QuantumStateSpace(config)
        self.action_space_handler = QuantumActionSpace(config)
        self.reward_function = QuantumRewardFunction(config)
        
        # Gym spaces
        self.observation_space = self.state_space.space
        self.action_space = self.action_space_handler.space
        
        # Environment state
        self.current_quantum_state = None
        self.episode_step = 0
        self.episode_count = 0
        self.target_type = 'ghz'  # Default target
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_fidelities = []
        self.best_fidelity = 0.0
        
        # Curriculum learning
        self.difficulty_level = 0
        self.success_count = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Initialize with random quantum state
        self.current_quantum_state = self.state_space.sample()
        self.episode_step = 0
        
        # Curriculum learning: adjust difficulty
        if self.config.curriculum_learning:
            self._update_curriculum()
        
        # Choose target based on difficulty
        self._select_target()
        
        # Return encoded state
        return self.state_space.encode_state(self.current_quantum_state)
    
    def step(self, action: Dict[str, Union[int, float]]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        # Apply quantum action
        new_quantum_state = self.action_space_handler.apply_action(
            self.current_quantum_state, action
        )
        
        # Add noise (decoherence simulation)
        if self.config.measurement_noise > 0:
            noise = torch.randn_like(new_quantum_state) * self.config.measurement_noise
            new_quantum_state = new_quantum_state + noise
            new_quantum_state = F.normalize(new_quantum_state, p=2, dim=0)
        
        self.current_quantum_state = new_quantum_state
        
        # Compute reward
        reward = self.reward_function.compute_reward(
            self.current_quantum_state, action, self.target_type
        )
        
        # Check termination conditions
        done = self._check_termination()
        
        # Update episode tracking
        self.episode_step += 1
        
        # Prepare info
        info = self._get_info()
        
        # Return observation, reward, done, info
        observation = self.state_space.encode_state(self.current_quantum_state)
        return observation, reward, done, info
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if max steps reached
        if self.episode_step >= self.config.max_steps_per_episode:
            return True
        
        # Terminate if target fidelity achieved
        target_state = self.reward_function.target_states[self.target_type]
        fidelity = quantum_fidelity(
            self.current_quantum_state.unsqueeze(0), 
            target_state.unsqueeze(0)
        ).item()
        
        if fidelity >= self.config.target_fidelity:
            self.success_count += 1
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info"""
        target_state = self.reward_function.target_states[self.target_type]
        fidelity = quantum_fidelity(
            self.current_quantum_state.unsqueeze(0), 
            target_state.unsqueeze(0)
        ).item()
        
        entanglement = self.reward_function._compute_entanglement(self.current_quantum_state)
        qfi_proxy = self.reward_function._compute_qfi_reward(self.current_quantum_state)
        
        return {
            'fidelity': fidelity,
            'entanglement': entanglement,
            'qfi_proxy': qfi_proxy,
            'episode_step': self.episode_step,
            'target_type': self.target_type,
            'difficulty_level': self.difficulty_level,
            'quantum_state_norm': torch.norm(self.current_quantum_state).item()
        }
    
    def _update_curriculum(self):
        """Update curriculum difficulty"""
        # Increase difficulty if success rate is high
        if self.episode_count > 0 and self.episode_count % 100 == 0:
            success_rate = self.success_count / 100
            
            if success_rate > 0.8 and self.difficulty_level < 3:
                self.difficulty_level += 1
                self.success_count = 0
            elif success_rate < 0.3 and self.difficulty_level > 0:
                self.difficulty_level -= 1
                self.success_count = 0
    
    def _select_target(self):
        """Select target state based on difficulty"""
        targets = ['uniform_superposition', 'bell', 'ghz', 'spin_squeezed']
        self.target_type = targets[min(self.difficulty_level, len(targets) - 1)]
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            target_state = self.reward_function.target_states[self.target_type]
            fidelity = quantum_fidelity(
                self.current_quantum_state.unsqueeze(0), 
                target_state.unsqueeze(0)
            ).item()
            
            print(f"Episode {self.episode_count}, Step {self.episode_step}")
            print(f"Target: {self.target_type}")
            print(f"Current Fidelity: {fidelity:.4f}")
            print(f"Quantum State Amplitudes: {self.current_quantum_state}")
            print("-" * 50)
    
    def close(self):
        """Close environment"""
        pass
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary"""
        return {
            'episode_count': self.episode_count,
            'success_count': self.success_count,
            'success_rate': self.success_count / max(self.episode_count, 1),
            'best_fidelity': self.best_fidelity,
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'difficulty_level': self.difficulty_level
        }

# Factory function for creating quantum environments
def create_quantum_environment(task_type: str = 'state_preparation', 
                             n_qubits: int = 4,
                             **kwargs) -> QuantumEnvironment:
    """Create quantum environment for specific task"""
    
    config = QuantumEnvironmentConfig(n_qubits=n_qubits, **kwargs)
    
    if task_type == 'state_preparation':
        # Standard state preparation environment
        return QuantumEnvironment(config)
    elif task_type == 'metrology':
        # Quantum metrology environment
        config.heisenberg_scaling_weight = 2.0
        config.target_fidelity = 0.995
        return QuantumEnvironment(config)
    elif task_type == 'tomography':
        # Quantum state tomography environment
        config.measurement_noise = 0.05
        config.max_steps_per_episode = 100
        return QuantumEnvironment(config)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

# Utility functions for quantum environment analysis
def analyze_quantum_state(state: torch.Tensor) -> Dict[str, float]:
    """Analyze properties of quantum state"""
    # Compute various quantum properties
    purity = torch.sum(torch.abs(state) ** 4).item()
    
    # Participation ratio (measure of localization)
    participation_ratio = 1.0 / torch.sum(torch.abs(state) ** 4).item()
    
    # Shannon entropy of amplitude distribution
    amplitudes = torch.abs(state) ** 2
    shannon_entropy = -torch.sum(amplitudes * torch.log2(amplitudes + 1e-12)).item()
    
    return {
        'purity': purity,
        'participation_ratio': participation_ratio,
        'shannon_entropy': shannon_entropy,
        'max_amplitude': torch.max(torch.abs(state)).item(),
        'mean_amplitude': torch.mean(torch.abs(state)).item()
    }