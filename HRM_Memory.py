import torch
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional


class HRMMemory:
    """
    Memory buffer for storing HRM training experiences.
    Adapted from replay buffer concepts for HR learning scenarios.
    """
    
    def __init__(self, buffer_size: int = 10000, batch_size: int = 64):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Experience buffer
        self.buffer = deque(maxlen=buffer_size)
        
        # Episode buffer for multi-step experiences
        self.episode_buffer = []
        
        # Priority weights for experience replay
        self.priorities = deque(maxlen=buffer_size)
        self.max_priority = 1.0
        
        # Statistics
        self.total_added = 0
        
    def store_transition(self, transition: Dict):
        """Store a single transition"""
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)
        self.total_added += 1
    
    def store_episode(self, episode_transitions: List[Dict]):
        """Store an entire episode of transitions"""
        for transition in episode_transitions:
            self.store_transition(transition)
    
    def can_sample(self) -> bool:
        """Check if buffer has enough experiences to sample"""
        return len(self.buffer) >= self.batch_size
    
    def sample(self, priority_sampling: bool = False) -> List[Dict]:
        """Sample a batch of experiences"""
        if not self.can_sample():
            raise ValueError("Not enough experiences in buffer to sample")
        
        if priority_sampling and len(self.priorities) == len(self.buffer):
            # Priority-based sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), self.batch_size, p=probabilities)
            batch = [self.buffer[i] for i in indices]
        else:
            # Uniform random sampling
            batch = random.sample(self.buffer, self.batch_size)
        
        return batch
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for priority experience replay"""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def get_size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.priorities.clear()
        self.episode_buffer.clear()
        self.total_added = 0
        self.max_priority = 1.0
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics"""
        return {
            'buffer_size': len(self.buffer),
            'max_buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'total_added': self.total_added,
            'utilization': len(self.buffer) / self.buffer_size
        }


class HRMTrajectoryBuffer:
    """
    Specialized buffer for storing complete trajectories.
    Used for algorithms like PPO that need full episode data.
    """
    
    def __init__(self, max_trajectories: int = 1000):
        self.max_trajectories = max_trajectories
        self.trajectories = deque(maxlen=max_trajectories)
        
    def store_trajectory(self, trajectory: List[Dict]):
        """Store a complete trajectory"""
        self.trajectories.append(trajectory)
    
    def sample_trajectories(self, num_trajectories: int) -> List[List[Dict]]:
        """Sample multiple trajectories"""
        if len(self.trajectories) < num_trajectories:
            return list(self.trajectories)
        
        return random.sample(self.trajectories, num_trajectories)
    
    def get_all_trajectories(self) -> List[List[Dict]]:
        """Get all stored trajectories"""
        return list(self.trajectories)
    
    def clear(self):
        """Clear all trajectories"""
        self.trajectories.clear()
    
    def get_size(self) -> int:
        """Get number of stored trajectories"""
        return len(self.trajectories)


class HRMHindsightBuffer:
    """
    Hindsight Experience Replay buffer for HRM.
    Stores experiences with modified goals for better learning.
    """
    
    def __init__(self, buffer_size: int = 10000, batch_size: int = 64, k: int = 4):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.k = k  # Number of additional goals to sample
        
        self.buffer = deque(maxlen=buffer_size)
        self.episode_buffer = []
        
    def store_episode(self, episode_transitions: List[Dict], goals: List[Dict]):
        """
        Store episode with hindsight experience replay.
        
        Args:
            episode_transitions: List of transitions in the episode
            goals: List of possible goals that could be achieved
        """
        # Store original episode
        for transition in episode_transitions:
            self.buffer.append(transition)
        
        # Generate hindsight experiences
        for transition in episode_transitions:
            # Sample k additional goals
            sampled_goals = random.sample(goals, min(self.k, len(goals)))
            
            for goal in sampled_goals:
                # Create modified transition with new goal
                modified_transition = transition.copy()
                modified_transition['goal'] = goal
                
                # Recompute reward based on new goal
                modified_transition['reward'] = self._compute_hindsight_reward(
                    transition, goal
                )
                
                self.buffer.append(modified_transition)
    
    def _compute_hindsight_reward(self, transition: Dict, goal: Dict) -> float:
        """Compute reward for hindsight experience"""
        # Simplified reward computation based on HR metrics
        current_metrics = transition['hr_metrics']
        
        # Example: Goal is to achieve certain satisfaction level
        if 'target_satisfaction' in goal:
            target = goal['target_satisfaction']
            achieved = current_metrics.employee_satisfaction
            reward = -abs(target - achieved)  # Negative distance
        else:
            reward = transition['reward']  # Use original reward
        
        return reward
    
    def sample(self) -> List[Dict]:
        """Sample batch of experiences"""
        if len(self.buffer) < self.batch_size:
            return list(self.buffer)
        
        return random.sample(self.buffer, self.batch_size)
    
    def can_sample(self) -> bool:
        """Check if buffer can provide samples"""
        return len(self.buffer) >= self.batch_size


class HRMMetaLearningBuffer:
    """
    Buffer for meta-learning in HRM context.
    Stores experiences from multiple organizations/scenarios.
    """
    
    def __init__(self, buffer_size: int = 50000, num_tasks: int = 10):
        self.buffer_size = buffer_size
        self.num_tasks = num_tasks
        
        # Task-specific buffers
        self.task_buffers = {}
        for task_id in range(num_tasks):
            self.task_buffers[task_id] = deque(maxlen=buffer_size // num_tasks)
        
        # Meta-learning specific storage
        self.support_sets = {}
        self.query_sets = {}
        
    def store_task_experience(self, task_id: int, transition: Dict):
        """Store experience for specific task"""
        if task_id not in self.task_buffers:
            self.task_buffers[task_id] = deque(maxlen=self.buffer_size // self.num_tasks)
        
        self.task_buffers[task_id].append(transition)
    
    def sample_task_batch(self, task_id: int, batch_size: int) -> List[Dict]:
        """Sample batch from specific task"""
        if task_id not in self.task_buffers:
            return []
        
        task_buffer = self.task_buffers[task_id]
        sample_size = min(batch_size, len(task_buffer))
        
        return random.sample(task_buffer, sample_size)
    
    def sample_meta_batch(self, num_tasks: int, support_size: int, query_size: int) -> Dict:
        """Sample meta-learning batch"""
        # Sample random tasks
        available_tasks = list(self.task_buffers.keys())
        sampled_tasks = random.sample(available_tasks, min(num_tasks, len(available_tasks)))
        
        meta_batch = {
            'support_sets': {},
            'query_sets': {}
        }
        
        for task_id in sampled_tasks:
            task_buffer = self.task_buffers[task_id]
            
            if len(task_buffer) >= support_size + query_size:
                # Sample support and query sets
                all_samples = random.sample(task_buffer, support_size + query_size)
                
                meta_batch['support_sets'][task_id] = all_samples[:support_size]
                meta_batch['query_sets'][task_id] = all_samples[support_size:]
        
        return meta_batch
    
    def get_task_statistics(self) -> Dict:
        """Get statistics for all tasks"""
        stats = {}
        for task_id, buffer in self.task_buffers.items():
            stats[task_id] = {
                'size': len(buffer),
                'max_size': buffer.maxlen
            }
        
        return stats


class HRMCuriosityBuffer:
    """
    Buffer that prioritizes novel or surprising experiences.
    Useful for exploration in HR scenarios.
    """
    
    def __init__(self, buffer_size: int = 10000, batch_size: int = 64):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        self.buffer = deque(maxlen=buffer_size)
        self.novelty_scores = deque(maxlen=buffer_size)
        
        # For computing novelty
        self.state_embeddings = []
        self.embedding_dim = 128
        
    def store_transition(self, transition: Dict, novelty_score: float = None):
        """Store transition with novelty score"""
        self.buffer.append(transition)
        
        if novelty_score is None:
            novelty_score = self._compute_novelty(transition)
        
        self.novelty_scores.append(novelty_score)
    
    def _compute_novelty(self, transition: Dict) -> float:
        """Compute novelty score for transition"""
        # Simple novelty computation based on state uniqueness
        state = transition['state']
        
        if len(self.state_embeddings) == 0:
            return 1.0  # First state is novel
        
        # Compute distances to previous states
        state_tensor = state.cpu().numpy() if hasattr(state, 'cpu') else np.array(state)
        
        min_distance = float('inf')
        for prev_embedding in self.state_embeddings[-100:]:  # Check last 100 states
            distance = np.linalg.norm(state_tensor - prev_embedding)
            min_distance = min(min_distance, distance)
        
        # Store state embedding
        self.state_embeddings.append(state_tensor)
        if len(self.state_embeddings) > 1000:  # Keep only recent states
            self.state_embeddings.pop(0)
        
        return min_distance
    
    def sample(self, novelty_bias: float = 0.5) -> List[Dict]:
        """Sample with bias towards novel experiences"""
        if len(self.buffer) < self.batch_size:
            return list(self.buffer)
        
        # Create sampling probabilities based on novelty
        novelty_array = np.array(self.novelty_scores)
        
        # Combine uniform and novelty-based sampling
        uniform_probs = np.ones(len(self.buffer)) / len(self.buffer)
        novelty_probs = novelty_array / novelty_array.sum()
        
        combined_probs = (1 - novelty_bias) * uniform_probs + novelty_bias * novelty_probs
        
        # Sample based on combined probabilities
        indices = np.random.choice(len(self.buffer), self.batch_size, p=combined_probs)
        
        return [self.buffer[i] for i in indices]
    
    def can_sample(self) -> bool:
        """Check if buffer can provide samples"""
        return len(self.buffer) >= self.batch_size