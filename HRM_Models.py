import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class HRMLearningAutomata(nn.Module):
    """
    Learning Automata for parameter adjustment in HRM context.
    Adapted from quantum parameter optimization for HR strategy optimization.
    """
    
    def __init__(self, num_actions: int, learning_rate: float = 0.1):
        super().__init__()
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.action_probabilities = torch.ones(num_actions) / num_actions
        self.action_rewards = torch.zeros(num_actions)
        self.action_counts = torch.zeros(num_actions)
        
    def select_action(self) -> int:
        """Select action based on current probability distribution"""
        return torch.multinomial(self.action_probabilities, 1).item()
    
    def update(self, action: int, reward: float):
        """Update action probabilities based on received reward"""
        self.action_counts[action] += 1
        self.action_rewards[action] += reward
        
        # Update probability based on average reward
        avg_reward = self.action_rewards[action] / self.action_counts[action]
        
        # Increase probability for good actions, decrease for bad ones
        if reward > 0:
            self.action_probabilities[action] += self.learning_rate * (1 - self.action_probabilities[action])
        else:
            self.action_probabilities[action] *= (1 - self.learning_rate)
        
        # Normalize probabilities
        self.action_probabilities = self.action_probabilities / self.action_probabilities.sum()


class HRMActor(nn.Module):
    """
    Actor network for HRM decision making.
    Adapted from quantum state preparation for HR strategy selection.
    """
    
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        
        # Input features: HR metrics, employee statistics, market conditions
        self.input_size = model_params.get('input_size', 52)  # From HRM environment
        
        # HR strategy spaces
        self.hr_strategies = model_params.get('hr_strategies', {
            'recruitment': 5,
            'training': 8, 
            'compensation': 6,
            'culture': 6,
            'performance': 5,
            'leadership': 5,
            'technology': 4,
            'restructuring': 4
        })
        
        self.strategy_sizes = list(self.hr_strategies.values())
        self.total_strategies = sum(self.strategy_sizes)
        
        # Network architecture - deeper for complex HR relationships
        self.dim1 = model_params.get('hidden_dim1', 128)
        self.dim2 = model_params.get('hidden_dim2', 256)
        self.dim3 = model_params.get('hidden_dim3', 128)
        self.dimf = model_params.get('final_dim', 64)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_size, self.dim1),
            nn.BatchNorm1d(self.dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(self.dim1, self.dim2),
            nn.BatchNorm1d(self.dim2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(self.dim2, self.dim3),
            nn.BatchNorm1d(self.dim3),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Strategy-specific heads
        self.strategy_heads = nn.ModuleDict()
        for strategy, size in self.hr_strategies.items():
            self.strategy_heads[strategy] = nn.Sequential(
                nn.Linear(self.dim3, self.dimf),
                nn.ReLU(),
                nn.Linear(self.dimf, size)
            )
        
        # Combined policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.dim3, self.dimf),
            nn.ReLU(),
            nn.Linear(self.dimf, self.total_strategies)
        )
        
        # Value estimation for actor-critic
        self.value_head = nn.Sequential(
            nn.Linear(self.dim3, self.dimf),
            nn.ReLU(),
            nn.Linear(self.dimf, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            state: Current HR state tensor
            
        Returns:
            Dictionary containing policy probabilities and value estimate
        """
        batch_size = state.size(0)
        
        # Extract features
        features = self.feature_extractor(state)
        
        # Get policy logits
        policy_logits = self.policy_head(features)
        policy_probs = F.softmax(policy_logits, dim=1)
        
        # Get value estimate
        value = self.value_head(features)
        
        # Get strategy-specific probabilities
        strategy_probs = {}
        for strategy, head in self.strategy_heads.items():
            strategy_logits = head(features)
            strategy_probs[strategy] = F.softmax(strategy_logits, dim=1)
        
        return {
            'policy_probs': policy_probs,
            'policy_logits': policy_logits,
            'value': value,
            'strategy_probs': strategy_probs,
            'features': features
        }
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Select action based on current policy
        
        Args:
            state: Current state
            deterministic: If True, select best action deterministically
            
        Returns:
            Selected action and log probability
        """
        with torch.no_grad():
            output = self.forward(state)
            policy_probs = output['policy_probs']
            
            if deterministic:
                action = torch.argmax(policy_probs, dim=1)
            else:
                action = torch.multinomial(policy_probs, 1).squeeze(1)
            
            log_prob = torch.log(policy_probs.gather(1, action.unsqueeze(1))).squeeze(1)
            
        return action, log_prob
    
    def get_action_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of given action in given state"""
        output = self.forward(state)
        policy_probs = output['policy_probs']
        log_probs = torch.log(policy_probs + 1e-8)
        return log_probs.gather(1, action.unsqueeze(1)).squeeze(1)


class HRMCritic(nn.Module):
    """
    Critic network for HR value estimation.
    Estimates the value of current HR state and predicts future performance.
    """
    
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        
        self.input_size = model_params.get('input_size', 52)
        
        # Network dimensions
        self.dim1 = model_params.get('critic_dim1', 128)
        self.dim2 = model_params.get('critic_dim2', 256) 
        self.dim3 = model_params.get('critic_dim3', 128)
        self.dimf = model_params.get('critic_final_dim', 64)
        
        # Main critic network
        self.critic_network = nn.Sequential(
            nn.Linear(self.input_size, self.dim1),
            nn.BatchNorm1d(self.dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(self.dim1, self.dim2),
            nn.BatchNorm1d(self.dim2),
            nn.ReLU(), 
            nn.Dropout(0.3),
            
            nn.Linear(self.dim2, self.dim3),
            nn.BatchNorm1d(self.dim3),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(self.dim3, self.dimf),
            nn.ReLU(),
            nn.Linear(self.dimf, 1)
        )
        
        # Multi-objective value heads for different HR outcomes
        self.outcome_heads = nn.ModuleDict({
            'employee_satisfaction': nn.Linear(self.dim3, 1),
            'productivity': nn.Linear(self.dim3, 1),
            'retention': nn.Linear(self.dim3, 1),
            'innovation': nn.Linear(self.dim3, 1),
            'culture': nn.Linear(self.dim3, 1)
        })
        
        # Feature layers for outcome prediction
        self.outcome_features = nn.Sequential(
            nn.Linear(self.input_size, self.dim1),
            nn.BatchNorm1d(self.dim1),
            nn.ReLU(),
            nn.Linear(self.dim1, self.dim2),
            nn.BatchNorm1d(self.dim2), 
            nn.ReLU(),
            nn.Linear(self.dim2, self.dim3),
            nn.ReLU()
        )
        
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to estimate state values
        
        Args:
            state: Current HR state
            
        Returns:
            Dictionary with value estimates
        """
        # Main value estimate
        main_value = self.critic_network(state)
        
        # Outcome-specific predictions
        outcome_features = self.outcome_features(state)
        outcome_values = {}
        
        for outcome, head in self.outcome_heads.items():
            outcome_values[outcome] = head(outcome_features)
            
        return {
            'main_value': main_value,
            'outcome_values': outcome_values,
            'features': outcome_features
        }


class HRMEvolutionStrategy(nn.Module):
    """
    Evolution Strategy for HRM optimization.
    Adapted from quantum evolution strategies for HR parameter optimization.
    """
    
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        
        # Population parameters
        self.population_size = model_params.get('population_size', 20)
        self.parameter_dim = model_params.get('parameter_dim', 10)
        
        # Strategy parameters
        self.sigma = nn.Parameter(torch.ones(self.parameter_dim) * 0.1)
        self.mu = nn.Parameter(torch.randn(self.parameter_dim))
        
        # Adaptation parameters
        self.tau = 1.0 / np.sqrt(2 * self.parameter_dim)
        self.tau_prime = 1.0 / np.sqrt(2 * np.sqrt(self.parameter_dim))
        
        # Fitness tracking
        self.fitness_history = []
        self.best_parameters = None
        self.best_fitness = float('-inf')
        
    def generate_population(self) -> torch.Tensor:
        """Generate population of HR strategy parameters"""
        # Global mutation
        global_factor = torch.randn(1) * self.tau_prime
        
        # Individual mutations
        individual_factors = torch.randn(self.parameter_dim) * self.tau
        
        # Update sigma
        new_sigma = self.sigma * torch.exp(global_factor + individual_factors)
        
        # Generate offspring
        population = []
        for _ in range(self.population_size):
            offspring = self.mu + new_sigma * torch.randn(self.parameter_dim)
            population.append(offspring)
            
        return torch.stack(population)
    
    def update_parameters(self, population: torch.Tensor, fitness_scores: torch.Tensor):
        """Update strategy parameters based on fitness"""
        # Select top performers
        top_k = max(1, self.population_size // 4)
        _, top_indices = torch.topk(fitness_scores, top_k)
        top_population = population[top_indices]
        
        # Update mean
        self.mu.data = torch.mean(top_population, dim=0)
        
        # Update best if improved
        best_idx = torch.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx].item()
            self.best_parameters = population[best_idx].clone()
        
        # Track fitness history
        self.fitness_history.append(torch.mean(fitness_scores).item())
        
    def get_best_strategy(self) -> torch.Tensor:
        """Get the best HR strategy found so far"""
        return self.best_parameters if self.best_parameters is not None else self.mu


class HRMPredictor(nn.Module):
    """
    Deep learning model for predicting HR outcomes.
    Predicts company revenue and customer satisfaction based on HR metrics.
    """
    
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        
        self.input_size = model_params.get('input_size', 52)
        self.num_revenue_classes = model_params.get('num_revenue_classes', 3)  # Low, Medium, High
        self.num_satisfaction_classes = model_params.get('num_satisfaction_classes', 5)  # 1-5 scale
        
        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Revenue prediction head
        self.revenue_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_revenue_classes)
        )
        
        # Customer satisfaction prediction head  
        self.satisfaction_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_satisfaction_classes)
        )
        
        # Regression heads for continuous predictions
        self.revenue_regression = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.satisfaction_regression = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict HR outcomes from current state
        
        Args:
            state: Current HR metrics and organizational state
            
        Returns:
            Dictionary with predictions
        """
        # Extract shared features
        features = self.shared_layers(state)
        
        # Classification predictions
        revenue_logits = self.revenue_head(features)
        satisfaction_logits = self.satisfaction_head(features)
        
        # Regression predictions
        revenue_value = self.revenue_regression(features)
        satisfaction_value = self.satisfaction_regression(features)
        
        return {
            'revenue_logits': revenue_logits,
            'revenue_probs': F.softmax(revenue_logits, dim=1),
            'satisfaction_logits': satisfaction_logits,
            'satisfaction_probs': F.softmax(satisfaction_logits, dim=1),
            'revenue_value': revenue_value,
            'satisfaction_value': satisfaction_value,
            'features': features
        }


class HRMMultiObjectiveOptimizer(nn.Module):
    """
    Multi-objective optimizer for HR decision making.
    Balances multiple HR objectives simultaneously.
    """
    
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        
        self.num_objectives = model_params.get('num_objectives', 5)
        self.input_size = model_params.get('input_size', 52)
        
        # Objective weights (learnable)
        self.objective_weights = nn.Parameter(torch.ones(self.num_objectives))
        
        # Pareto front approximation network
        self.pareto_network = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_objectives)
        )
        
        # Scalarization network for combining objectives
        self.scalarization_network = nn.Sequential(
            nn.Linear(self.num_objectives + self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Optimize multiple HR objectives
        
        Args:
            state: Current HR state
            
        Returns:
            Dictionary with multi-objective predictions
        """
        # Predict objective values
        objective_values = self.pareto_network(state)
        
        # Normalize weights
        normalized_weights = F.softmax(self.objective_weights, dim=0)
        
        # Weighted combination
        weighted_objectives = objective_values * normalized_weights.unsqueeze(0)
        
        # Scalarized value
        combined_input = torch.cat([state, objective_values], dim=1)
        scalarized_value = self.scalarization_network(combined_input)
        
        return {
            'objective_values': objective_values,
            'weighted_objectives': weighted_objectives,
            'scalarized_value': scalarized_value,
            'objective_weights': normalized_weights
        }