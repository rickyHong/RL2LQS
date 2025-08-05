import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
from torch.optim import Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.multiprocessing as mp
from multiprocessing.connection import wait
from logging import getLogger
from typing import Dict, List, Tuple, Optional

from HRM_Environment import HRMEnvironment
from HRM_Models import (
    HRMActor, HRMCritic, HRMEvolutionStrategy, 
    HRMPredictor, HRMMultiObjectiveOptimizer, HRMLearningAutomata
)
from HRM_Memory import HRMMemory
from utils.utils import get_result_folder, TimeEstimator


class HRMTrainer:
    """
    Human Resource Management Trainer using Deep Reinforcement Learning.
    Adapted from RL2LQS quantum learning framework for HR optimization.
    """
    
    def __init__(self,
                 env_params: Dict,
                 training_params: Dict,
                 model_params: Dict,
                 optimizer_params: Dict,
                 run_params: Dict):
        
        # Save parameters
        self.env_params = env_params
        self.training_params = training_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.run_params = run_params
        
        # Initialize logger and result folder
        self.logger = getLogger()
        self.result_folder = get_result_folder()
        
        # CUDA setup
        self.USE_CUDA = self.run_params['use_cuda']
        if self.USE_CUDA:
            self.device = torch.device('cuda', 0)
            torch.cuda.set_device(0)
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.DoubleTensor')
        
        # Training parameters
        self.batch_size = self.run_params['batch_size']
        self.episodes = self.run_params['episodes']
        self.learning_mode = self.training_params.get('learning_mode', 'actor_critic')  # 'actor_critic', 'ppo', 'evolution'
        
        # Initialize environment
        self.env = HRMEnvironment(**env_params)
        self.observation_size = self.env.observation_space_size
        self.action_size = self.env.action_space_size
        
        # Update model parameters with environment info
        self.model_params['input_size'] = self.observation_size
        self.model_params['action_size'] = self.action_size
        
        # Initialize models
        self._initialize_models()
        
        # Initialize optimizers
        self._initialize_optimizers()
        
        # Initialize memory
        self.memory = HRMMemory(
            buffer_size=self.training_params.get('buffer_size', 10000),
            batch_size=self.batch_size
        )
        
        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'hr_metrics': [],
            'loss_history': {},
            'prediction_accuracy': {'revenue': [], 'satisfaction': []},
            'policy_entropy': [],
            'value_estimates': []
        }
        
        # Best model tracking
        self.best_reward = float('-inf')
        self.best_model_state = None
        
        # Time estimator
        self.time_estimator = TimeEstimator()
        
    def _initialize_models(self):
        """Initialize all neural network models"""
        # Actor-Critic models
        self.actor = HRMActor(**self.model_params).to(self.device)
        self.critic = HRMCritic(**self.model_params).to(self.device)
        
        # Prediction models
        self.predictor = HRMPredictor(**self.model_params).to(self.device)
        
        # Multi-objective optimizer
        self.multi_objective_optimizer = HRMMultiObjectiveOptimizer(**self.model_params).to(self.device)
        
        # Evolution strategy (if using evolution mode)
        if self.learning_mode == 'evolution':
            self.evolution_strategy = HRMEvolutionStrategy(**self.model_params).to(self.device)
        
        # Learning automata for hyperparameter adjustment
        self.learning_automata = HRMLearningAutomata(
            num_actions=10,  # Different learning rate adjustments
            learning_rate=0.1
        )
        
        # Load pre-trained models if specified
        if self.model_params.get('load', False):
            self._load_models()
            
    def _initialize_optimizers(self):
        """Initialize optimizers for all models"""
        # Get optimizer parameters
        lr = self.optimizer_params['optimizer']['lr']
        weight_decay = self.optimizer_params['optimizer']['weight_decay']
        
        # Actor optimizer
        self.actor_optimizer = AdamW(
            self.actor.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Critic optimizer
        self.critic_optimizer = AdamW(
            self.critic.parameters(),
            lr=lr * 2,  # Critic learns faster
            weight_decay=weight_decay
        )
        
        # Predictor optimizer
        self.predictor_optimizer = AdamW(
            self.predictor.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Multi-objective optimizer
        self.multi_obj_optimizer = AdamW(
            self.multi_objective_optimizer.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate schedulers
        self.actor_scheduler = ReduceLROnPlateau(
            self.actor_optimizer, mode='max', patience=100, factor=0.8
        )
        self.critic_scheduler = ReduceLROnPlateau(
            self.critic_optimizer, mode='min', patience=100, factor=0.8
        )
        
    def run(self):
        """Main training loop"""
        self.logger.info("Starting HRM Training...")
        self.logger.info(f"Learning Mode: {self.learning_mode}")
        self.logger.info(f"Total Episodes: {self.episodes}")
        self.logger.info(f"Batch Size: {self.batch_size}")
        
        start_time = time.time()
        
        if self.learning_mode == 'actor_critic':
            self._run_actor_critic_training()
        elif self.learning_mode == 'ppo':
            self._run_ppo_training()
        elif self.learning_mode == 'evolution':
            self._run_evolution_training()
        else:
            raise ValueError(f"Unknown learning mode: {self.learning_mode}")
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final models and results
        self._save_models()
        self._save_training_results()
        
    def _run_actor_critic_training(self):
        """Actor-Critic training loop"""
        for episode in range(self.episodes):
            episode_start_time = time.time()
            
            # Reset environment
            state = self.env.reset()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            episode_reward = 0
            episode_length = 0
            episode_transitions = []
            
            # Collect episode data
            while True:
                # Select action
                with torch.no_grad():
                    actor_output = self.actor(state_tensor)
                    action_probs = actor_output['policy_probs']
                    value_estimate = actor_output['value']
                    
                    # Sample action
                    action_dist = Categorical(action_probs)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                
                # Take environment step
                next_state, reward, done, info = self.env.step(action.item())
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                
                # Store transition
                episode_transitions.append({
                    'state': state_tensor,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state_tensor,
                    'done': done,
                    'log_prob': log_prob,
                    'value': value_estimate,
                    'hr_metrics': info['current_metrics']
                })
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                    
                state_tensor = next_state_tensor
            
            # Store episode in memory
            self.memory.store_episode(episode_transitions)
            
            # Update models
            if self.memory.can_sample():
                losses = self._update_actor_critic()
                self._update_prediction_models()
                
                # Store loss history
                for loss_name, loss_value in losses.items():
                    if loss_name not in self.training_history['loss_history']:
                        self.training_history['loss_history'][loss_name] = []
                    self.training_history['loss_history'][loss_name].append(loss_value)
            
            # Record episode metrics
            self.training_history['episode_rewards'].append(episode_reward)
            if episode_transitions:
                self.training_history['hr_metrics'].append(episode_transitions[-1]['hr_metrics'])
            
            # Update learning rate based on performance
            self._update_learning_rates(episode_reward)
            
            # Logging
            if episode % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                self.logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.4f}, "
                               f"Episode Reward = {episode_reward:.4f}, Length = {episode_length}")
                
                # Log HR metrics
                if self.training_history['hr_metrics']:
                    latest_metrics = self.training_history['hr_metrics'][-1]
                    self.logger.info(f"Employee Satisfaction: {latest_metrics.employee_satisfaction:.3f}, "
                                   f"Productivity: {latest_metrics.productivity_index:.3f}, "
                                   f"Retention: {latest_metrics.retention_rate:.3f}")
            
            # Save best model
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_model_state = {
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'predictor': self.predictor.state_dict(),
                    'episode': episode,
                    'reward': episode_reward
                }
    
    def _run_ppo_training(self):
        """Proximal Policy Optimization training"""
        clip_epsilon = self.training_params.get('clip_epsilon', 0.2)
        ppo_epochs = self.training_params.get('ppo_epochs', 4)
        
        for episode in range(self.episodes):
            # Collect trajectories
            trajectories = []
            for _ in range(self.batch_size):
                state = self.env.reset()
                trajectory = []
                
                while True:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        actor_output = self.actor(state_tensor)
                        action_probs = actor_output['policy_probs']
                        value_estimate = actor_output['value']
                        
                        action_dist = Categorical(action_probs)
                        action = action_dist.sample()
                        log_prob = action_dist.log_prob(action)
                    
                    next_state, reward, done, info = self.env.step(action.item())
                    
                    trajectory.append({
                        'state': state,
                        'action': action.item(),
                        'reward': reward,
                        'log_prob': log_prob.item(),
                        'value': value_estimate.item(),
                        'done': done
                    })
                    
                    if done:
                        break
                    state = next_state
                
                trajectories.append(trajectory)
            
            # Compute advantages and returns
            processed_trajectories = self._compute_gae_advantages(trajectories)
            
            # PPO updates
            for _ in range(ppo_epochs):
                losses = self._update_ppo(processed_trajectories, clip_epsilon)
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean([sum(t['reward'] for t in traj) for traj in trajectories])
                self.logger.info(f"PPO Episode {episode}: Avg Reward = {avg_reward:.4f}")
    
    def _run_evolution_training(self):
        """Evolution Strategy training"""
        for generation in range(self.episodes // 10):  # Fewer generations
            # Generate population
            population = self.evolution_strategy.generate_population()
            fitness_scores = []
            
            # Evaluate population
            for individual in population:
                # Set network parameters
                self._set_network_parameters(individual)
                
                # Evaluate fitness
                fitness = self._evaluate_individual()
                fitness_scores.append(fitness)
            
            fitness_scores = torch.tensor(fitness_scores)
            
            # Update evolution strategy
            self.evolution_strategy.update_parameters(population, fitness_scores)
            
            # Logging
            if generation % 5 == 0:
                avg_fitness = torch.mean(fitness_scores).item()
                best_fitness = torch.max(fitness_scores).item()
                self.logger.info(f"Generation {generation}: Avg Fitness = {avg_fitness:.4f}, "
                               f"Best Fitness = {best_fitness:.4f}")
    
    def _update_actor_critic(self) -> Dict[str, float]:
        """Update actor and critic networks"""
        # Sample batch from memory
        batch = self.memory.sample()
        
        # Prepare batch tensors
        states = torch.cat([t['state'] for t in batch]).to(self.device)
        actions = torch.cat([t['action'].unsqueeze(0) for t in batch]).to(self.device)
        rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float).to(self.device)
        next_states = torch.cat([t['next_state'] for t in batch]).to(self.device)
        dones = torch.tensor([t['done'] for t in batch], dtype=torch.float).to(self.device)
        old_log_probs = torch.cat([t['log_prob'].unsqueeze(0) for t in batch]).to(self.device)
        old_values = torch.cat([t['value'] for t in batch]).to(self.device)
        
        # Compute target values
        with torch.no_grad():
            next_values = self.critic(next_states)['main_value'].squeeze(1)
            target_values = rewards + 0.99 * next_values * (1 - dones)
        
        # Compute advantages
        advantages = target_values - old_values.squeeze(1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        current_values = self.critic(states)['main_value'].squeeze(1)
        critic_loss = F.mse_loss(current_values, target_values)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_output = self.actor(states)
        policy_probs = actor_output['policy_probs']
        log_probs = torch.log(policy_probs + 1e-8)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Policy loss
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Entropy bonus
        entropy = -(policy_probs * log_probs).sum(1).mean()
        entropy_bonus = 0.01 * entropy
        
        actor_loss = policy_loss - entropy_bonus
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item()
        }
    
    def _update_ppo(self, trajectories: List, clip_epsilon: float) -> Dict[str, float]:
        """PPO update step"""
        # Flatten trajectories
        all_states = []
        all_actions = []
        all_returns = []
        all_advantages = []
        all_old_log_probs = []
        
        for traj in trajectories:
            for step in traj:
                all_states.append(step['state'])
                all_actions.append(step['action'])
                all_returns.append(step['return'])
                all_advantages.append(step['advantage'])
                all_old_log_probs.append(step['log_prob'])
        
        # Convert to tensors
        states = torch.FloatTensor(all_states).to(self.device)
        actions = torch.LongTensor(all_actions).to(self.device)
        returns = torch.FloatTensor(all_returns).to(self.device)
        advantages = torch.FloatTensor(all_advantages).to(self.device)
        old_log_probs = torch.FloatTensor(all_old_log_probs).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_output = self.actor(states)
        new_log_probs = self.actor.get_action_log_prob(states, actions)
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        current_values = self.critic(states)['main_value'].squeeze(1)
        value_loss = F.mse_loss(current_values, returns)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def _update_prediction_models(self):
        """Update prediction models for revenue and satisfaction"""
        if not self.memory.can_sample():
            return
        
        batch = self.memory.sample()
        
        # Prepare batch data
        states = torch.cat([t['state'] for t in batch]).to(self.device)
        
        # Generate synthetic targets based on HR metrics
        revenue_targets, satisfaction_targets = self._generate_prediction_targets(batch)
        
        # Update predictor
        self.predictor_optimizer.zero_grad()
        predictions = self.predictor(states)
        
        # Classification losses
        revenue_loss = F.cross_entropy(predictions['revenue_logits'], revenue_targets['class'])
        satisfaction_loss = F.cross_entropy(predictions['satisfaction_logits'], satisfaction_targets['class'])
        
        # Regression losses
        revenue_reg_loss = F.mse_loss(predictions['revenue_value'].squeeze(), revenue_targets['value'])
        satisfaction_reg_loss = F.mse_loss(predictions['satisfaction_value'].squeeze(), satisfaction_targets['value'])
        
        total_prediction_loss = revenue_loss + satisfaction_loss + revenue_reg_loss + satisfaction_reg_loss
        total_prediction_loss.backward()
        self.predictor_optimizer.step()
        
        # Update multi-objective optimizer
        self.multi_obj_optimizer.zero_grad()
        multi_obj_output = self.multi_objective_optimizer(states)
        
        # Multi-objective loss (simplified)
        obj_targets = self._compute_objective_targets(batch)
        multi_obj_loss = F.mse_loss(multi_obj_output['objective_values'], obj_targets)
        multi_obj_loss.backward()
        self.multi_obj_optimizer.step()
    
    def _generate_prediction_targets(self, batch: List) -> Tuple[Dict, Dict]:
        """Generate prediction targets from HR metrics"""
        revenue_classes = []
        revenue_values = []
        satisfaction_classes = []
        satisfaction_values = []
        
        for transition in batch:
            metrics = transition['hr_metrics']
            
            # Revenue prediction based on productivity and satisfaction
            revenue_score = (
                metrics.productivity_index * 0.4 +
                metrics.employee_satisfaction * 0.3 +
                metrics.innovation_index * 0.3
            )
            
            # Map to classes
            if revenue_score < 0.7:
                revenue_class = 0  # Low
            elif revenue_score < 0.85:
                revenue_class = 1  # Medium
            else:
                revenue_class = 2  # High
                
            revenue_classes.append(revenue_class)
            revenue_values.append(revenue_score)
            
            # Customer satisfaction prediction
            satisfaction_score = (
                metrics.employee_satisfaction * 0.5 +
                metrics.organizational_culture_score * 0.3 +
                metrics.innovation_index * 0.2
            )
            
            # Map to 5-point scale
            satisfaction_class = min(4, int(satisfaction_score * 5))
            satisfaction_classes.append(satisfaction_class)
            satisfaction_values.append(satisfaction_score)
        
        revenue_targets = {
            'class': torch.LongTensor(revenue_classes).to(self.device),
            'value': torch.FloatTensor(revenue_values).to(self.device)
        }
        
        satisfaction_targets = {
            'class': torch.LongTensor(satisfaction_classes).to(self.device),
            'value': torch.FloatTensor(satisfaction_values).to(self.device)
        }
        
        return revenue_targets, satisfaction_targets
    
    def _compute_objective_targets(self, batch: List) -> torch.Tensor:
        """Compute multi-objective targets"""
        objectives = []
        
        for transition in batch:
            metrics = transition['hr_metrics']
            obj_vector = torch.tensor([
                metrics.employee_satisfaction,
                metrics.productivity_index,
                metrics.retention_rate,
                metrics.innovation_index,
                metrics.organizational_culture_score
            ])
            objectives.append(obj_vector)
        
        return torch.stack(objectives).to(self.device)
    
    def _compute_gae_advantages(self, trajectories: List) -> List:
        """Compute Generalized Advantage Estimation"""
        gamma = 0.99
        gae_lambda = 0.95
        
        processed_trajectories = []
        
        for trajectory in trajectories:
            values = [step['value'] for step in trajectory]
            rewards = [step['reward'] for step in trajectory]
            
            # Compute returns and advantages
            returns = []
            advantages = []
            gae = 0
            
            for t in reversed(range(len(trajectory))):
                if t == len(trajectory) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + gamma * next_value - values[t]
                gae = delta + gamma * gae_lambda * gae
                
                returns.insert(0, gae + values[t])
                advantages.insert(0, gae)
            
            # Add returns and advantages to trajectory
            for i, step in enumerate(trajectory):
                step['return'] = returns[i]
                step['advantage'] = advantages[i]
            
            processed_trajectories.append(trajectory)
        
        return processed_trajectories
    
    def _update_learning_rates(self, episode_reward: float):
        """Update learning rates based on performance"""
        # Use learning automata to adjust learning rates
        if len(self.training_history['episode_rewards']) > 100:
            recent_performance = np.mean(self.training_history['episode_rewards'][-100:])
            
            # Reward the learning automata based on performance improvement
            if len(self.training_history['episode_rewards']) > 200:
                prev_performance = np.mean(self.training_history['episode_rewards'][-200:-100])
                performance_improvement = recent_performance - prev_performance
                self.learning_automata.update(0, performance_improvement)  # Simplified
        
        # Update schedulers
        self.actor_scheduler.step(episode_reward)
        if self.training_history['loss_history'].get('critic_loss'):
            self.critic_scheduler.step(self.training_history['loss_history']['critic_loss'][-1])
    
    def _evaluate_individual(self) -> float:
        """Evaluate an individual in evolution strategy"""
        total_reward = 0
        num_episodes = 5
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            while True:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, _ = self.actor.select_action(state_tensor, deterministic=True)
                
                next_state, reward, done, _ = self.env.step(action.item())
                episode_reward += reward
                
                if done:
                    break
                state = next_state
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def _set_network_parameters(self, parameters: torch.Tensor):
        """Set network parameters for evolution strategy"""
        # Simplified parameter setting - in practice, you'd map parameters to network weights
        param_idx = 0
        for param in self.actor.parameters():
            if param_idx < len(parameters):
                param.data.fill_(parameters[param_idx].item())
                param_idx += 1
    
    def _load_models(self):
        """Load pre-trained models"""
        model_path = self.model_params['model_path']
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.predictor.load_state_dict(checkpoint['predictor'])
            self.logger.info(f"Models loaded from {model_path}")
        except Exception as e:
            self.logger.warning(f"Could not load models: {e}")
    
    def _save_models(self):
        """Save trained models"""
        save_path = f"{self.result_folder}/hrm_models.pt"
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'predictor': self.predictor.state_dict(),
            'multi_objective': self.multi_objective_optimizer.state_dict(),
            'training_history': self.training_history,
            'best_reward': self.best_reward
        }, save_path)
        
        self.logger.info(f"Models saved to {save_path}")
    
    def _save_training_results(self):
        """Save training results and plots"""
        # Plot training curves
        self._plot_training_curves()
        
        # Save HR metrics analysis
        self._save_hr_analysis()
        
        # Generate prediction accuracy report
        self._generate_prediction_report()
    
    def _plot_training_curves(self):
        """Plot training performance curves"""
        import matplotlib.pyplot as plt
        
        # Episode rewards
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(self.training_history['episode_rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Loss curves
        if self.training_history['loss_history']:
            plt.subplot(2, 3, 2)
            for loss_name, loss_values in self.training_history['loss_history'].items():
                plt.plot(loss_values, label=loss_name)
            plt.title('Training Losses')
            plt.xlabel('Update Step')
            plt.ylabel('Loss')
            plt.legend()
        
        # HR metrics over time
        if self.training_history['hr_metrics']:
            satisfaction_scores = [m.employee_satisfaction for m in self.training_history['hr_metrics']]
            productivity_scores = [m.productivity_index for m in self.training_history['hr_metrics']]
            retention_scores = [m.retention_rate for m in self.training_history['hr_metrics']]
            
            plt.subplot(2, 3, 3)
            plt.plot(satisfaction_scores, label='Satisfaction')
            plt.plot(productivity_scores, label='Productivity')
            plt.plot(retention_scores, label='Retention')
            plt.title('HR Metrics Evolution')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.result_folder}/training_curves.png")
        plt.close()
    
    def _save_hr_analysis(self):
        """Save detailed HR metrics analysis"""
        if not self.training_history['hr_metrics']:
            return
        
        import pandas as pd
        
        # Convert HR metrics to DataFrame
        metrics_data = []
        for i, metrics in enumerate(self.training_history['hr_metrics']):
            metrics_data.append({
                'episode': i,
                'employee_satisfaction': metrics.employee_satisfaction,
                'employee_engagement': metrics.employee_engagement,
                'productivity_index': metrics.productivity_index,
                'retention_rate': metrics.retention_rate,
                'recruitment_quality': metrics.recruitment_quality,
                'training_effectiveness': metrics.training_effectiveness,
                'organizational_culture_score': metrics.organizational_culture_score,
                'leadership_effectiveness': metrics.leadership_effectiveness,
                'innovation_index': metrics.innovation_index,
                'turnover_rate': metrics.turnover_rate,
                'absenteeism_rate': metrics.absenteeism_rate
            })
        
        df = pd.DataFrame(metrics_data)
        df.to_csv(f"{self.result_folder}/hr_metrics_analysis.csv", index=False)
        
        # Compute summary statistics
        summary = df.describe()
        summary.to_csv(f"{self.result_folder}/hr_metrics_summary.csv")
    
    def _generate_prediction_report(self):
        """Generate prediction accuracy report"""
        # This would involve evaluating the predictor on test data
        # For now, we'll save the model parameters and structure
        report = {
            'model_architecture': str(self.predictor),
            'training_episodes': len(self.training_history['episode_rewards']),
            'best_reward': self.best_reward,
            'final_reward': self.training_history['episode_rewards'][-1] if self.training_history['episode_rewards'] else 0
        }
        
        import json
        with open(f"{self.result_folder}/prediction_report.json", 'w') as f:
            json.dump(report, f, indent=2)
    
    def evaluate(self, num_episodes: int = 100) -> Dict:
        """Evaluate the trained model"""
        self.logger.info(f"Evaluating model for {num_episodes} episodes...")
        
        self.actor.eval()
        self.critic.eval()
        
        evaluation_rewards = []
        evaluation_metrics = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            while True:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, _ = self.actor.select_action(state_tensor, deterministic=True)
                
                next_state, reward, done, info = self.env.step(action.item())
                episode_reward += reward
                
                if done:
                    evaluation_metrics.append(info['current_metrics'])
                    break
                    
                state = next_state
            
            evaluation_rewards.append(episode_reward)
        
        # Compute evaluation statistics
        avg_reward = np.mean(evaluation_rewards)
        std_reward = np.std(evaluation_rewards)
        
        # Compute average HR metrics
        avg_satisfaction = np.mean([m.employee_satisfaction for m in evaluation_metrics])
        avg_productivity = np.mean([m.productivity_index for m in evaluation_metrics])
        avg_retention = np.mean([m.retention_rate for m in evaluation_metrics])
        
        evaluation_results = {
            'average_reward': avg_reward,
            'reward_std': std_reward,
            'average_satisfaction': avg_satisfaction,
            'average_productivity': avg_productivity,
            'average_retention': avg_retention,
            'num_episodes': num_episodes
        }
        
        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"Average Reward: {avg_reward:.4f} ± {std_reward:.4f}")
        self.logger.info(f"Average Satisfaction: {avg_satisfaction:.3f}")
        self.logger.info(f"Average Productivity: {avg_productivity:.3f}")
        self.logger.info(f"Average Retention: {avg_retention:.3f}")
        
        return evaluation_results