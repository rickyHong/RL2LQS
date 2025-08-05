"""
Main Hierarchical Reasoning Model (HRM) implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import logging

from .components import (
    HighLevelModule, LowLevelModule, HierarchicalCommunication, 
    ReasoningController, AdaptiveComputationTime
)
from ..utils.config import HRMConfig


logger = logging.getLogger(__name__)


class HierarchicalReasoningModel(nn.Module):
    """
    Hierarchical Reasoning Model for complex sequential decision-making tasks.
    
    The model consists of two interdependent recurrent modules:
    - High-level module: Slow, abstract planning
    - Low-level module: Fast, detailed computations
    
    Key features:
    - Hierarchical convergence with adaptive computation time
    - Dynamic reasoning depth based on task complexity
    - Efficient one-step gradient learning
    - Support for various reasoning tasks
    """
    
    def __init__(self, config: HRMConfig):
        super().__init__()
        
        self.config = config
        self.input_dim = config.model.input_dim
        self.high_level_dim = config.model.high_level_dim
        self.low_level_dim = config.model.low_level_dim
        self.max_high_level_cycles = config.reasoning.max_high_level_cycles
        self.max_low_level_steps = config.reasoning.max_low_level_steps
        self.use_adaptive_computation = config.reasoning.adaptive_computation
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim, config.model.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.LayerNorm(config.model.hidden_dim)
        )
        
        # High-level module for abstract planning
        self.high_level_module = HighLevelModule(
            input_dim=config.model.hidden_dim,
            hidden_dim=self.high_level_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            use_act=config.reasoning.use_act_loss
        )
        
        # Low-level module for detailed computation
        self.low_level_module = LowLevelModule(
            input_dim=config.model.hidden_dim,
            hidden_dim=self.low_level_dim,
            high_level_dim=self.high_level_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            max_steps=self.max_low_level_steps
        )
        
        # Hierarchical communication
        self.hierarchical_communication = HierarchicalCommunication(
            high_level_dim=self.high_level_dim,
            low_level_dim=self.low_level_dim,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout
        )
        
        # Reasoning controller
        self.reasoning_controller = ReasoningController(
            high_level_dim=self.high_level_dim,
            low_level_dim=self.low_level_dim,
            max_cycles=self.max_high_level_cycles
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.low_level_dim, config.model.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_dim, self.input_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, 
                x: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the HRM model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            target: Target tensor for training (optional)
            return_intermediate: Whether to return intermediate states
            
        Returns:
            Dictionary containing:
                - output: Final model output
                - loss: Total loss (if target provided)
                - intermediate_states: Intermediate reasoning states (if requested)
                - reasoning_trace: Detailed reasoning trace
        """
        batch_size = x.size(0)
        device = x.device
        
        # Embed input
        x_embedded = self.input_embedding(x)
        
        # Initialize states
        high_level_state = None
        low_level_states = None
        
        # Storage for intermediate states and losses
        intermediate_states = []
        reasoning_trace = {
            'high_level_states': [],
            'low_level_states': [],
            'termination_probs': [],
            'progress_scores': []
        }
        total_act_loss = 0.0
        
        # Hierarchical reasoning loop
        for cycle in range(self.max_high_level_cycles):
            # High-level planning step
            if cycle == 0:
                # First cycle: process input
                high_result = self.high_level_module(
                    x_embedded, 
                    hidden_state=high_level_state,
                    training=self.training
                )
            else:
                # Subsequent cycles: incorporate low-level feedback
                high_result = self.high_level_module(
                    x_embedded,
                    low_level_feedback=low_result['feedback'],
                    hidden_state=high_level_state,
                    training=self.training
                )
            
            high_level_state = high_result['hidden_state']
            high_level_guidance = high_result['output']
            
            # Add ACT loss if available
            if 'act_loss' in high_result:
                total_act_loss += high_result['act_loss']
            
            # Low-level execution step
            low_result = self.low_level_module(
                x_embedded,
                high_level_guidance=high_level_guidance,
                hidden_states=low_level_states,
                num_steps=self.max_low_level_steps
            )
            
            low_level_states = low_result['hidden_states']
            low_level_output = low_result['output']
            
            # Hierarchical communication
            if self.config.model.use_residual:
                high_level_state, low_level_output = self.hierarchical_communication(
                    high_level_state, low_level_output
                )
            
            # Store intermediate states
            if return_intermediate:
                intermediate_states.append({
                    'cycle': cycle,
                    'high_level_state': high_level_state.clone(),
                    'low_level_output': low_level_output.clone(),
                    'low_level_intermediate': low_result['intermediate_states']
                })
            
            # Update reasoning trace
            reasoning_trace['high_level_states'].append(high_level_state.clone())
            reasoning_trace['low_level_states'].append(low_level_output.clone())
            
            # Check termination condition
            if self.use_adaptive_computation and cycle > 0:
                should_terminate = self.reasoning_controller.should_terminate(
                    high_level_state, low_level_output, cycle
                )
                
                termination_prob = torch.mean(should_terminate.float())
                reasoning_trace['termination_probs'].append(termination_prob.item())
                
                # Early termination for entire batch if all samples should stop
                if torch.all(should_terminate):
                    logger.debug(f"Early termination at cycle {cycle}")
                    break
            
            # Progress monitoring
            progress = self.reasoning_controller.get_progress(
                high_level_state, low_level_output
            )
            reasoning_trace['progress_scores'].append(torch.mean(progress).item())
        
        # Generate final output
        final_output = self.output_head(low_level_output)
        
        # Prepare return dictionary
        result = {
            'output': final_output,
            'reasoning_trace': reasoning_trace,
            'num_cycles': cycle + 1
        }
        
        if return_intermediate:
            result['intermediate_states'] = intermediate_states
        
        # Compute loss if target is provided
        if target is not None:
            # Main reconstruction loss
            main_loss = F.mse_loss(final_output, target)
            
            # ACT regularization loss
            act_loss = total_act_loss * self.config.reasoning.act_loss_weight
            
            # Total loss
            total_loss = main_loss + act_loss
            
            result.update({
                'loss': total_loss,
                'main_loss': main_loss,
                'act_loss': act_loss
            })
        
        return result
    
    def generate(self, 
                 x: torch.Tensor,
                 max_cycles: Optional[int] = None,
                 temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Generate output with optional early stopping
        
        Args:
            x: Input tensor
            max_cycles: Maximum number of reasoning cycles
            temperature: Sampling temperature for stochastic generation
            
        Returns:
            Generated output and reasoning information
        """
        self.eval()
        
        if max_cycles is None:
            max_cycles = self.max_high_level_cycles
        
        with torch.no_grad():
            # Temporarily modify max cycles
            original_max_cycles = self.max_high_level_cycles
            self.max_high_level_cycles = max_cycles
            
            # Forward pass
            result = self.forward(x, return_intermediate=True)
            
            # Apply temperature if specified
            if temperature != 1.0:
                result['output'] = result['output'] / temperature
            
            # Restore original max cycles
            self.max_high_level_cycles = original_max_cycles
        
        return result
    
    def get_reasoning_depth(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate required reasoning depth for given input
        
        Args:
            x: Input tensor
            
        Returns:
            Estimated reasoning depth for each sample
        """
        self.eval()
        
        with torch.no_grad():
            result = self.forward(x, return_intermediate=True)
            
            # Extract termination probabilities
            termination_probs = result['reasoning_trace']['termination_probs']
            
            if not termination_probs:
                return torch.full((x.size(0),), self.max_high_level_cycles, device=x.device)
            
            # Find first cycle where termination probability > 0.5
            depth_estimates = []
            for i, prob in enumerate(termination_probs):
                if prob > 0.5:
                    depth_estimates.append(i + 1)
                    break
            else:
                depth_estimates.append(len(termination_probs))
            
            return torch.tensor(depth_estimates, device=x.device)
    
    def visualize_reasoning(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Generate detailed visualization data for reasoning process
        
        Args:
            x: Input tensor (single sample)
            
        Returns:
            Visualization data including state evolution and attention maps
        """
        assert x.size(0) == 1, "Visualization supports single sample only"
        
        self.eval()
        
        with torch.no_grad():
            result = self.forward(x, return_intermediate=True)
            
            # Extract visualization data
            viz_data = {
                'input': x.squeeze(0).cpu().numpy(),
                'output': result['output'].squeeze(0).cpu().numpy(),
                'num_cycles': result['num_cycles'],
                'reasoning_trace': result['reasoning_trace'],
                'intermediate_states': result['intermediate_states']
            }
            
            # Compute state evolution metrics
            high_level_states = torch.stack(result['reasoning_trace']['high_level_states'])
            low_level_states = torch.stack(result['reasoning_trace']['low_level_states'])
            
            # State change magnitudes
            if len(high_level_states) > 1:
                high_level_changes = torch.norm(
                    high_level_states[1:] - high_level_states[:-1], dim=-1
                )
                low_level_changes = torch.norm(
                    low_level_states[1:] - low_level_states[:-1], dim=-1
                )
                
                viz_data['state_changes'] = {
                    'high_level': high_level_changes.cpu().numpy(),
                    'low_level': low_level_changes.cpu().numpy()
                }
            
            return viz_data
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': self.config.to_dict(),
            'architecture': {
                'high_level_dim': self.high_level_dim,
                'low_level_dim': self.low_level_dim,
                'max_high_level_cycles': self.max_high_level_cycles,
                'max_low_level_steps': self.max_low_level_steps,
                'adaptive_computation': self.use_adaptive_computation
            }
        }