"""
Core components of the Hierarchical Reasoning Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from .layers import TransformerBlock, ResidualBlock, StateProjection, RecurrentCell
from .attention import HierarchicalAttention, CrossAttention


class AdaptiveComputationTime(nn.Module):
    """Adaptive Computation Time for dynamic reasoning depth"""
    
    def __init__(self, 
                 hidden_dim: int,
                 max_steps: int = 20,
                 halt_threshold: float = 0.95,
                 epsilon: float = 0.01):
        super().__init__()
        
        self.max_steps = max_steps
        self.halt_threshold = halt_threshold
        self.epsilon = epsilon
        
        # Halting probability predictor
        self.halt_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                hidden_states: torch.Tensor,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Hidden states at each step (batch_size, max_steps, hidden_dim)
            training: Whether in training mode
            
        Returns:
            halt_probs: Halting probabilities for each step
            remainders: Remainder probabilities
            n_updates: Number of updates for each sample
        """
        batch_size, max_steps, hidden_dim = hidden_states.size()
        
        # Compute halting probabilities
        halt_probs = self.halt_predictor(hidden_states).squeeze(-1)  # (batch_size, max_steps)
        
        # Compute cumulative probabilities
        cum_probs = torch.cumsum(halt_probs, dim=1)
        
        # Determine stopping points
        if training:
            # During training, use soft stopping
            remainders = 1.0 - cum_probs
            n_updates = torch.sum(halt_probs, dim=1)
        else:
            # During inference, use hard stopping
            stop_mask = cum_probs >= self.halt_threshold
            first_stop = torch.argmax(stop_mask.float(), dim=1)
            
            # Create masks for valid steps
            step_mask = torch.arange(max_steps, device=hidden_states.device).unsqueeze(0) <= first_stop.unsqueeze(1)
            
            remainders = torch.zeros_like(cum_probs)
            remainders[torch.arange(batch_size), first_stop] = 1.0 - cum_probs[torch.arange(batch_size), first_stop - 1]
            remainders = remainders * step_mask.float()
            
            n_updates = first_stop.float() + 1
        
        return halt_probs, remainders, n_updates
    
    def compute_act_loss(self, halt_probs: torch.Tensor, n_updates: torch.Tensor) -> torch.Tensor:
        """Compute ACT regularization loss"""
        # Ponder cost: encourage fewer computation steps
        ponder_cost = torch.mean(n_updates)
        
        # Entropy regularization: encourage confident decisions
        entropy = -torch.sum(halt_probs * torch.log(halt_probs + 1e-8), dim=1)
        entropy_cost = torch.mean(entropy)
        
        return ponder_cost + self.epsilon * entropy_cost


class HighLevelModule(nn.Module):
    """High-level module for abstract planning and strategic reasoning"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_act: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_act = use_act
        
        # Input projection
        self.input_projection = StateProjection(input_dim, hidden_dim, activation="gelu")
        
        # Transformer layers for abstract reasoning
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Recurrent cell for maintaining long-term state
        self.recurrent_cell = RecurrentCell(hidden_dim, hidden_dim)
        
        # State update mechanism
        self.state_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection for low-level guidance
        self.output_projection = StateProjection(hidden_dim, hidden_dim, activation="tanh")
        
        # Adaptive computation time
        if use_act:
            self.act = AdaptiveComputationTime(hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self,
                input_x: torch.Tensor,
                low_level_feedback: Optional[torch.Tensor] = None,
                hidden_state: Optional[torch.Tensor] = None,
                training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_x: Input tensor (batch_size, seq_len, input_dim)
            low_level_feedback: Feedback from low-level module
            hidden_state: Previous hidden state
            training: Whether in training mode
            
        Returns:
            Dictionary containing:
                - output: High-level output for guiding low-level module
                - hidden_state: Updated hidden state
                - act_loss: ACT regularization loss (if using ACT)
        """
        batch_size = input_x.size(0)
        
        # Project input
        x = self.input_projection(input_x)
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Pool sequence information (mean pooling)
        x_pooled = torch.mean(x, dim=1)
        
        # Incorporate low-level feedback if provided
        if low_level_feedback is not None:
            combined_input = torch.cat([x_pooled, low_level_feedback], dim=-1)
            x_pooled = self.state_update(combined_input)
        
        # Update recurrent state
        new_hidden_state = self.recurrent_cell(x_pooled, hidden_state)
        new_hidden_state = self.layer_norm(new_hidden_state)
        
        # Generate output for low-level guidance
        output = self.output_projection(new_hidden_state)
        
        result = {
            'output': output,
            'hidden_state': new_hidden_state,
        }
        
        # Add ACT loss if using adaptive computation
        if self.use_act and training:
            # For ACT, we need to track multiple reasoning steps
            # This is a simplified version - full implementation would track all steps
            halt_probs = torch.ones(batch_size, 1, device=x.device) * 0.5
            n_updates = torch.ones(batch_size, device=x.device)
            act_loss = self.act.compute_act_loss(halt_probs, n_updates)
            result['act_loss'] = act_loss
        
        return result


class LowLevelModule(nn.Module):
    """Low-level module for detailed computation and execution"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 high_level_dim: int,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_steps: int = 20):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # High-level guidance integration
        self.guidance_integration = CrossAttention(
            hidden_dim, high_level_dim, num_heads, dropout
        )
        
        # Transformer layers for detailed computation
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Recurrent processing for iterative refinement
        self.recurrent_layers = nn.ModuleList([
            RecurrentCell(hidden_dim, hidden_dim)
            for _ in range(2)  # Multiple recurrent layers for depth
        ])
        
        # Residual blocks for non-linear processing
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim * 2, dropout)
            for _ in range(2)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Feedback generation for high-level module
        self.feedback_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, high_level_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self,
                input_x: torch.Tensor,
                high_level_guidance: torch.Tensor,
                hidden_states: Optional[torch.Tensor] = None,
                num_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_x: Input tensor (batch_size, seq_len, input_dim)
            high_level_guidance: Guidance from high-level module
            hidden_states: Previous hidden states for recurrent layers
            num_steps: Number of processing steps (default: max_steps)
            
        Returns:
            Dictionary containing:
                - output: Low-level output
                - hidden_states: Updated hidden states
                - feedback: Feedback for high-level module
                - intermediate_states: All intermediate processing states
        """
        if num_steps is None:
            num_steps = self.max_steps
            
        batch_size, seq_len, _ = input_x.size()
        
        # Embed input
        x = self.input_embedding(input_x)
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(len(self.recurrent_layers))
            ]
        
        # Store intermediate states for analysis
        intermediate_states = []
        
        # Iterative processing with high-level guidance
        for step in range(num_steps):
            # Apply transformer layers
            for layer in self.transformer_layers:
                x = layer(x)
            
            # Pool sequence information
            x_pooled = torch.mean(x, dim=1)
            
            # Integrate high-level guidance
            guidance_expanded = high_level_guidance.unsqueeze(1)  # Add sequence dimension
            x_guided = self.guidance_integration(
                x_pooled.unsqueeze(1), guidance_expanded
            ).squeeze(1)
            
            # Apply recurrent processing
            for i, recurrent_layer in enumerate(self.recurrent_layers):
                hidden_states[i] = recurrent_layer(x_guided, hidden_states[i])
                x_guided = hidden_states[i]
            
            # Apply residual blocks
            for residual_block in self.residual_blocks:
                x_guided = residual_block(x_guided)
            
            # Store intermediate state
            intermediate_states.append(x_guided.clone())
            
            # Update x for next iteration
            x = x_guided.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Final processing
        final_state = self.layer_norm(x_guided)
        output = self.output_projection(final_state)
        
        # Generate feedback for high-level module
        feedback = self.feedback_generator(final_state)
        
        return {
            'output': output,
            'hidden_states': hidden_states,
            'feedback': feedback,
            'intermediate_states': torch.stack(intermediate_states, dim=1)  # (batch_size, num_steps, hidden_dim)
        }


class HierarchicalCommunication(nn.Module):
    """Communication mechanism between hierarchical levels"""
    
    def __init__(self,
                 high_level_dim: int,
                 low_level_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        # Bidirectional attention between levels
        self.hierarchical_attention = HierarchicalAttention(
            high_level_dim, low_level_dim, num_heads, dropout
        )
        
        # Information fusion
        self.high_level_fusion = nn.Sequential(
            nn.Linear(high_level_dim * 2, high_level_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.low_level_fusion = nn.Sequential(
            nn.Linear(low_level_dim * 2, low_level_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self,
                high_level_state: torch.Tensor,
                low_level_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            high_level_state: High-level state tensor
            low_level_state: Low-level state tensor
            
        Returns:
            Updated high-level and low-level states
        """
        # Cross-attention between levels
        high_attended, low_attended = self.hierarchical_attention(
            high_level_state.unsqueeze(1), low_level_state.unsqueeze(1)
        )
        
        # Fusion with original states
        high_fused = self.high_level_fusion(
            torch.cat([high_level_state, high_attended.squeeze(1)], dim=-1)
        )
        
        low_fused = self.low_level_fusion(
            torch.cat([low_level_state, low_attended.squeeze(1)], dim=-1)
        )
        
        return high_fused, low_fused


class ReasoningController(nn.Module):
    """Controller for managing the reasoning process"""
    
    def __init__(self,
                 high_level_dim: int,
                 low_level_dim: int,
                 max_cycles: int = 10):
        super().__init__()
        
        self.max_cycles = max_cycles
        
        # Cycle termination predictor
        self.termination_predictor = nn.Sequential(
            nn.Linear(high_level_dim + low_level_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Progress monitor
        self.progress_monitor = nn.Sequential(
            nn.Linear(high_level_dim + low_level_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def should_terminate(self,
                        high_level_state: torch.Tensor,
                        low_level_state: torch.Tensor,
                        cycle: int,
                        threshold: float = 0.8) -> torch.Tensor:
        """
        Determine if reasoning should terminate
        
        Args:
            high_level_state: Current high-level state
            low_level_state: Current low-level state  
            cycle: Current cycle number
            threshold: Termination threshold
            
        Returns:
            Boolean tensor indicating termination for each sample
        """
        # Combine states
        combined_state = torch.cat([high_level_state, low_level_state], dim=-1)
        
        # Predict termination probability
        termination_prob = self.termination_predictor(combined_state).squeeze(-1)
        
        # Force termination at max cycles
        max_cycle_mask = cycle >= self.max_cycles
        
        # Combine conditions
        should_stop = (termination_prob > threshold) | max_cycle_mask
        
        return should_stop
    
    def get_progress(self,
                    high_level_state: torch.Tensor,
                    low_level_state: torch.Tensor) -> torch.Tensor:
        """Get reasoning progress score"""
        combined_state = torch.cat([high_level_state, low_level_state], dim=-1)
        return self.progress_monitor(combined_state).squeeze(-1)