"""
Loss functions for HRM training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class HRMLoss(nn.Module):
    """Main loss function for HRM training"""
    
    def __init__(self,
                 main_loss_weight: float = 1.0,
                 act_loss_weight: float = 0.01,
                 consistency_loss_weight: float = 0.1,
                 diversity_loss_weight: float = 0.05):
        super().__init__()
        
        self.main_loss_weight = main_loss_weight
        self.act_loss_weight = act_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                model_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute HRM loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            model_outputs: Additional model outputs (ACT loss, reasoning trace, etc.)
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Main reconstruction/prediction loss
        if targets.dtype == torch.long:
            # Classification task
            main_loss = F.cross_entropy(predictions, targets)
        else:
            # Regression task
            main_loss = F.mse_loss(predictions, targets)
        
        losses['main_loss'] = main_loss
        
        # ACT regularization loss
        if 'act_loss' in model_outputs:
            losses['act_loss'] = model_outputs['act_loss']
        else:
            losses['act_loss'] = torch.tensor(0.0, device=predictions.device)
        
        # Consistency loss across reasoning cycles
        if 'reasoning_trace' in model_outputs:
            consistency_loss = self._compute_consistency_loss(
                model_outputs['reasoning_trace']
            )
            losses['consistency_loss'] = consistency_loss
        else:
            losses['consistency_loss'] = torch.tensor(0.0, device=predictions.device)
        
        # Diversity loss to encourage different reasoning paths
        if 'intermediate_states' in model_outputs:
            diversity_loss = self._compute_diversity_loss(
                model_outputs['intermediate_states']
            )
            losses['diversity_loss'] = diversity_loss
        else:
            losses['diversity_loss'] = torch.tensor(0.0, device=predictions.device)
        
        # Total weighted loss
        total_loss = (
            self.main_loss_weight * losses['main_loss'] +
            self.act_loss_weight * losses['act_loss'] +
            self.consistency_loss_weight * losses['consistency_loss'] +
            self.diversity_loss_weight * losses['diversity_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_consistency_loss(self, reasoning_trace: Dict[str, Any]) -> torch.Tensor:
        """Compute consistency loss across reasoning cycles"""
        if 'high_level_states' not in reasoning_trace or len(reasoning_trace['high_level_states']) < 2:
            return torch.tensor(0.0)
        
        high_level_states = torch.stack(reasoning_trace['high_level_states'])
        
        # Encourage smooth transitions between cycles
        state_diffs = high_level_states[1:] - high_level_states[:-1]
        consistency_loss = torch.mean(torch.norm(state_diffs, dim=-1) ** 2)
        
        return consistency_loss
    
    def _compute_diversity_loss(self, intermediate_states: list) -> torch.Tensor:
        """Compute diversity loss to encourage exploration"""
        if len(intermediate_states) < 2:
            return torch.tensor(0.0)
        
        # Extract low-level intermediate states
        all_states = []
        for cycle_states in intermediate_states:
            if 'low_level_intermediate' in cycle_states:
                states = cycle_states['low_level_intermediate']  # (batch_size, num_steps, hidden_dim)
                all_states.append(states)
        
        if len(all_states) < 2:
            return torch.tensor(0.0)
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(all_states)):
            for j in range(i + 1, len(all_states)):
                # Average pool over steps
                state_i = torch.mean(all_states[i], dim=1)  # (batch_size, hidden_dim)
                state_j = torch.mean(all_states[j], dim=1)  # (batch_size, hidden_dim)
                
                # Cosine similarity
                similarity = F.cosine_similarity(state_i, state_j, dim=-1)
                similarities.append(similarity)
        
        if similarities:
            # Encourage diversity by penalizing high similarity
            avg_similarity = torch.mean(torch.stack(similarities))
            diversity_loss = torch.clamp(avg_similarity - 0.3, min=0.0)  # Target similarity < 0.3
        else:
            diversity_loss = torch.tensor(0.0)
        
        return diversity_loss


class AdaptiveLoss(nn.Module):
    """Adaptive loss that adjusts weights based on training progress"""
    
    def __init__(self,
                 base_loss: nn.Module,
                 adaptation_rate: float = 0.01,
                 min_weight: float = 0.01,
                 max_weight: float = 1.0):
        super().__init__()
        
        self.base_loss = base_loss
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Adaptive weights (learnable parameters)
        self.register_parameter('act_weight', nn.Parameter(torch.tensor(0.01)))
        self.register_parameter('consistency_weight', nn.Parameter(torch.tensor(0.1)))
        self.register_parameter('diversity_weight', nn.Parameter(torch.tensor(0.05)))
        
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                model_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive weights"""
        
        # Clamp weights to valid range
        act_weight = torch.clamp(self.act_weight, self.min_weight, self.max_weight)
        consistency_weight = torch.clamp(self.consistency_weight, self.min_weight, self.max_weight)
        diversity_weight = torch.clamp(self.diversity_weight, self.min_weight, self.max_weight)
        
        # Temporarily update base loss weights
        original_weights = (
            self.base_loss.act_loss_weight,
            self.base_loss.consistency_loss_weight,
            self.base_loss.diversity_loss_weight
        )
        
        self.base_loss.act_loss_weight = act_weight.item()
        self.base_loss.consistency_loss_weight = consistency_weight.item()
        self.base_loss.diversity_loss_weight = diversity_weight.item()
        
        # Compute losses
        losses = self.base_loss(predictions, targets, model_outputs)
        
        # Restore original weights
        (self.base_loss.act_loss_weight,
         self.base_loss.consistency_loss_weight,
         self.base_loss.diversity_loss_weight) = original_weights
        
        # Add adaptive weight information
        losses['adaptive_weights'] = {
            'act_weight': act_weight.item(),
            'consistency_weight': consistency_weight.item(),
            'diversity_weight': diversity_weight.item()
        }
        
        return losses


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning better representations"""
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self,
                embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            embeddings: Normalized embeddings (batch_size, embedding_dim)
            labels: Labels for positive/negative pairs
            
        Returns:
            Contrastive loss
        """
        # Compute pairwise similarities
        similarities = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # Create positive and negative masks
        batch_size = embeddings.size(0)
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.t()).float()
        negative_mask = (labels != labels.t()).float()
        
        # Remove self-similarities
        mask = torch.eye(batch_size, device=embeddings.device)
        positive_mask = positive_mask * (1 - mask)
        negative_mask = negative_mask * (1 - mask)
        
        # Compute positive and negative losses
        positive_loss = -torch.log(torch.exp(similarities) + 1e-8) * positive_mask
        negative_loss = torch.clamp(self.margin - similarities, min=0) * negative_mask
        
        # Average over valid pairs
        num_positive = torch.sum(positive_mask)
        num_negative = torch.sum(negative_mask)
        
        if num_positive > 0:
            positive_loss = torch.sum(positive_loss) / num_positive
        else:
            positive_loss = torch.tensor(0.0, device=embeddings.device)
            
        if num_negative > 0:
            negative_loss = torch.sum(negative_loss) / num_negative
        else:
            negative_loss = torch.tensor(0.0, device=embeddings.device)
        
        return positive_loss + negative_loss


class CurriculumLoss(nn.Module):
    """Curriculum learning loss that gradually increases difficulty"""
    
    def __init__(self,
                 base_loss: nn.Module,
                 curriculum_steps: int = 1000,
                 difficulty_ramp: str = "linear"):
        super().__init__()
        
        self.base_loss = base_loss
        self.curriculum_steps = curriculum_steps
        self.difficulty_ramp = difficulty_ramp
        self.current_step = 0
        
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                model_outputs: Dict[str, Any],
                difficulty_scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with curriculum weighting
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            model_outputs: Additional model outputs
            difficulty_scores: Per-sample difficulty scores (optional)
        """
        # Compute base losses
        losses = self.base_loss(predictions, targets, model_outputs)
        
        # Apply curriculum weighting if difficulty scores provided
        if difficulty_scores is not None:
            curriculum_weight = self._get_curriculum_weight()
            
            # Weight samples based on difficulty and curriculum progress
            if self.difficulty_ramp == "linear":
                max_difficulty = curriculum_weight
            elif self.difficulty_ramp == "exponential":
                max_difficulty = curriculum_weight ** 2
            else:
                max_difficulty = curriculum_weight
            
            # Create sample weights
            sample_weights = (difficulty_scores <= max_difficulty).float()
            
            # Apply weights to main loss
            if sample_weights.sum() > 0:
                weighted_main_loss = (losses['main_loss'] * sample_weights).sum() / sample_weights.sum()
                losses['main_loss'] = weighted_main_loss
                losses['total_loss'] = (
                    losses['total_loss'] - self.base_loss.main_loss_weight * losses['main_loss'] +
                    self.base_loss.main_loss_weight * weighted_main_loss
                )
        
        # Update step counter
        self.current_step += 1
        
        losses['curriculum_weight'] = self._get_curriculum_weight()
        
        return losses
    
    def _get_curriculum_weight(self) -> float:
        """Get current curriculum weight based on training progress"""
        progress = min(self.current_step / self.curriculum_steps, 1.0)
        
        if self.difficulty_ramp == "linear":
            return progress
        elif self.difficulty_ramp == "exponential":
            return progress ** 2
        elif self.difficulty_ramp == "cosine":
            return 0.5 * (1 - torch.cos(torch.tensor(progress * 3.14159)).item())
        else:
            return progress