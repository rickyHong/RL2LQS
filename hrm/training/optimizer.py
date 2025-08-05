"""
Optimizer and scheduler utilities for HRM training
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Dict, Any, Optional


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        
    Returns:
        Configured optimizer
    """
    optimizer_name = config.get('optimizer', 'adamw').lower()
    learning_rate = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 1e-5)
    
    # Separate parameters for different components
    param_groups = []
    
    # High-level module parameters (slower learning rate)
    high_level_params = []
    low_level_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'high_level_module' in name:
            high_level_params.append(param)
        elif 'low_level_module' in name:
            low_level_params.append(param)
        else:
            other_params.append(param)
    
    # Different learning rates for different components
    if high_level_params:
        param_groups.append({
            'params': high_level_params,
            'lr': learning_rate * 0.5,  # Slower for high-level
            'weight_decay': weight_decay
        })
    
    if low_level_params:
        param_groups.append({
            'params': low_level_params,
            'lr': learning_rate,  # Normal rate for low-level
            'weight_decay': weight_decay
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': learning_rate,
            'weight_decay': weight_decay
        })
    
    # Create optimizer
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8)
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            param_groups,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8)
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            momentum=config.get('momentum', 0.9),
            nesterov=config.get('nesterov', True)
        )
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(
            param_groups,
            alpha=config.get('alpha', 0.99),
            momentum=config.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        
    Returns:
        Configured scheduler or None
    """
    scheduler_name = config.get('scheduler', 'cosine').lower()
    
    if scheduler_name == 'none':
        return None
    
    max_epochs = config.get('max_epochs', 100)
    warmup_steps = config.get('warmup_steps', 1000)
    
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=config.get('min_lr', 1e-6)
        )
    elif scheduler_name == 'cosine_warmup':
        scheduler = CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_epochs * config.get('steps_per_epoch', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
    elif scheduler_name == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.get('start_factor', 1.0),
            end_factor=config.get('end_factor', 0.1),
            total_iters=max_epochs
        )
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1)
        )
    elif scheduler_name == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.get('milestones', [60, 80]),
            gamma=config.get('gamma', 0.1)
        )
    elif scheduler_name == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.get('gamma', 0.95)
        )
    elif scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get('mode', 'min'),
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 10),
            threshold=config.get('threshold', 1e-4)
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


class CosineAnnealingWarmupScheduler(_LRScheduler):
    """Cosine annealing scheduler with warmup"""
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 max_steps: int,
                 eta_min: float = 1e-6,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class AdaptiveScheduler(_LRScheduler):
    """Adaptive scheduler that adjusts based on model performance"""
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 patience: int = 10,
                 factor: float = 0.5,
                 min_lr: float = 1e-6,
                 threshold: float = 1e-4,
                 last_epoch: int = -1):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        
        self.best_loss = float('inf')
        self.num_bad_epochs = 0
        self.losses = []
        
        super().__init__(optimizer, last_epoch)
    
    def step(self, loss: Optional[float] = None):
        """Step with optional loss for adaptive adjustment"""
        if loss is not None:
            self.losses.append(loss)
            
            # Check if loss improved
            if loss < self.best_loss - self.threshold:
                self.best_loss = loss
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
            
            # Reduce learning rate if no improvement
            if self.num_bad_epochs >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                
                self.num_bad_epochs = 0
        
        super().step()
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class WarmupScheduler(_LRScheduler):
    """Simple warmup scheduler"""
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            return self.base_lrs


class CyclicScheduler(_LRScheduler):
    """Cyclic learning rate scheduler"""
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 base_lr: float,
                 max_lr: float,
                 step_size: int,
                 mode: str = 'triangular',
                 last_epoch: int = -1):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_factor = 1.0
        elif self.mode == 'triangular2':
            scale_factor = 1 / (2 ** (cycle - 1))
        elif self.mode == 'exp_range':
            scale_factor = 0.99999 ** self.last_epoch
        else:
            scale_factor = 1.0
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x) * scale_factor
        return [lr for _ in self.base_lrs]


def get_parameter_groups(model: torch.nn.Module, 
                        weight_decay: float = 1e-5) -> list:
    """
    Get parameter groups with different weight decay settings
    
    Args:
        model: Model to get parameters from
        weight_decay: Default weight decay
        
    Returns:
        List of parameter groups
    """
    # Parameters that should not have weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    
    param_groups = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            'weight_decay': weight_decay
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            'weight_decay': 0.0
        }
    ]
    
    return param_groups