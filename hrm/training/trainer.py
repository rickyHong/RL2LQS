"""
Main trainer for HRM model
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np

from ..models.hrm_model import HierarchicalReasoningModel
from ..utils.config import HRMConfig
from .losses import HRMLoss
from .optimizer import create_optimizer, create_scheduler


logger = logging.getLogger(__name__)


class HRMTrainer:
    """Main trainer for Hierarchical Reasoning Model"""
    
    def __init__(self, 
                 model: HierarchicalReasoningModel,
                 config: HRMConfig,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None):
        
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup device
        self.device = self._setup_device(config.device)
        self.model.to(self.device)
        
        # Setup loss function
        self.criterion = HRMLoss(
            main_loss_weight=1.0,
            act_loss_weight=config.reasoning.act_loss_weight,
            consistency_loss_weight=0.1,
            diversity_loss_weight=0.05
        )
        
        # Setup optimizer and scheduler
        optimizer_config = {
            'optimizer': config.training.optimizer,
            'learning_rate': config.training.learning_rate,
            'weight_decay': config.training.weight_decay,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        }
        self.optimizer = create_optimizer(self.model, optimizer_config)
        
        scheduler_config = {
            'scheduler': config.training.scheduler,
            'max_epochs': config.training.max_epochs,
            'warmup_steps': config.training.warmup_steps,
            'min_lr': 1e-6
        }
        self.scheduler = create_scheduler(self.optimizer, scheduler_config)
        
        # Setup directories
        self.save_dir = Path(config.save_dir)
        self.log_dir = Path(config.log_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.writer = SummaryWriter(self.log_dir / config.experiment_name)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Mixed precision training
        if config.training.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Metrics tracking
        self.train_metrics = {'loss': [], 'accuracy': []}
        self.val_metrics = {'loss': [], 'accuracy': []}
        
    def _setup_device(self, device_config: str) -> torch.device:
        """Setup training device"""
        if device_config == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(device_config)
            logger.info(f"Using specified device: {device}")
        
        return device
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info("Starting HRM training...")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.max_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = None
            if self.val_loader is not None:
                val_metrics = self._validate_epoch()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics is not None:
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step(train_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Logging
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Model checkpointing
            if epoch % self.config.training.save_every == 0:
                self._save_checkpoint(epoch, train_metrics, val_metrics)
            
            # Early stopping check
            if val_metrics is not None:
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.early_stopping_counter = 0
                    self._save_best_model(epoch, val_metrics)
                else:
                    self.early_stopping_counter += 1
                
                if self.early_stopping_counter >= self.config.training.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Final evaluation
        final_metrics = {}
        if self.test_loader is not None:
            test_metrics = self._test()
            final_metrics['test'] = test_metrics
        
        self.writer.close()
        
        return {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'final_metrics': final_metrics,
            'training_time': total_time,
            'best_val_loss': self.best_val_loss
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_main_loss = 0.0
        total_act_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            inputs, targets = self._prepare_batch(batch)
            batch_size = inputs.size(0)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    model_outputs = self.model(inputs, targets, return_intermediate=False)
                    predictions = model_outputs['output']
                    loss_dict = self.criterion(predictions, targets, model_outputs)
                    loss = loss_dict['total_loss']
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                # Standard training
                model_outputs = self.model(inputs, targets, return_intermediate=False)
                predictions = model_outputs['output']
                loss_dict = self.criterion(predictions, targets, model_outputs)
                loss = loss_dict['total_loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * batch_size
            total_main_loss += loss_dict['main_loss'].item() * batch_size
            total_act_loss += loss_dict['act_loss'].item() * batch_size
            total_samples += batch_size
            
            # Calculate accuracy (for classification tasks)
            if targets.dtype == torch.long:
                _, predicted = torch.max(predictions.data, 1)
                correct_predictions += (predicted == targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Main': f"{loss_dict['main_loss'].item():.4f}",
                'ACT': f"{loss_dict['act_loss'].item():.4f}",
                'Cycles': f"{model_outputs.get('num_cycles', 0):.1f}"
            })
            
            # Tensorboard logging
            if self.global_step % 100 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/MainLoss', loss_dict['main_loss'].item(), self.global_step)
                self.writer.add_scalar('Train/ACTLoss', loss_dict['act_loss'].item(), self.global_step)
                self.writer.add_scalar('Train/NumCycles', model_outputs.get('num_cycles', 0), self.global_step)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        avg_main_loss = total_main_loss / total_samples
        avg_act_loss = total_act_loss / total_samples
        accuracy = correct_predictions / total_samples if targets.dtype == torch.long else 0.0
        
        metrics = {
            'loss': avg_loss,
            'main_loss': avg_main_loss,
            'act_loss': avg_act_loss,
            'accuracy': accuracy
        }
        
        # Update tracking
        self.train_metrics['loss'].append(avg_loss)
        self.train_metrics['accuracy'].append(accuracy)
        
        return metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_main_loss = 0.0
        total_act_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch in pbar:
                inputs, targets = self._prepare_batch(batch)
                batch_size = inputs.size(0)
                
                # Forward pass
                model_outputs = self.model(inputs, targets, return_intermediate=False)
                predictions = model_outputs['output']
                loss_dict = self.criterion(predictions, targets, model_outputs)
                
                # Update metrics
                total_loss += loss_dict['total_loss'].item() * batch_size
                total_main_loss += loss_dict['main_loss'].item() * batch_size
                total_act_loss += loss_dict['act_loss'].item() * batch_size
                total_samples += batch_size
                
                # Calculate accuracy
                if targets.dtype == torch.long:
                    _, predicted = torch.max(predictions.data, 1)
                    correct_predictions += (predicted == targets).sum().item()
                
                pbar.set_postfix({
                    'Val Loss': f"{loss_dict['total_loss'].item():.4f}"
                })
        
        # Calculate metrics
        avg_loss = total_loss / total_samples
        avg_main_loss = total_main_loss / total_samples
        avg_act_loss = total_act_loss / total_samples
        accuracy = correct_predictions / total_samples if targets.dtype == torch.long else 0.0
        
        metrics = {
            'loss': avg_loss,
            'main_loss': avg_main_loss,
            'act_loss': avg_act_loss,
            'accuracy': accuracy
        }
        
        # Update tracking
        self.val_metrics['loss'].append(avg_loss)
        self.val_metrics['accuracy'].append(accuracy)
        
        return metrics
    
    def _test(self) -> Dict[str, float]:
        """Test the model"""
        logger.info("Running final test evaluation...")
        
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        reasoning_depths = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            
            for batch in pbar:
                inputs, targets = self._prepare_batch(batch)
                batch_size = inputs.size(0)
                
                # Forward pass with reasoning analysis
                model_outputs = self.model(inputs, targets, return_intermediate=True)
                predictions = model_outputs['output']
                loss_dict = self.criterion(predictions, targets, model_outputs)
                
                # Track reasoning depth
                reasoning_depths.append(model_outputs.get('num_cycles', 0))
                
                # Update metrics
                total_loss += loss_dict['total_loss'].item() * batch_size
                total_samples += batch_size
                
                if targets.dtype == torch.long:
                    _, predicted = torch.max(predictions.data, 1)
                    correct_predictions += (predicted == targets).sum().item()
        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples if targets.dtype == torch.long else 0.0
        avg_reasoning_depth = np.mean(reasoning_depths)
        
        test_metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'avg_reasoning_depth': avg_reasoning_depth
        }
        
        logger.info(f"Test Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, "
                   f"Avg Reasoning Depth: {avg_reasoning_depth:.2f}")
        
        return test_metrics
    
    def _prepare_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch for training"""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        else:
            # Assume batch is just inputs for unsupervised tasks
            inputs = batch
            targets = batch  # Self-supervision
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        return inputs, targets
    
    def _log_epoch_metrics(self, 
                          epoch: int, 
                          train_metrics: Dict[str, float],
                          val_metrics: Optional[Dict[str, float]] = None):
        """Log metrics for the epoch"""
        
        # Console logging
        log_str = f"Epoch {epoch + 1}/{self.config.training.max_epochs} - "
        log_str += f"Train Loss: {train_metrics['loss']:.4f}"
        
        if train_metrics['accuracy'] > 0:
            log_str += f", Train Acc: {train_metrics['accuracy']:.4f}"
        
        if val_metrics is not None:
            log_str += f", Val Loss: {val_metrics['loss']:.4f}"
            if val_metrics['accuracy'] > 0:
                log_str += f", Val Acc: {val_metrics['accuracy']:.4f}"
        
        logger.info(log_str)
        
        # Tensorboard logging
        self.writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
        self.writer.add_scalar('Epoch/Train_Accuracy', train_metrics['accuracy'], epoch)
        
        if val_metrics is not None:
            self.writer.add_scalar('Epoch/Val_Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/Val_Accuracy', val_metrics['accuracy'], epoch)
    
    def _save_checkpoint(self, 
                        epoch: int,
                        train_metrics: Dict[str, float],
                        val_metrics: Optional[Dict[str, float]] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config.to_dict(),
            'global_step': self.global_step
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_best_model(self, epoch: int, val_metrics: Dict[str, float]):
        """Save best model based on validation loss"""
        best_model_path = self.save_dir / "best_model.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config.to_dict(),
            'global_step': self.global_step
        }
        
        torch.save(checkpoint, best_model_path)
        logger.info(f"Best model saved with val loss: {val_metrics['loss']:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {self.current_epoch}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        return self.model.get_model_info()