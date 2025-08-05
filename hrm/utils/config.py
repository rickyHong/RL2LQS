"""
Configuration management for HRM
"""
import yaml
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    input_dim: int = 256
    hidden_dim: int = 512
    high_level_dim: int = 256
    low_level_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm: bool = True
    use_residual: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    save_every: int = 10
    eval_every: int = 5
    early_stopping_patience: int = 20
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    mixed_precision: bool = True


@dataclass
class ReasoningConfig:
    """Reasoning-specific configuration"""
    max_high_level_cycles: int = 10
    max_low_level_steps: int = 20
    adaptive_computation: bool = True
    halt_threshold: float = 0.95
    min_cycles: int = 1
    max_reasoning_depth: int = 100
    use_act_loss: bool = True
    act_loss_weight: float = 0.01


@dataclass
class DataConfig:
    """Data configuration"""
    dataset_name: str = "sudoku"
    data_path: str = "./data"
    train_size: int = 1000
    val_size: int = 200
    test_size: int = 200
    max_sequence_length: int = 512
    preprocessing: Dict[str, Any] = None
    augmentation: bool = False


@dataclass
class HRMConfig:
    """Main HRM configuration"""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    reasoning: ReasoningConfig = ReasoningConfig()
    data: DataConfig = DataConfig()
    
    # Experiment settings
    experiment_name: str = "hrm_experiment"
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    log_level: str = "INFO"
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    def __post_init__(self):
        if self.data.preprocessing is None:
            self.data.preprocessing = {}
    
    @classmethod
    def from_yaml(cls, path: str) -> 'HRMConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> 'HRMConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HRMConfig':
        """Create configuration from dictionary"""
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        reasoning_config = ReasoningConfig(**config_dict.get('reasoning', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        # Extract main level configs
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['model', 'training', 'reasoning', 'data']}
        
        return cls(
            model=model_config,
            training=training_config,
            reasoning=reasoning_config,
            data=data_config,
            **main_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'reasoning': asdict(self.reasoning),
            'data': asdict(self.data),
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'device': self.device,
            'log_level': self.log_level,
            'save_dir': self.save_dir,
            'log_dir': self.log_dir,
        }
    
    def save_yaml(self, path: str):
        """Save configuration to YAML file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def save_json(self, path: str):
        """Save configuration to JSON file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_default_config(task: str = "sudoku") -> HRMConfig:
    """Get default configuration for specific task"""
    config = HRMConfig()
    
    if task == "sudoku":
        config.data.dataset_name = "sudoku"
        config.model.input_dim = 81  # 9x9 grid
        config.reasoning.max_high_level_cycles = 15
        config.reasoning.max_low_level_steps = 25
        
    elif task == "maze":
        config.data.dataset_name = "maze"
        config.model.input_dim = 900  # 30x30 grid
        config.reasoning.max_high_level_cycles = 20
        config.reasoning.max_low_level_steps = 30
        
    elif task == "arc":
        config.data.dataset_name = "arc"
        config.model.input_dim = 900  # 30x30 grid
        config.reasoning.max_high_level_cycles = 25
        config.reasoning.max_low_level_steps = 35
        
    return config