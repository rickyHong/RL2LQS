"""
Data utilities for HRM
"""
import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional, Dict, Any
import logging

from .datasets import SudokuDataset, MazeDataset, ARCDataset, CustomDataset

logger = logging.getLogger(__name__)


def create_data_loaders(dataset: torch.utils.data.Dataset,
                       batch_size: int = 32,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       num_workers: int = 0,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders from a dataset
    
    Args:
        dataset: The dataset to split
        batch_size: Batch size for data loaders
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        logger.warning(f"Ratios sum to {total_ratio}, normalizing...")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logger.info(f"Created data loaders - Train: {len(train_loader)}, "
               f"Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def get_dataset(dataset_name: str, 
               config: Dict[str, Any],
               **kwargs) -> torch.utils.data.Dataset:
    """
    Get dataset by name
    
    Args:
        dataset_name: Name of the dataset
        config: Dataset configuration
        **kwargs: Additional arguments
        
    Returns:
        Dataset instance
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "sudoku":
        return SudokuDataset(
            num_samples=config.get('train_size', 1000),
            difficulty=config.get('difficulty', 'mixed'),
            **kwargs
        )
    elif dataset_name == "maze":
        return MazeDataset(
            num_samples=config.get('train_size', 1000),
            maze_size=config.get('maze_size', 30),
            complexity=config.get('complexity', 0.3),
            **kwargs
        )
    elif dataset_name == "arc":
        return ARCDataset(
            num_samples=config.get('train_size', 1000),
            grid_size=config.get('grid_size', 30),
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def preprocess_sudoku(puzzle: torch.Tensor, solution: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess Sudoku data"""
    # Normalize to [0, 1]
    puzzle = puzzle / 9.0
    solution = solution / 9.0
    return puzzle, solution


def preprocess_maze(maze: torch.Tensor, path: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess maze data"""
    # Already in [0, 1] range
    return maze, path


def preprocess_arc(input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess ARC data"""
    # Normalize to [0, 1]
    input_grid = input_grid / 9.0
    output_grid = output_grid / 9.0
    return input_grid, output_grid