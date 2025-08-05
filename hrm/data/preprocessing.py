"""
Preprocessing utilities for different reasoning tasks
"""
import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


def preprocess_sudoku(puzzle: np.ndarray, solution: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess Sudoku puzzle and solution
    
    Args:
        puzzle: 9x9 Sudoku puzzle (0 for empty cells)
        solution: 9x9 Sudoku solution
        
    Returns:
        Preprocessed puzzle and solution tensors
    """
    # Flatten and normalize
    puzzle_flat = puzzle.flatten().astype(np.float32) / 9.0
    solution_flat = solution.flatten().astype(np.float32) / 9.0
    
    return torch.from_numpy(puzzle_flat), torch.from_numpy(solution_flat)


def preprocess_maze(maze: np.ndarray, path: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess maze and optimal path
    
    Args:
        maze: 2D maze array (1 for walls, 0 for paths)
        path: 2D path array (1 for optimal path, 0 otherwise)
        
    Returns:
        Preprocessed maze and path tensors
    """
    # Flatten and convert to float
    maze_flat = maze.flatten().astype(np.float32)
    path_flat = path.flatten().astype(np.float32)
    
    return torch.from_numpy(maze_flat), torch.from_numpy(path_flat)


def preprocess_arc(input_grid: np.ndarray, output_grid: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess ARC input and output grids
    
    Args:
        input_grid: Input grid
        output_grid: Output grid
        
    Returns:
        Preprocessed input and output tensors
    """
    # Flatten and normalize
    input_flat = input_grid.flatten().astype(np.float32) / 9.0
    output_flat = output_grid.flatten().astype(np.float32) / 9.0
    
    return torch.from_numpy(input_flat), torch.from_numpy(output_flat)


def augment_sudoku(puzzle: np.ndarray, solution: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Augment Sudoku data with rotations and reflections
    
    Args:
        puzzle: Original puzzle
        solution: Original solution
        
    Returns:
        List of augmented (puzzle, solution) pairs
    """
    augmented = [(puzzle, solution)]
    
    # Rotations
    for k in range(1, 4):
        rot_puzzle = np.rot90(puzzle, k)
        rot_solution = np.rot90(solution, k)
        augmented.append((rot_puzzle, rot_solution))
    
    # Reflections
    flip_puzzle = np.fliplr(puzzle)
    flip_solution = np.fliplr(solution)
    augmented.append((flip_puzzle, flip_solution))
    
    flip_puzzle = np.flipud(puzzle)
    flip_solution = np.flipud(solution)
    augmented.append((flip_puzzle, flip_solution))
    
    return augmented


def augment_maze(maze: np.ndarray, path: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Augment maze data with rotations and reflections
    
    Args:
        maze: Original maze
        path: Original path
        
    Returns:
        List of augmented (maze, path) pairs
    """
    augmented = [(maze, path)]
    
    # Rotations
    for k in range(1, 4):
        rot_maze = np.rot90(maze, k)
        rot_path = np.rot90(path, k)
        augmented.append((rot_maze, rot_path))
    
    return augmented


def create_difficulty_curriculum(dataset: List[Tuple[np.ndarray, np.ndarray]], 
                               difficulty_metric: str = "sparsity") -> List[float]:
    """
    Create difficulty scores for curriculum learning
    
    Args:
        dataset: List of (input, target) pairs
        difficulty_metric: Metric to use for difficulty assessment
        
    Returns:
        List of difficulty scores (0.0 to 1.0)
    """
    difficulties = []
    
    for input_data, target_data in dataset:
        if difficulty_metric == "sparsity":
            # For Sudoku: fewer given numbers = harder
            # For other tasks: more zeros = potentially harder
            sparsity = np.mean(input_data == 0)
            difficulties.append(sparsity)
        elif difficulty_metric == "complexity":
            # Measure structural complexity
            complexity = np.std(input_data.flatten())
            difficulties.append(complexity)
        else:
            # Default: random difficulty
            difficulties.append(np.random.random())
    
    # Normalize to [0, 1]
    difficulties = np.array(difficulties)
    if len(difficulties) > 1:
        difficulties = (difficulties - difficulties.min()) / (difficulties.max() - difficulties.min())
    
    return difficulties.tolist()


def batch_preprocess(batch_data: List[Tuple[np.ndarray, np.ndarray]], 
                    task_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess a batch of data
    
    Args:
        batch_data: List of (input, target) pairs
        task_type: Type of task ('sudoku', 'maze', 'arc')
        
    Returns:
        Batched input and target tensors
    """
    inputs = []
    targets = []
    
    for input_data, target_data in batch_data:
        if task_type == "sudoku":
            input_tensor, target_tensor = preprocess_sudoku(input_data, target_data)
        elif task_type == "maze":
            input_tensor, target_tensor = preprocess_maze(input_data, target_data)
        elif task_type == "arc":
            input_tensor, target_tensor = preprocess_arc(input_data, target_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        inputs.append(input_tensor)
        targets.append(target_tensor)
    
    return torch.stack(inputs), torch.stack(targets)


def normalize_features(data: torch.Tensor, 
                      method: str = "minmax") -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Normalize features
    
    Args:
        data: Input data tensor
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized data and normalization parameters
    """
    if method == "minmax":
        min_val = data.min()
        max_val = data.max()
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params = {"min": min_val.item(), "max": max_val.item()}
    
    elif method == "zscore":
        mean_val = data.mean()
        std_val = data.std()
        normalized = (data - mean_val) / (std_val + 1e-8)
        params = {"mean": mean_val.item(), "std": std_val.item()}
    
    elif method == "robust":
        median_val = data.median()
        mad_val = torch.median(torch.abs(data - median_val))
        normalized = (data - median_val) / (mad_val + 1e-8)
        params = {"median": median_val.item(), "mad": mad_val.item()}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_features(data: torch.Tensor, 
                        params: Dict[str, float], 
                        method: str = "minmax") -> torch.Tensor:
    """
    Denormalize features
    
    Args:
        data: Normalized data tensor
        params: Normalization parameters
        method: Normalization method used
        
    Returns:
        Denormalized data
    """
    if method == "minmax":
        return data * (params["max"] - params["min"]) + params["min"]
    elif method == "zscore":
        return data * params["std"] + params["mean"]
    elif method == "robust":
        return data * params["mad"] + params["median"]
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def add_noise(data: torch.Tensor, 
              noise_type: str = "gaussian", 
              noise_level: float = 0.01) -> torch.Tensor:
    """
    Add noise to data for regularization
    
    Args:
        data: Input data
        noise_type: Type of noise ('gaussian', 'uniform', 'dropout')
        noise_level: Noise intensity
        
    Returns:
        Noisy data
    """
    if noise_type == "gaussian":
        noise = torch.randn_like(data) * noise_level
        return data + noise
    
    elif noise_type == "uniform":
        noise = (torch.rand_like(data) - 0.5) * 2 * noise_level
        return data + noise
    
    elif noise_type == "dropout":
        mask = torch.rand_like(data) > noise_level
        return data * mask.float()
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def create_masks(data: torch.Tensor, 
                mask_ratio: float = 0.15, 
                mask_strategy: str = "random") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create masks for masked learning
    
    Args:
        data: Input data
        mask_ratio: Ratio of elements to mask
        mask_strategy: Masking strategy ('random', 'block', 'structured')
        
    Returns:
        Masked data and mask tensor
    """
    mask = torch.ones_like(data)
    
    if mask_strategy == "random":
        num_mask = int(data.numel() * mask_ratio)
        indices = torch.randperm(data.numel())[:num_mask]
        mask.view(-1)[indices] = 0
    
    elif mask_strategy == "block":
        # Mask contiguous blocks
        block_size = int(np.sqrt(data.numel() * mask_ratio))
        start_idx = torch.randint(0, data.numel() - block_size, (1,))
        mask.view(-1)[start_idx:start_idx + block_size] = 0
    
    else:
        raise ValueError(f"Unknown mask strategy: {mask_strategy}")
    
    masked_data = data * mask
    
    return masked_data, mask