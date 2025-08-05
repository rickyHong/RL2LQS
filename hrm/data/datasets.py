"""
Dataset implementations for HRM reasoning tasks
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import random
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path


class SudokuDataset(Dataset):
    """Dataset for Sudoku solving tasks"""
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 num_samples: int = 1000,
                 difficulty: str = "mixed",
                 generate_on_fly: bool = True):
        """
        Args:
            data_path: Path to pre-generated Sudoku data
            num_samples: Number of samples to generate
            difficulty: Difficulty level ('easy', 'medium', 'hard', 'extreme', 'mixed')
            generate_on_fly: Whether to generate puzzles on the fly
        """
        self.num_samples = num_samples
        self.difficulty = difficulty
        self.generate_on_fly = generate_on_fly
        
        if data_path and Path(data_path).exists():
            self.data = self._load_data(data_path)
        else:
            self.data = self._generate_sudoku_data()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.generate_on_fly:
            puzzle, solution = self._generate_single_sudoku()
        else:
            puzzle, solution = self.data[idx]
        
        # Convert to tensors
        puzzle_tensor = torch.FloatTensor(puzzle).flatten()  # 81-dimensional
        solution_tensor = torch.FloatTensor(solution).flatten()  # 81-dimensional
        
        return puzzle_tensor, solution_tensor
    
    def _load_data(self, data_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Load pre-generated Sudoku data"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        return [(np.array(item['puzzle']), np.array(item['solution'])) 
                for item in data]
    
    def _generate_sudoku_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate Sudoku puzzles and solutions"""
        data = []
        for _ in range(self.num_samples):
            puzzle, solution = self._generate_single_sudoku()
            data.append((puzzle, solution))
        return data
    
    def _generate_single_sudoku(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single Sudoku puzzle and solution"""
        # Start with a complete valid Sudoku grid
        solution = self._generate_complete_sudoku()
        
        # Create puzzle by removing numbers
        puzzle = solution.copy()
        
        # Determine number of cells to remove based on difficulty
        if self.difficulty == "easy":
            cells_to_remove = random.randint(35, 45)
        elif self.difficulty == "medium":
            cells_to_remove = random.randint(45, 55)
        elif self.difficulty == "hard":
            cells_to_remove = random.randint(55, 65)
        elif self.difficulty == "extreme":
            cells_to_remove = random.randint(65, 75)
        else:  # mixed
            cells_to_remove = random.randint(35, 75)
        
        # Remove cells
        positions = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(positions)
        
        for i, (row, col) in enumerate(positions[:cells_to_remove]):
            puzzle[row, col] = 0
        
        return puzzle, solution
    
    def _generate_complete_sudoku(self) -> np.ndarray:
        """Generate a complete valid Sudoku grid"""
        grid = np.zeros((9, 9), dtype=int)
        
        # Fill diagonal 3x3 boxes first
        for box in range(0, 9, 3):
            self._fill_box(grid, box, box)
        
        # Fill remaining cells
        self._solve_sudoku(grid)
        
        return grid
    
    def _fill_box(self, grid: np.ndarray, row: int, col: int):
        """Fill a 3x3 box with random valid numbers"""
        numbers = list(range(1, 10))
        random.shuffle(numbers)
        
        for i in range(3):
            for j in range(3):
                grid[row + i][col + j] = numbers[i * 3 + j]
    
    def _solve_sudoku(self, grid: np.ndarray) -> bool:
        """Solve Sudoku using backtracking"""
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    numbers = list(range(1, 10))
                    random.shuffle(numbers)  # Add randomness
                    
                    for num in numbers:
                        if self._is_valid_move(grid, i, j, num):
                            grid[i][j] = num
                            
                            if self._solve_sudoku(grid):
                                return True
                            
                            grid[i][j] = 0
                    
                    return False
        return True
    
    def _is_valid_move(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid"""
        # Check row
        if num in grid[row]:
            return False
        
        # Check column
        if num in grid[:, col]:
            return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in grid[box_row:box_row + 3, box_col:box_col + 3]:
            return False
        
        return True


class MazeDataset(Dataset):
    """Dataset for maze pathfinding tasks"""
    
    def __init__(self,
                 data_path: Optional[str] = None,
                 num_samples: int = 1000,
                 maze_size: int = 30,
                 complexity: float = 0.3):
        """
        Args:
            data_path: Path to pre-generated maze data
            num_samples: Number of samples to generate
            maze_size: Size of the maze (maze_size x maze_size)
            complexity: Maze complexity (0.0 to 1.0)
        """
        self.num_samples = num_samples
        self.maze_size = maze_size
        self.complexity = complexity
        
        if data_path and Path(data_path).exists():
            self.data = self._load_data(data_path)
        else:
            self.data = self._generate_maze_data()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        maze, path = self.data[idx]
        
        # Convert to tensors
        maze_tensor = torch.FloatTensor(maze).flatten()  # maze_size^2 dimensional
        path_tensor = torch.FloatTensor(path).flatten()  # maze_size^2 dimensional
        
        return maze_tensor, path_tensor
    
    def _load_data(self, data_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Load pre-generated maze data"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        return [(np.array(item['maze']), np.array(item['path'])) 
                for item in data]
    
    def _generate_maze_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate maze and optimal path data"""
        data = []
        for _ in range(self.num_samples):
            maze, path = self._generate_single_maze()
            data.append((maze, path))
        return data
    
    def _generate_single_maze(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single maze and its optimal path"""
        # Generate maze using recursive backtracking
        maze = np.ones((self.maze_size, self.maze_size))  # 1 = wall, 0 = path
        
        # Create paths
        self._carve_maze(maze, 1, 1)
        
        # Ensure start and end are open
        maze[0, 0] = 0  # Start
        maze[self.maze_size - 1, self.maze_size - 1] = 0  # End
        
        # Find optimal path using A*
        path = self._find_optimal_path(maze)
        
        return maze, path
    
    def _carve_maze(self, maze: np.ndarray, x: int, y: int):
        """Carve maze paths using recursive backtracking"""
        maze[x, y] = 0
        
        # Directions: up, right, down, left
        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if (0 < nx < self.maze_size - 1 and 
                0 < ny < self.maze_size - 1 and 
                maze[nx, ny] == 1):
                
                maze[x + dx // 2, y + dy // 2] = 0
                self._carve_maze(maze, nx, ny)
    
    def _find_optimal_path(self, maze: np.ndarray) -> np.ndarray:
        """Find optimal path using A* algorithm"""
        from heapq import heappush, heappop
        
        start = (0, 0)
        goal = (self.maze_size - 1, self.maze_size - 1)
        
        # A* implementation
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal:
                break
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (0 <= neighbor[0] < self.maze_size and
                    0 <= neighbor[1] < self.maze_size and
                    maze[neighbor[0], neighbor[1]] == 0):
                    
                    tentative_g = g_score[current] + 1
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                        heappush(open_set, (f_score[neighbor], neighbor))
        
        # Reconstruct path
        path_array = np.zeros_like(maze)
        current = goal
        
        while current in came_from:
            path_array[current[0], current[1]] = 1
            current = came_from[current]
        path_array[start[0], start[1]] = 1
        
        return path_array
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


class ARCDataset(Dataset):
    """Dataset for Abstraction and Reasoning Corpus (ARC) tasks"""
    
    def __init__(self,
                 data_path: Optional[str] = None,
                 num_samples: int = 1000,
                 grid_size: int = 30):
        """
        Args:
            data_path: Path to ARC data
            num_samples: Number of samples to generate
            grid_size: Maximum grid size
        """
        self.num_samples = num_samples
        self.grid_size = grid_size
        
        if data_path and Path(data_path).exists():
            self.data = self._load_arc_data(data_path)
        else:
            # Generate synthetic ARC-like tasks
            self.data = self._generate_synthetic_arc_data()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_grid, output_grid = self.data[idx]
        
        # Pad grids to fixed size
        input_padded = self._pad_grid(input_grid)
        output_padded = self._pad_grid(output_grid)
        
        # Convert to tensors
        input_tensor = torch.FloatTensor(input_padded).flatten()
        output_tensor = torch.FloatTensor(output_padded).flatten()
        
        return input_tensor, output_tensor
    
    def _load_arc_data(self, data_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Load ARC data from JSON file"""
        with open(data_path, 'r') as f:
            arc_data = json.load(f)
        
        data = []
        for task in arc_data.values():
            for example in task['train']:
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                data.append((input_grid, output_grid))
        
        return data[:self.num_samples]
    
    def _generate_synthetic_arc_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate synthetic ARC-like tasks"""
        data = []
        
        for _ in range(self.num_samples):
            # Generate different types of synthetic tasks
            task_type = random.choice(['copy', 'rotate', 'mirror', 'color_change', 'pattern'])
            
            if task_type == 'copy':
                input_grid, output_grid = self._generate_copy_task()
            elif task_type == 'rotate':
                input_grid, output_grid = self._generate_rotation_task()
            elif task_type == 'mirror':
                input_grid, output_grid = self._generate_mirror_task()
            elif task_type == 'color_change':
                input_grid, output_grid = self._generate_color_change_task()
            else:  # pattern
                input_grid, output_grid = self._generate_pattern_task()
            
            data.append((input_grid, output_grid))
        
        return data
    
    def _generate_copy_task(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a simple copy task"""
        size = random.randint(3, min(10, self.grid_size))
        grid = np.random.randint(0, 10, (size, size))
        return grid, grid.copy()
    
    def _generate_rotation_task(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a rotation task"""
        size = random.randint(3, min(10, self.grid_size))
        input_grid = np.random.randint(0, 10, (size, size))
        
        # Rotate 90 degrees
        output_grid = np.rot90(input_grid)
        
        return input_grid, output_grid
    
    def _generate_mirror_task(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a mirror task"""
        size = random.randint(3, min(10, self.grid_size))
        input_grid = np.random.randint(0, 10, (size, size))
        
        # Mirror horizontally
        output_grid = np.fliplr(input_grid)
        
        return input_grid, output_grid
    
    def _generate_color_change_task(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a color change task"""
        size = random.randint(3, min(10, self.grid_size))
        input_grid = np.random.randint(0, 5, (size, size))
        
        # Change specific color
        old_color = random.randint(0, 4)
        new_color = random.randint(5, 9)
        
        output_grid = input_grid.copy()
        output_grid[output_grid == old_color] = new_color
        
        return input_grid, output_grid
    
    def _generate_pattern_task(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a pattern completion task"""
        size = random.randint(4, min(8, self.grid_size))
        input_grid = np.zeros((size, size))
        
        # Create a simple pattern
        pattern_size = size // 2
        pattern = np.random.randint(1, 5, (pattern_size, pattern_size))
        
        # Place pattern in top-left
        input_grid[:pattern_size, :pattern_size] = pattern
        
        # Output should complete the pattern
        output_grid = input_grid.copy()
        output_grid[pattern_size:, :pattern_size] = pattern
        output_grid[:pattern_size, pattern_size:] = pattern
        output_grid[pattern_size:, pattern_size:] = pattern
        
        return input_grid, output_grid
    
    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        """Pad grid to fixed size"""
        padded = np.zeros((self.grid_size, self.grid_size))
        h, w = grid.shape
        
        # Place in top-left corner
        padded[:min(h, self.grid_size), :min(w, self.grid_size)] = \
            grid[:min(h, self.grid_size), :min(w, self.grid_size)]
        
        return padded


class CustomDataset(Dataset):
    """Custom dataset for user-defined tasks"""
    
    def __init__(self, 
                 inputs: List[np.ndarray],
                 targets: List[np.ndarray]):
        """
        Args:
            inputs: List of input arrays
            targets: List of target arrays
        """
        assert len(inputs) == len(targets), "Inputs and targets must have same length"
        
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor = torch.FloatTensor(self.inputs[idx])
        target_tensor = torch.FloatTensor(self.targets[idx])
        
        return input_tensor, target_tensor