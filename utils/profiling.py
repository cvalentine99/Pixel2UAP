"""
Profiling utilities for performance optimization.
"""

import os
import time
import logging
import cProfile
import pstats
from functools import wraps
import numpy as np

logger = logging.getLogger(__name__)


def timeit(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: The function to time
        
    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def profile_function(func, output_dir=None):
    """
    Profile a function using cProfile.
    
    Args:
        func: The function to profile
        output_dir: Directory to save profiling results, defaults to logs directory
        
    Returns:
        Wrapped function that profiles execution
    """
    from voxel_projector_v2 import CONFIG_PATH
    
    if output_dir is None:
        output_dir = os.path.join(CONFIG_PATH, "profiles")
        os.makedirs(output_dir, exist_ok=True)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a profile object
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Disable profiling
        profiler.disable()
        
        # Save results
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        profile_file = os.path.join(output_dir, f"{func.__name__}_{timestamp}.prof")
        stats_file = os.path.join(output_dir, f"{func.__name__}_{timestamp}.stats")
        
        # Save binary profile data
        profiler.dump_stats(profile_file)
        
        # Create readable stats file
        with open(stats_file, 'w') as f:
            stats = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
            stats.print_stats(30)  # Print top 30 entries
            
        logger.info(f"Profile saved to {profile_file}")
        logger.info(f"Stats saved to {stats_file}")
        
        return result
    
    return wrapper


def memory_usage(obj):
    """
    Calculate memory usage of a numpy array or similar object.
    
    Args:
        obj: Object to check memory usage (should have nbytes attribute)
        
    Returns:
        Memory usage in MB or None if not available
    """
    try:
        if hasattr(obj, 'nbytes'):
            return obj.nbytes / (1024 * 1024)
        elif isinstance(obj, (list, tuple)):
            return sum(memory_usage(item) for item in obj if item is not None)
        else:
            return None
    except:
        return None


def log_memory_usage(array_dict):
    """
    Log memory usage of a dictionary of arrays.
    
    Args:
        array_dict: Dictionary of name: array pairs
    """
    logger.debug("Memory usage:")
    for name, array in array_dict.items():
        usage = memory_usage(array)
        if usage is not None:
            logger.debug(f"  - {name}: {usage:.2f} MB")
            
            
def optimize_voxel_grid(grid, threshold=0.0):
    """
    Optimize a voxel grid's memory usage by using appropriate data types
    and potentially conversion to sparse representation.
    
    Args:
        grid: NumPy array representing voxel grid
        threshold: Value considered as empty/background
        
    Returns:
        Optimized grid (dense or sparse) and info dictionary
    """
    from scipy import sparse
    
    info = {
        "original_shape": grid.shape,
        "original_dtype": grid.dtype,
        "original_size_mb": grid.nbytes / (1024 * 1024),
        "sparse_conversion": False,
    }
    
    # Check sparsity
    non_zero = np.sum(grid != threshold)
    sparsity = 1.0 - (non_zero / grid.size)
    info["sparsity"] = sparsity
    
    # Convert to appropriate data type based on values
    max_val = np.max(np.abs(grid))
    
    # For sparse matrices, use a more efficient format but keep dtype
    if sparsity > 0.9 and max(grid.shape) > 128:
        # Sparse conversion is beneficial
        sparse_grid = sparse.csr_matrix(grid.ravel()).reshape(grid.shape)
        info["sparse_conversion"] = True
        info["final_size_mb"] = (sparse_grid.data.nbytes + 
                                sparse_grid.indptr.nbytes + 
                                sparse_grid.indices.nbytes) / (1024 * 1024)
        info["compression_ratio"] = info["original_size_mb"] / info["final_size_mb"]
        logger.info(f"Converted voxel grid to sparse format. "
                   f"Compression ratio: {info['compression_ratio']:.2f}x")
        return sparse_grid, info
    
    # For dense matrices with smaller values, optimize dtype
    if max_val < 1.0 and grid.dtype != np.float32:
        # Convert to float32 for small values
        grid = grid.astype(np.float32)
        info["final_dtype"] = np.float32
    elif max_val < 255 and grid.dtype not in (np.uint8, np.int8):
        # Convert to uint8 for values < 255
        grid = grid.astype(np.uint8)
        info["final_dtype"] = np.uint8
    else:
        info["final_dtype"] = grid.dtype
    
    info["final_size_mb"] = grid.nbytes / (1024 * 1024)
    info["compression_ratio"] = info["original_size_mb"] / info["final_size_mb"]
    logger.info(f"Optimized voxel grid dtype to {info['final_dtype']}. "
                f"Compression ratio: {info['compression_ratio']:.2f}x")
    
    return grid, info