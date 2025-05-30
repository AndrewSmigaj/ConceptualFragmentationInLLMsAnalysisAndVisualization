"""Input validation utilities."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


def validate_path(path: Union[str, Path], 
                 must_exist: bool = False,
                 create_parent: bool = False) -> Path:
    """Validate and convert path.
    
    Args:
        path: Input path
        must_exist: Whether path must exist
        create_parent: Whether to create parent directory
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid
        FileNotFoundError: If path doesn't exist and must_exist=True
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
        
    if create_parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        
    return path
    

def validate_data(data: Any,
                 expected_type: Optional[type] = None,
                 expected_shape: Optional[tuple] = None,
                 min_size: Optional[int] = None,
                 max_size: Optional[int] = None) -> Any:
    """Validate data structure and shape.
    
    Args:
        data: Data to validate
        expected_type: Expected data type
        expected_shape: Expected shape (for arrays)
        min_size: Minimum size/length
        max_size: Maximum size/length
        
    Returns:
        Validated data
        
    Raises:
        TypeError: If type doesn't match
        ValueError: If validation fails
    """
    # Type check
    if expected_type is not None:
        if not isinstance(data, expected_type):
            raise TypeError(f"Expected {expected_type}, got {type(data)}")
            
    # Array shape check
    if expected_shape is not None and hasattr(data, 'shape'):
        if data.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {data.shape}")
            
    # Size checks
    if hasattr(data, '__len__'):
        size = len(data)
        
        if min_size is not None and size < min_size:
            raise ValueError(f"Size {size} is less than minimum {min_size}")
            
        if max_size is not None and size > max_size:
            raise ValueError(f"Size {size} is greater than maximum {max_size}")
            
    return data
    

def validate_config(config: Dict[str, Any],
                   required_keys: Optional[List[str]] = None,
                   allowed_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: Keys that must be present
        allowed_keys: Keys that are allowed (if None, all keys allowed)
        
    Returns:
        Validated configuration
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(config, dict):
        raise TypeError(f"Config must be dict, got {type(config)}")
        
    # Check required keys
    if required_keys:
        missing = set(required_keys) - set(config.keys())
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
            
    # Check allowed keys
    if allowed_keys is not None:
        extra = set(config.keys()) - set(allowed_keys)
        if extra:
            raise ValueError(f"Unknown config keys: {extra}")
            
    return config