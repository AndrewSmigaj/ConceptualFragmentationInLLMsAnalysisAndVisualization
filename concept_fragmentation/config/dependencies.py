"""
Dependency checking for the configuration system.

This module provides functions to check that all required dependencies for
the configuration system are available.
"""

import importlib
import logging
from typing import Dict, List, Tuple, Optional

# Set up logger
logger = logging.getLogger(__name__)

# Define required dependencies
REQUIRED_DEPENDENCIES = [
    ("yaml", "PyYAML", "5.1"),
    ("torch", "torch", "1.0.0"),  # Optional for device initialization
]


def check_dependency(module_name: str, package_name: str, min_version: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a dependency is available and meets the minimum version.
    
    Args:
        module_name: The name of the module to import
        package_name: The name of the package (for error messages)
        min_version: The minimum required version
        
    Returns:
        A tuple of (is_available, error_message)
    """
    try:
        module = importlib.import_module(module_name)
        
        # Check if it has a version attribute
        if hasattr(module, "__version__"):
            version = module.__version__
            
            # Very simple version comparison - just checks if the version
            # string is >= the minimum version string
            if version < min_version:
                return False, f"{package_name} version {version} is older than the required version {min_version}"
        
        return True, None
    except ImportError:
        return False, f"{package_name} is not installed"


def check_dependencies() -> Dict[str, Tuple[bool, Optional[str]]]:
    """
    Check all required dependencies and return their status.
    
    Returns:
        A dictionary mapping package names to (is_available, error_message) tuples
    """
    results = {}
    
    for module_name, package_name, min_version in REQUIRED_DEPENDENCIES:
        is_available, error_message = check_dependency(module_name, package_name, min_version)
        
        if not is_available:
            logger.warning(f"Dependency check for {package_name}: {error_message}")
        else:
            logger.debug(f"Dependency check for {package_name}: OK")
        
        results[package_name] = (is_available, error_message)
    
    return results


def check_yaml() -> bool:
    """
    Check if the YAML package is available.
    
    This is a convenience function used elsewhere in the codebase.
    
    Returns:
        True if YAML is available, False otherwise
    """
    is_available, _ = check_dependency("yaml", "PyYAML", "5.1")
    return is_available


def check_torch() -> bool:
    """
    Check if PyTorch is available.
    
    This is a convenience function used elsewhere in the codebase.
    
    Returns:
        True if PyTorch is available, False otherwise
    """
    is_available, _ = check_dependency("torch", "torch", "1.0.0")
    return is_available