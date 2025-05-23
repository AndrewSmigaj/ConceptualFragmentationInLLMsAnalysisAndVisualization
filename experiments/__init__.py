"""
Experiments module with backward compatibility imports.

This module ensures that code can still import from the root directory
while we transition to the new structure.
"""
import sys
from pathlib import Path

# Add experiment directories to path for backward compatibility
experiments_dir = Path(__file__).parent

# Add GPT-2 shared utilities to path
sys.path.insert(0, str(experiments_dir / "gpt2" / "shared"))

# Add specific experiment directories if needed
sys.path.insert(0, str(experiments_dir / "gpt2" / "pivot"))
sys.path.insert(0, str(experiments_dir / "gpt2" / "pos"))
sys.path.insert(0, str(experiments_dir / "gpt2" / "semantic_subtypes"))