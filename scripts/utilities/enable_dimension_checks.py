#!/usr/bin/env python
"""
Script to enable dimension checking and consistency features.

This script enables dimension validation, consistency checks, and sets
appropriate dimension mismatch handling strategies throughout the system.
Run this script before analysis to ensure consistent dimensions between
train and test activations.

Usage:
    python enable_dimension_checks.py [strategy]

Where [strategy] is optional and can be one of:
    - 'warn' (default): Log warnings but don't modify activations
    - 'truncate': Truncate to smallest common dimension
    - 'pad': Pad to largest common dimension
    - 'error': Raise an exception on dimension mismatch
"""

import os
import sys
import argparse
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import dimension handling functions
from concept_fragmentation.hooks.activation_hooks import (
    set_dimension_logging,
    set_dimension_mismatch_strategy,
    get_dimension_mismatch_strategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dimension_checks.log")
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to enable dimension checks with specified strategy.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enable dimension checking and consistency features")
    parser.add_argument(
        "strategy", 
        nargs="?", 
        default="warn",
        choices=["warn", "truncate", "pad", "error"],
        help="Strategy for handling dimension mismatches (default: warn)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging of dimensions"
    )
    
    args = parser.parse_args()
    
    # Enable dimension logging if verbose flag is set
    set_dimension_logging(args.verbose)
    logger.info(f"Dimension logging {'enabled' if args.verbose else 'disabled'}")
    
    # Set dimension mismatch strategy
    set_dimension_mismatch_strategy(args.strategy)
    logger.info(f"Dimension mismatch strategy set to: {args.strategy}")
    
    print(f"""
Dimension checking enabled successfully!
------------------------------------------
Strategy: {args.strategy}
Verbose logging: {'Enabled' if args.verbose else 'Disabled'}

Description of '{args.strategy}' strategy:
{get_strategy_description(args.strategy)}

Now run your analysis scripts with automatic dimension handling.
""")


def get_strategy_description(strategy):
    """
    Get a human-readable description of the dimension mismatch strategy.
    """
    descriptions = {
        "warn": "Log warnings about dimension mismatches but don't modify activations. "
                "This is useful for diagnostics without changing behavior.",
        
        "truncate": "Automatically truncate activations to the smallest common dimension. "
                    "This ensures consistency by using only the overlapping features.",
        
        "pad": "Automatically pad activations to the largest common dimension. "
               "This preserves all features by filling missing dimensions with zeros.",
        
        "error": "Raise an exception when a dimension mismatch is detected. "
                "This is the strictest option, forcing you to fix mismatches rather than working around them."
    }
    
    return descriptions.get(strategy, "No description available.")


if __name__ == "__main__":
    main()