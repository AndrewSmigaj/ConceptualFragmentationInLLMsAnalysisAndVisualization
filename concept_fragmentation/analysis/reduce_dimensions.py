#!/usr/bin/env python
"""
Command-line utility for dimensionality reduction on transformer activations.

This script provides a command-line interface to apply the enhanced dimensionality 
reduction techniques on activation files, saving the reduced representations
for further analysis.
"""

import argparse
import os
import sys
import numpy as np
import logging
from pathlib import Path
import pickle
import json
from typing import Dict, List, Any, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reduce_dimensions')

# Add parent directory to path to import the module
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import the dimensionality reduction module
from concept_fragmentation.analysis.transformer_dimensionality import (
    TransformerDimensionalityReducer,
    DimensionalityReductionResult
)


def load_activations(file_path: str) -> Dict[str, Any]:
    """
    Load activations from a file.
    
    Args:
        file_path: Path to the activations file (pickle or numpy)
        
    Returns:
        Dictionary mapping layer names to activation arrays
    """
    file_path = Path(file_path)
    file_ext = file_path.suffix.lower()
    
    try:
        if file_ext == '.pkl' or file_ext == '.pickle':
            with open(file_path, 'rb') as f:
                activations = pickle.load(f)
                logger.info(f"Loaded activations from pickle file: {file_path}")
                return activations
        elif file_ext == '.npy':
            activations = np.load(file_path, allow_pickle=True)
            if isinstance(activations, np.ndarray) and activations.dtype == np.dtype('O'):
                # Handle object arrays (likely containing a dictionary)
                if len(activations.shape) == 0:  # This is a 0-d array containing a dict
                    return activations.item()
            
            # If it's just an array, wrap it in a dictionary with a default name
            logger.info(f"Loaded activations from numpy file: {file_path}")
            return {"activations": activations}
        elif file_ext == '.npz':
            npz_data = np.load(file_path, allow_pickle=True)
            activations = {}
            for key in npz_data.files:
                activations[key] = npz_data[key]
            logger.info(f"Loaded activations from npz file: {file_path}")
            return activations
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    except Exception as e:
        logger.error(f"Error loading activations from {file_path}: {e}")
        raise


def save_reduced_activations(
    reduced_activations: Dict[str, Any],
    metadata: Dict[str, Any],
    output_path: str,
    format: str = 'npz'
) -> None:
    """
    Save reduced activations to a file.
    
    Args:
        reduced_activations: Dictionary mapping layer names to reduced activations
        metadata: Dictionary with reduction metadata
        output_path: Path to save the reduced activations
        format: Output format ('npz', 'pkl', or 'both')
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    if format in ['npz', 'both']:
        # Save as npz
        npz_path = output_path.with_suffix('.npz')
        np.savez_compressed(
            npz_path,
            **reduced_activations
        )
        logger.info(f"Saved reduced activations to npz file: {npz_path}")
        
        # Save metadata as JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            serializable_metadata = {}
            for layer, layer_meta in metadata.items():
                serializable_metadata[layer] = {
                    k: v for k, v in layer_meta.items() 
                    if not isinstance(v, (np.ndarray, np.number))
                }
                # Convert numpy values to Python types
                for k, v in layer_meta.items():
                    if isinstance(v, np.number):
                        serializable_metadata[layer][k] = v.item()
            
            json.dump(serializable_metadata, f, indent=2)
        logger.info(f"Saved reduction metadata to json file: {json_path}")
    
    if format in ['pkl', 'both']:
        # Save as pickle
        pkl_path = output_path.with_suffix('.pkl')
        with open(pkl_path, 'wb') as f:
            combined_data = {
                "reduced_activations": reduced_activations,
                "metadata": metadata
            }
            pickle.dump(combined_data, f)
        logger.info(f"Saved reduced activations and metadata to pickle file: {pkl_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Reduce dimensionality of transformer model activations"
    )
    
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the activations file (.pkl, .npy, or .npz)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to save the reduced activations"
    )
    parser.add_argument(
        "--components", "-c", type=int, default=50,
        help="Number of components for dimensionality reduction (default: 50)"
    )
    parser.add_argument(
        "--method", "-m", default="auto", 
        choices=["auto", "pca", "umap", "truncated_svd", "kernel_pca", "random_projection"],
        help="Dimensionality reduction method (default: auto)"
    )
    parser.add_argument(
        "--progressive", "-p", action="store_true",
        help="Use progressive dimensionality reduction for very high-dimensional data"
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Directory for caching reduction results"
    )
    parser.add_argument(
        "--format", "-f", default="npz", choices=["npz", "pkl", "both"],
        help="Output format (default: npz)"
    )
    parser.add_argument(
        "--filter", default=None,
        help="Regex pattern to filter layer names"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load activations
        logger.info(f"Loading activations from {args.input}")
        all_activations = load_activations(args.input)
        
        # Create dimensionality reducer
        reducer = TransformerDimensionalityReducer(
            cache_dir=args.cache_dir,
            random_state=args.random_seed,
            use_cache=True,
            verbose=args.verbose
        )
        
        # Filter layers if requested
        if args.filter:
            import re
            pattern = re.compile(args.filter)
            filtered_activations = {}
            
            for layer_name, activations in all_activations.items():
                if pattern.search(layer_name):
                    filtered_activations[layer_name] = activations
            
            logger.info(f"Filtered {len(filtered_activations)} of {len(all_activations)} layers with pattern: {args.filter}")
            all_activations = filtered_activations
        
        # Process each layer
        reduced_activations = {}
        reduction_metadata = {}
        
        for layer_name, activations in all_activations.items():
            # Skip non-array items
            if not isinstance(activations, (np.ndarray, list)):
                logger.info(f"Skipping non-array item: {layer_name}")
                continue
                
            # Convert to numpy array if needed
            if isinstance(activations, list):
                try:
                    activations = np.array(activations)
                except Exception as e:
                    logger.warning(f"Could not convert {layer_name} to numpy array: {e}")
                    continue
            
            logger.info(f"Processing layer: {layer_name} with shape {activations.shape}")
            
            # Skip if already low-dimensional
            if len(activations.shape) == 2 and activations.shape[1] <= args.components:
                logger.info(f"Layer {layer_name} is already low-dimensional ({activations.shape[1]} <= {args.components})")
                reduced_activations[layer_name] = activations
                reduction_metadata[layer_name] = {
                    "original_dim": activations.shape[1],
                    "reduced_dim": activations.shape[1],
                    "method": "identity",
                    "success": True
                }
                continue
            
            try:
                # Apply dimensionality reduction
                if args.progressive and (
                    len(activations.shape) == 2 and activations.shape[1] > 500 or
                    len(activations.shape) > 2 and activations.shape[-1] > 500
                ):
                    # Use progressive reduction for high-dimensional data
                    logger.info(f"Using progressive reduction for layer {layer_name}")
                    result = reducer.progressive_dimensionality_reduction(
                        activations=activations,
                        target_dim=args.components,
                        layer_name=layer_name
                    )
                else:
                    # Use standard reduction
                    result = reducer.reduce_dimensionality(
                        activations=activations,
                        n_components=args.components,
                        method=args.method,
                        layer_name=layer_name
                    )
                
                if result.success:
                    reduced_activations[layer_name] = result.reduced_activations
                    
                    # Store reduction metadata
                    reduction_metadata[layer_name] = {
                        "original_dim": result.original_dim,
                        "reduced_dim": result.reduced_dim,
                        "method": result.method,
                        "success": True,
                        "n_components": args.components
                    }
                    
                    # Add explained variance if available
                    if result.explained_variance is not None:
                        reduction_metadata[layer_name]["explained_variance"] = result.explained_variance
                    
                    logger.info(f"Reduced {layer_name} from {result.original_dim} to {result.reduced_dim} dimensions using {result.method}")
                else:
                    logger.warning(f"Reduction failed for {layer_name}: {result.error_message}")
                    reduced_activations[layer_name] = activations
                    reduction_metadata[layer_name] = {
                        "original_dim": result.original_dim,
                        "reduced_dim": result.original_dim,
                        "method": "failed",
                        "success": False,
                        "error": result.error_message
                    }
            except Exception as e:
                logger.error(f"Error processing layer {layer_name}: {e}")
                reduced_activations[layer_name] = activations
                reduction_metadata[layer_name] = {
                    "original_dim": activations.shape[-1] if len(activations.shape) > 1 else 1,
                    "reduced_dim": activations.shape[-1] if len(activations.shape) > 1 else 1,
                    "method": "error",
                    "success": False,
                    "error": str(e)
                }
        
        # Save reduced activations
        save_reduced_activations(
            reduced_activations=reduced_activations,
            metadata=reduction_metadata,
            output_path=args.output,
            format=args.format
        )
        
        # Print summary
        success_count = sum(1 for meta in reduction_metadata.values() if meta["success"])
        total_count = len(reduction_metadata)
        
        logger.info(f"Dimensionality reduction summary:")
        logger.info(f"- Successfully reduced {success_count} of {total_count} layers")
        logger.info(f"- Methods used: {set(meta['method'] for meta in reduction_metadata.values())}")
        logger.info(f"- Output saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())