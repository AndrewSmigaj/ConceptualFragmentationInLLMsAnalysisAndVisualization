"""
Example script for using the GPT-2 adapter with Archetypal Path Analysis (APA).

This example demonstrates how to:
1. Load a GPT-2 model and extract activations
2. Analyze activations with APA
3. Visualize results 
4. Extract attention-weighted path analysis
"""

import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import relevant modules
from concept_fragmentation.models.transformer_adapter import GPT2Adapter
from concept_fragmentation.analysis.cluster_paths import (
    analyze_layer_paths,
    calculate_path_metrics
)
from concept_fragmentation.analysis.similarity_metrics import (
    calculate_cross_layer_similarity
)
from concept_fragmentation.hooks.activation_hooks import set_dimension_logging


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GPT-2 Archetypal Path Analysis")
    
    # Model and data options
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="GPT-2 model type to use"
    )
    parser.add_argument(
        "--input_text", 
        type=str, 
        default="The quick brown fox jumps over the lazy dog.",
        help="Text to analyze"
    )
    parser.add_argument(
        "--window_size", 
        type=int, 
        default=3,
        help="Number of layers in each window"
    )
    parser.add_argument(
        "--stride", 
        type=int, 
        default=1,
        help="Stride for sliding windows"
    )
    
    # Analysis options
    parser.add_argument(
        "--n_clusters", 
        type=int, 
        default=10,
        help="Number of clusters for path analysis"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./apa_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def main():
    """Main function for GPT-2 APA example."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        set_dimension_logging(True)
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load GPT-2 model
    logger.info(f"Loading GPT-2 model: {args.model_type}")
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        raise ImportError("transformers library required. Install with: pip install transformers")
    
    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(
        args.model_type,
        output_hidden_states=True,
        output_attentions=True
    )
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_type)
    
    model.to(device)
    model.eval()
    
    # Initialize adapter
    adapter = GPT2Adapter(model, tokenizer=tokenizer)
    
    # Extract activations
    logger.info("Extracting activations")
    input_text = args.input_text
    windows = adapter.extract_activations_for_windows(
        input_text,
        window_size=args.window_size,
        stride=args.stride
    )
    
    # Save activations
    logger.info("Saving activations")
    metadata = windows[list(windows.keys())[0]].get("metadata", {})
    
    # Save first window as example
    first_window = list(windows.keys())[0]
    window_activations = windows[first_window]["activations"]
    metadata_file = adapter.save_activations_with_metadata(
        window_activations,
        {"window_name": first_window, **metadata},
        output_dir=str(output_dir / "activations"),
        prefix="gpt2_example"
    )
    logger.info(f"Saved activations metadata to: {metadata_file}")
    
    # Run APA analysis for each window
    logger.info("Running APA analysis for each window")
    results = {}
    
    for window_name, window_data in windows.items():
        logger.info(f"Analyzing window: {window_name}")
        
        # Get activations for this window
        activations = window_data["activations"]
        
        # Convert to the format expected by APA
        apa_activations = {}
        for layer_name, layer_activations in activations.items():
            # Reshape to [samples, features]
            batch_size, seq_len, hidden_dim = layer_activations.shape
            samples = batch_size * seq_len
            apa_activations[layer_name] = layer_activations.reshape(samples, hidden_dim)
        
        # Run APA analysis
        result = analyze_layer_paths(
            apa_activations,
            n_clusters=args.n_clusters,
            random_state=args.seed
        )
        
        # Calculate path metrics
        metrics = calculate_path_metrics(result)
        
        # Calculate cross-layer similarity
        similarity = calculate_cross_layer_similarity(result)
        
        # Store results
        results[window_name] = {
            "apa_result": result,
            "metrics": metrics,
            "similarity": similarity
        }
    
    # Save APA results
    logger.info("Saving APA results")
    for window_name, result_data in results.items():
        # Create output subdirectory for this window
        window_dir = output_dir / "results" / window_name
        window_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as NPY file
        metrics_file = window_dir / "metrics.npy"
        np.save(metrics_file, result_data["metrics"])
        
        # Save similarity as NPY file
        similarity_file = window_dir / "similarity.npy"
        np.save(similarity_file, result_data["similarity"])
        
        # Create and save basic visualizations
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot fragmentation scores
        axes[0].plot(result_data["metrics"]["fragmentation"])
        axes[0].set_title(f"Fragmentation Scores - {window_name}")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Fragmentation Score")
        
        # Plot similarity matrix
        im = axes[1].imshow(result_data["similarity"], cmap="viridis")
        axes[1].set_title(f"Cross-Layer Similarity - {window_name}")
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Layer")
        plt.colorbar(im, ax=axes[1])
        
        # Save figure
        plt.tight_layout()
        plt.savefig(window_dir / "visualization.png")
        plt.close()
    
    logger.info(f"Analysis complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()