"""
GPT-2 Archetypal Path Analysis.

This module implements Archetypal Path Analysis for GPT-2 transformer models.
It provides functionality to:
1. Load GPT-2 models of different sizes
2. Extract activations from specified text inputs
3. Analyze activation patterns with the APA framework
4. Visualize token paths through neural network layers
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import torch
from datetime import datetime

# Import GPT-2 adapter
from concept_fragmentation.models.transformer_adapter import GPT2Adapter

# Import visualization tools
try:
    from visualization.gpt2_token_sankey import (
        extract_token_paths,
        generate_token_sankey_diagram,
        get_token_path_stats,
        create_token_path_comparison,
        create_3layer_window_sankey
    )
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    print("Warning: Visualization tools not available. Install required dependencies for visualizations.")
    VISUALIZATIONS_AVAILABLE = False

# Import APA analysis functions
from concept_fragmentation.analysis.cluster_paths import (
    compute_clusters_for_layer,
    assign_unique_cluster_ids,
    compute_cluster_paths,
    compute_path_archetypes,
    compute_fragmentation_score,
    extract_unique_centroids
)

# Import from similarity metrics
from concept_fragmentation.analysis.similarity_metrics import (
    compute_centroid_similarity,
    normalize_similarity_matrix
)

# Import from config
from concept_fragmentation.config import RESULTS_DIR, LOG_LEVEL, CACHE_DIR

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gpt2_apa")

# Default sample texts for quick testing
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
    "In the beginning God created the heavens and the earth.",
    "It was the best of times, it was the worst of times.",
    "Four score and seven years ago our fathers brought forth on this continent a new nation."
]


def setup_output_dirs(base_dir: str, model_name: str, timestamp: Optional[str] = None) -> Dict[str, str]:
    """
    Set up output directories for results.
    
    Args:
        base_dir: Base output directory
        model_name: Name of the GPT-2 model
        timestamp: Optional timestamp to use (if None, current time is used)
        
    Returns:
        Dictionary of output directory paths
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main output directory
    output_name = f"{model_name}_{timestamp}"
    output_path = os.path.join(base_dir, output_name)
    
    # Create subdirectories
    dirs = {
        "main": output_path,
        "activations": os.path.join(output_path, "activations"),
        "clusters": os.path.join(output_path, "clusters"),
        "results": os.path.join(output_path, "results"),
        "visualizations": os.path.join(output_path, "visualizations")
    }
    
    # Create directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def load_text_from_file(file_path: str) -> str:
    """
    Load text from a file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Text content
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_gpt2_activations(
    text: str,
    model_type: Union[str, GPT2ModelType],
    window_size: int = 3,
    stride: int = 1,
    device: str = "cpu",
    context_window: int = 512,
    output_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Extract GPT-2 activations for the given text.
    
    Args:
        text: Input text
        model_type: GPT-2 model type
        window_size: Size of the sliding window for analysis
        stride: Stride for the sliding window
        device: Device to use for extraction ("cpu" or "cuda")
        context_window: Maximum context window size
        output_dir: Optional directory to save activations
        
    Returns:
        Dictionary mapping window names to activation data
    """
    # Load GPT-2 model and tokenizer
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        raise ImportError("transformers library required for GPT-2 analysis. Install with: pip install transformers")
    
    # Ensure model_type is a string
    if not isinstance(model_type, str):
        model_type = str(model_type)
    
    logger.info(f"Loading GPT-2 model: {model_type}")
    
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(
        model_type,
        output_hidden_states=True,
        output_attentions=True
    )
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    
    # Move to device
    if device != 'cpu' and torch.cuda.is_available():
        model.to(device)
    elif device != 'cpu':
        logger.warning(f"Device {device} not available, using CPU")
        device = 'cpu'
    
    model.eval()
    
    # Initialize adapter
    adapter = GPT2Adapter(model, tokenizer=tokenizer)
    
    # Extract activations with sliding windows
    logger.info(f"Extracting activations with window_size={window_size}, stride={stride}")
    windows = adapter.extract_activations_for_windows(
        text,
        window_size=window_size,
        stride=stride
    )
    
    # Save activations if output_dir is provided
    if output_dir:
        # Get metadata from first window
        metadata = windows[list(windows.keys())[0]].get("metadata", {})
        
        # Save all window activations
        activation_files = {}
        for window_name, window_data in windows.items():
            window_activations = window_data["activations"]
            metadata_file = adapter.save_activations_with_metadata(
                window_activations,
                {"window_name": window_name, **metadata},
                output_dir=output_dir,
                prefix=f"gpt2_apa_{window_name}"
            )
            activation_files[window_name] = metadata_file
        logger.info(f"Saved activations to: {metadata_file}")
    
    return windows


def analyze_window_activations(
    window_data: Dict[str, Any],
    window_name: str,
    n_clusters: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Analyze window activations using Archetypal Path Analysis.
    
    Args:
        window_data: Window data from GPT2ActivationExtractor
        window_name: Name of the window for logging
        n_clusters: Maximum number of clusters to use
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with APA analysis results
    """
    logger.info(f"Analyzing window: {window_name}")
    
    # Extract activations
    activations = window_data["activations"]
    metadata = window_data["metadata"]
    
    # Reshape activations for clustering
    # We reshape to [n_samples, n_features] where each token is a sample
    layer_clusters = {}
    for layer_name, layer_activations in activations.items():
        # Reshape: [batch_size, seq_len, hidden_dim] -> [batch_size * seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = layer_activations.shape
        reshaped_activations = layer_activations.reshape(batch_size * seq_len, hidden_dim)
        
        # Filter out padding based on attention mask if available
        if "attention_mask" in metadata:
            attention_mask = metadata["attention_mask"]
            mask_flat = attention_mask.reshape(-1)
            reshaped_activations = reshaped_activations[mask_flat == 1]
            logger.info(f"  Layer {layer_name}: Filtered to {reshaped_activations.shape[0]} active tokens")
        
        # Compute clusters
        logger.info(f"  Computing clusters for layer {layer_name}")
        k, centers, labels = compute_clusters_for_layer(
            reshaped_activations,
            max_k=n_clusters,
            random_state=random_state
        )
        logger.info(f"  Found {k} clusters for layer {layer_name}")
        
        # Store results
        layer_clusters[layer_name] = {
            "k": k,
            "centers": centers,
            "labels": labels,
            "activations": reshaped_activations
        }
    
    # Assign unique IDs to clusters across layers
    logger.info("Assigning unique cluster IDs")
    layer_clusters, id_to_layer_cluster, cluster_to_unique_id = assign_unique_cluster_ids(layer_clusters)
    
    # Compute cluster paths
    logger.info("Computing cluster paths")
    unique_paths, layer_names, id_mapping, original_paths, human_readable_paths = compute_cluster_paths(layer_clusters)
    
    # Compute centroid similarity
    logger.info("Computing centroid similarity")
    similarity_matrix = compute_centroid_similarity(
        layer_clusters,
        id_to_layer_cluster,
        metric="cosine",
        min_similarity=0.0,
        same_layer=True
    )
    normalized_matrix = normalize_similarity_matrix(similarity_matrix, metric="cosine")
    
    # Compute path fragmentation scores
    logger.info("Computing fragmentation scores")
    fragmentation_scores = []
    for path in unique_paths:
        score = compute_fragmentation_score(path, normalized_matrix, id_to_layer_cluster)
        fragmentation_scores.append(score)
    
    # Extract centroids
    logger.info("Extracting centroids")
    unique_centroids = extract_unique_centroids(layer_clusters, id_to_layer_cluster)
    
    # Prepare results
    results = {
        "window_name": window_name,
        "clusters": layer_clusters,
        "id_mapping": id_to_layer_cluster,
        "unique_paths": unique_paths,
        "layer_names": layer_names,
        "original_paths": original_paths,
        "human_readable_paths": human_readable_paths,
        "similarity_matrix": normalized_matrix,
        "fragmentation_scores": fragmentation_scores
    }
    
    return results


def visualize_window_results(
    window_data: Dict[str, Any],
    analysis_results: Dict[str, Any],
    output_dir: Optional[str] = None,
    highlight_tokens: Optional[List[str]] = None,
    min_path_count: int = 1
) -> Dict[str, Any]:
    """
    Create visualizations for APA analysis results.
    
    Args:
        window_data: Window data from GPT2ActivationExtractor
        analysis_results: Results from analyze_window_activations
        output_dir: Directory to save visualizations
        highlight_tokens: List of token keys to highlight
        min_path_count: Minimum token count for paths in Sankey diagram
        
    Returns:
        Dictionary with visualization results
    """
    if not VISUALIZATIONS_AVAILABLE:
        logger.warning("Visualizations not available. Skipping visualization step.")
        return {}
    
    window_name = analysis_results["window_name"]
    logger.info(f"Creating visualizations for window: {window_name}")
    
    # Create 3-layer window Sankey diagram
    viz_results = create_3layer_window_sankey(
        window_data,
        analysis_results,
        highlight_tokens=highlight_tokens,
        min_path_count=min_path_count,
        output_dir=output_dir,
        title=f"GPT-2 Token Path Flow: {window_name}",
        save_html=True
    )
    
    return viz_results


def write_results(
    windows_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    prefix: str = "gpt2_apa_results"
) -> str:
    """
    Write analysis results to disk.
    
    Args:
        windows_results: Dictionary mapping window names to analysis results
        output_dir: Directory to save results
        prefix: Prefix for result files
        
    Returns:
        Path to the saved results file
    """
    # Create serializable results
    serializable_results = {
        "windows": {},
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_windows": len(windows_results)
        }
    }
    
    # Process each window's results
    for window_name, window_results in windows_results.items():
        # Filter out non-serializable elements (like numpy arrays)
        window_dict = {
            "window_name": window_name,
            "n_clusters_per_layer": {layer: info["k"] for layer, info in window_results["clusters"].items()},
            "layer_names": window_results["layer_names"],
            "n_paths": len(window_results["unique_paths"]),
            "fragmentation_scores": {
                "mean": float(np.mean(window_results["fragmentation_scores"])),
                "std": float(np.std(window_results["fragmentation_scores"])),
                "min": float(np.min(window_results["fragmentation_scores"])),
                "max": float(np.max(window_results["fragmentation_scores"]))
            }
        }
        
        # Add to results
        serializable_results["windows"][window_name] = window_dict
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{prefix}.json")
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved results to: {output_file}")
    return output_file


def run_gpt2_apa_pipeline(args):
    """
    Run the complete GPT-2 Archetypal Path Analysis pipeline.
    
    Args:
        args: Command-line arguments
    """
    # Set up output directories
    output_dirs = setup_output_dirs(
        args.output_dir,
        args.model.value if hasattr(args.model, 'value') else args.model,
        args.timestamp
    )
    
    # Get input text
    if args.input_file:
        logger.info(f"Loading text from file: {args.input_file}")
        text = load_text_from_file(args.input_file)
    elif args.text:
        text = args.text
    else:
        logger.info("Using sample text")
        text = SAMPLE_TEXTS[0]
    
    # Log text length
    logger.info(f"Analyzing text with {len(text)} characters")
    
    # Convert model type string to enum if needed
    model_type = args.model if isinstance(args.model, GPT2ModelType) else GPT2ModelType.from_string(args.model)
    
    # Extract activations
    windows = extract_gpt2_activations(
        text=text,
        model_type=model_type,
        window_size=args.window_size,
        stride=args.stride,
        device=args.device,
        context_window=args.context_window,
        output_dir=output_dirs["activations"]
    )
    
    # Analyze each window
    window_results = {}
    for window_name, window_data in windows.items():
        # Run analysis
        analysis_results = analyze_window_activations(
            window_data=window_data,
            window_name=window_name,
            n_clusters=args.n_clusters,
            random_state=args.seed
        )
        
        # Save analysis to results directory
        window_results_dir = os.path.join(output_dirs["results"], window_name)
        os.makedirs(window_results_dir, exist_ok=True)
        
        # Add to window results
        window_results[window_name] = analysis_results
        
        # Create visualizations if requested
        if args.visualize:
            viz_output_dir = os.path.join(output_dirs["visualizations"], window_name)
            visualize_window_results(
                window_data=window_data,
                analysis_results=analysis_results,
                output_dir=viz_output_dir,
                highlight_tokens=args.highlight_tokens,
                min_path_count=args.min_path_count
            )
    
    # Write overall results
    write_results(
        windows_results=window_results,
        output_dir=output_dirs["main"],
        prefix="gpt2_apa_results"
    )
    
    logger.info(f"GPT-2 APA analysis complete. Results in: {output_dirs['main']}")
    
    # Print summary
    print("\nSummary:")
    print(f"Model: {model_type.value}")
    print(f"Windows analyzed: {len(windows)}")
    print(f"Results directory: {output_dirs['main']}")
    if args.visualize:
        print(f"Visualizations: {output_dirs['visualizations']}")


def main():
    """Main entry point for GPT-2 Archetypal Path Analysis."""
    parser = argparse.ArgumentParser(
        description="Run Archetypal Path Analysis on GPT-2 models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_source = input_group.add_mutually_exclusive_group()
    input_source.add_argument("--input-file", type=str, help="Path to input text file")
    input_source.add_argument("--text", type=str, help="Direct text input")
    input_source.add_argument("--sample", action="store_true", help="Use sample text")
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model", 
        type=str, 
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="GPT-2 model size"
    )
    model_group.add_argument(
        "--device", 
        type=str, 
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for computation"
    )
    model_group.add_argument(
        "--context-window", 
        type=int, 
        default=512,
        help="Maximum context window size"
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--window-size", 
        type=int, 
        default=3,
        help="Size of sliding window (number of consecutive layers)"
    )
    analysis_group.add_argument(
        "--stride", 
        type=int, 
        default=1,
        help="Stride for sliding window"
    )
    analysis_group.add_argument(
        "--n-clusters", 
        type=int, 
        default=10,
        help="Maximum number of clusters to use"
    )
    analysis_group.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Visualization options
    vis_group = parser.add_argument_group("Visualization Options")
    vis_group.add_argument(
        "--visualize", 
        action="store_true", 
        help="Generate visualizations"
    )
    vis_group.add_argument(
        "--highlight-tokens", 
        type=str, 
        nargs="+",
        help="Tokens to highlight in visualizations"
    )
    vis_group.add_argument(
        "--min-path-count", 
        type=int, 
        default=1,
        help="Minimum token count for paths in Sankey diagram"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir", 
        type=str, 
        default=os.path.join(RESULTS_DIR if RESULTS_DIR else "results", "gpt2_apa"),
        help="Output directory for results"
    )
    output_group.add_argument(
        "--timestamp", 
        type=str, 
        default=None,
        help="Timestamp to use for output directory (default: current time)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert model string to GPT2ModelType
    if isinstance(args.model, str):
        args.model = GPT2ModelType.from_string(args.model)
    
    # Handle CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Using CPU instead.")
        args.device = "cpu"
    
    # Set default text if sample requested
    if args.sample:
        args.text = SAMPLE_TEXTS[0]
    
    # Ensure either input_file, text, or sample is provided
    if not args.input_file and not args.text and not args.sample:
        logger.error("No input provided. Use --input-file, --text, or --sample.")
        sys.exit(1)
    
    # Run the pipeline
    run_gpt2_apa_pipeline(args)


if __name__ == "__main__":
    main()