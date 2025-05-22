#!/usr/bin/env python
"""
Command-line tool for analyzing transformer attention patterns and metrics.

This script provides a command-line interface to compute various metrics on
transformer model activations and attention patterns, generating comprehensive
analysis reports.
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analyze_transformer')

# Add parent directory to path to import modules
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import our metrics
from concept_fragmentation.metrics.transformer_metrics import (
    TransformerMetricsCalculator,
    AttentionMetricsResult
)


def load_data(file_path: str) -> Dict[str, Any]:
    """
    Load activations and attention data from a file.
    
    Args:
        file_path: Path to the data file (pickle, numpy, or npz)
        
    Returns:
        Dictionary with loaded data
    """
    file_path = Path(file_path)
    file_ext = file_path.suffix.lower()
    
    try:
        if file_ext == '.pkl' or file_ext == '.pickle':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"Loaded data from pickle file: {file_path}")
                return data
        elif file_ext == '.npy':
            data = np.load(file_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == np.dtype('O'):
                # Handle object arrays (likely containing a dictionary)
                if len(data.shape) == 0:  # This is a 0-d array containing a dict
                    return data.item()
            
            # If it's just an array, wrap it in a dictionary with a default name
            logger.info(f"Loaded data from numpy file: {file_path}")
            return {"data": data}
        elif file_ext == '.npz':
            npz_data = np.load(file_path, allow_pickle=True)
            data = {}
            for key in npz_data.files:
                data[key] = npz_data[key]
            logger.info(f"Loaded data from npz file: {file_path}")
            return data
        elif file_ext == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded data from JSON file: {file_path}")
                return data
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def extract_attention_data(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extract attention data from the loaded data.
    
    Args:
        data: Data dictionary
        
    Returns:
        Dictionary mapping layer names to attention arrays
    """
    attention_data = {}
    
    # Look for attention data under common keys
    attention_keys = ['attention', 'attention_probs', 'attentions', 'self_attention']
    
    # Check if any attention keys are in the data
    found_attention = False
    for key in attention_keys:
        if key in data:
            attention_data = data[key]
            found_attention = True
            logger.info(f"Found attention data under key: {key}")
            break
    
    # If not found directly, look for a dictionary structure
    if not found_attention:
        for key, value in data.items():
            if isinstance(value, dict) and any(attn_key in value for attn_key in attention_keys):
                # Found a nested attention dictionary
                for attn_key in attention_keys:
                    if attn_key in value:
                        attention_data[key] = value[attn_key]
                        found_attention = True
                        logger.info(f"Found attention data for layer {key} under key: {attn_key}")
            elif isinstance(value, np.ndarray) and len(value.shape) in (3, 4):
                # Looks like an attention array (heads, seq, seq) or (batch, heads, seq, seq)
                attention_data[key] = value
                found_attention = True
                logger.info(f"Found attention array for key: {key} with shape {value.shape}")
    
    if not found_attention:
        logger.warning("No attention data found in the provided file.")
        
    return attention_data


def extract_activation_data(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extract activation data from the loaded data.
    
    Args:
        data: Data dictionary
        
    Returns:
        Dictionary mapping layer names to activation arrays
    """
    activation_data = {}
    
    # Look for activation data under common keys
    activation_keys = ['activations', 'hidden_states', 'outputs', 'representations']
    
    # Check if any activation keys are in the data
    found_activations = False
    for key in activation_keys:
        if key in data:
            activation_data = data[key]
            found_activations = True
            logger.info(f"Found activation data under key: {key}")
            break
    
    # If not found directly, look for a dictionary structure
    if not found_activations:
        for key, value in data.items():
            if isinstance(value, dict) and any(act_key in value for act_key in activation_keys):
                # Found a nested activation dictionary
                for act_key in activation_keys:
                    if act_key in value:
                        activation_data[key] = value[act_key]
                        found_activations = True
                        logger.info(f"Found activation data for layer {key} under key: {act_key}")
            elif isinstance(value, np.ndarray) and len(value.shape) >= 2:
                # Looks like an activation array
                activation_data[key] = value
                found_activations = True
                logger.info(f"Found activation array for key: {key} with shape {value.shape}")
    
    if not found_activations:
        logger.warning("No activation data found in the provided file.")
        
    return activation_data


def save_metrics_report(
    results: Dict[str, Any],
    output_path: str,
    plot_figures: bool = True
) -> None:
    """
    Save metrics results to a report file and generate visualizations.
    
    Args:
        results: Dictionary of analysis results
        output_path: Path to save the report
        plot_figures: Whether to generate and save figures
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Save results as JSON
    with open(output_path.with_suffix('.json'), 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        serializable_results = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    elif isinstance(v, np.number):
                        serializable_results[key][k] = v.item()
                    else:
                        serializable_results[key][k] = v
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.number):
                serializable_results[key] = value.item()
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved results to JSON: {output_path.with_suffix('.json')}")
    
    # Generate and save figures if requested
    if plot_figures:
        figure_output = output_path.parent / "figures"
        os.makedirs(figure_output, exist_ok=True)
        
        # Plot attention entropy by layer
        if 'attention_metrics' in results and 'layer_entropy' in results['attention_metrics']:
            try:
                layer_entropy = results['attention_metrics']['layer_entropy']
                layers = list(layer_entropy.keys())
                entropy_values = [layer_entropy[layer] for layer in layers]
                
                plt.figure(figsize=(10, 6))
                plt.bar(range(len(layers)), entropy_values)
                plt.xticks(range(len(layers)), layers, rotation=45)
                plt.xlabel('Layer')
                plt.ylabel('Entropy')
                plt.title('Attention Entropy by Layer')
                plt.tight_layout()
                plt.savefig(figure_output / "attention_entropy_by_layer.png", dpi=300)
                plt.close()
                
                logger.info(f"Saved entropy figure to {figure_output}/attention_entropy_by_layer.png")
            except Exception as e:
                logger.warning(f"Error generating entropy figure: {e}")
        
        # Plot attention sparsity by layer
        if 'attention_metrics' in results and 'layer_sparsity' in results['attention_metrics']:
            try:
                layer_sparsity = results['attention_metrics']['layer_sparsity']
                layers = list(layer_sparsity.keys())
                sparsity_values = [layer_sparsity[layer] for layer in layers]
                
                plt.figure(figsize=(10, 6))
                plt.bar(range(len(layers)), sparsity_values)
                plt.xticks(range(len(layers)), layers, rotation=45)
                plt.xlabel('Layer')
                plt.ylabel('Sparsity')
                plt.title('Attention Sparsity by Layer')
                plt.tight_layout()
                plt.savefig(figure_output / "attention_sparsity_by_layer.png", dpi=300)
                plt.close()
                
                logger.info(f"Saved sparsity figure to {figure_output}/attention_sparsity_by_layer.png")
            except Exception as e:
                logger.warning(f"Error generating sparsity figure: {e}")
        
        # Plot head importance heatmap
        if 'attention_metrics' in results and 'head_importance' in results['attention_metrics']:
            try:
                head_importance = results['attention_metrics']['head_importance']
                
                # Extract layers and heads
                heads = set()
                layers = set()
                for (layer, head) in head_importance.keys():
                    layers.add(layer)
                    heads.add(head)
                
                layers = sorted(list(layers))
                heads = sorted(list(heads))
                
                # Create importance matrix
                importance_matrix = np.zeros((len(layers), len(heads)))
                
                for i, layer in enumerate(layers):
                    for j, head in enumerate(heads):
                        key = (layer, head)
                        if key in head_importance:
                            importance_matrix[i, j] = head_importance[key]
                
                # Plot heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(importance_matrix, cmap='viridis', 
                            xticklabels=heads, yticklabels=layers)
                plt.xlabel('Head')
                plt.ylabel('Layer')
                plt.title('Attention Head Importance')
                plt.tight_layout()
                plt.savefig(figure_output / "head_importance_heatmap.png", dpi=300)
                plt.close()
                
                logger.info(f"Saved head importance heatmap to {figure_output}/head_importance_heatmap.png")
            except Exception as e:
                logger.warning(f"Error generating head importance heatmap: {e}")
        
        # Plot cross-layer attention consistency if available
        if 'cross_attention' in results and 'layer_consistency' in results['cross_attention']:
            try:
                layer_consistency = results['cross_attention']['layer_consistency']
                
                # Extract all layers
                layers = set()
                for layer_pair in layer_consistency.keys():
                    layers.add(layer_pair[0])
                    layers.add(layer_pair[1])
                
                layers = sorted(list(layers))
                
                # Create consistency matrix
                consistency_matrix = np.zeros((len(layers), len(layers)))
                
                # Set diagonal to 1.0 (self-consistency)
                np.fill_diagonal(consistency_matrix, 1.0)
                
                # Fill in consistency values
                for i, layer1 in enumerate(layers):
                    for j, layer2 in enumerate(layers):
                        if i == j:
                            continue  # Skip diagonal
                        
                        key = (layer1, layer2)
                        if key in layer_consistency:
                            consistency_matrix[i, j] = layer_consistency[key]
                
                # Plot heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(consistency_matrix, cmap='coolwarm', 
                            xticklabels=layers, yticklabels=layers, vmin=-1, vmax=1)
                plt.xlabel('Layer')
                plt.ylabel('Layer')
                plt.title('Cross-Layer Attention Consistency')
                plt.tight_layout()
                plt.savefig(figure_output / "cross_layer_consistency.png", dpi=300)
                plt.close()
                
                logger.info(f"Saved cross-layer consistency heatmap to {figure_output}/cross_layer_consistency.png")
            except Exception as e:
                logger.warning(f"Error generating cross-layer consistency heatmap: {e}")


def generate_summary_report(results: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of the analysis results.
    
    Args:
        results: Dictionary of analysis results
        
    Returns:
        Summary string
    """
    summary_lines = ["# Transformer Analysis Summary", ""]
    
    # Add attention metrics summary
    if 'attention_metrics' in results:
        attn_metrics = results['attention_metrics']
        
        summary_lines.append("## Attention Metrics")
        summary_lines.append("")
        
        if 'entropy' in attn_metrics:
            summary_lines.append(f"- Overall Entropy: {attn_metrics['entropy']:.4f}")
        
        if 'sparsity' in attn_metrics:
            summary_lines.append(f"- Overall Sparsity: {attn_metrics['sparsity']:.4f}")
        
        if 'layers_analyzed' in attn_metrics:
            summary_lines.append(f"- Layers Analyzed: {len(attn_metrics['layers_analyzed'])}")
        
        if 'layer_entropy' in attn_metrics:
            # Find layers with min and max entropy
            layer_entropies = attn_metrics['layer_entropy']
            min_entropy_layer = min(layer_entropies.items(), key=lambda x: x[1])
            max_entropy_layer = max(layer_entropies.items(), key=lambda x: x[1])
            
            summary_lines.append(f"- Layer with Min Entropy: {min_entropy_layer[0]} ({min_entropy_layer[1]:.4f})")
            summary_lines.append(f"- Layer with Max Entropy: {max_entropy_layer[0]} ({max_entropy_layer[1]:.4f})")
        
        summary_lines.append("")
    
    # Add activation statistics summary
    if 'activation_stats' in results:
        act_stats = results['activation_stats']
        
        summary_lines.append("## Activation Statistics")
        summary_lines.append("")
        
        # Calculate overall statistics across layers
        mean_values = []
        sparsity_values = []
        l2_norm_values = []
        
        for layer, stats in act_stats.items():
            if 'mean' in stats:
                mean_values.append(stats['mean'])
            if 'sparsity' in stats:
                sparsity_values.append(stats['sparsity'])
            if 'l2_norm' in stats:
                l2_norm_values.append(stats['l2_norm'])
        
        if mean_values:
            summary_lines.append(f"- Average Activation Mean: {np.mean(mean_values):.4f}")
        
        if sparsity_values:
            summary_lines.append(f"- Average Activation Sparsity: {np.mean(sparsity_values):.4f}")
        
        if l2_norm_values:
            summary_lines.append(f"- Average L2 Norm: {np.mean(l2_norm_values):.4f}")
        
        summary_lines.append(f"- Layers Analyzed: {len(act_stats)}")
        summary_lines.append("")
    
    # Add cross-attention consistency summary
    if 'cross_attention' in results:
        cross_attn = results['cross_attention']
        
        summary_lines.append("## Cross-Layer Attention Consistency")
        summary_lines.append("")
        
        if 'overall_consistency' in cross_attn:
            summary_lines.append(f"- Overall Consistency: {cross_attn['overall_consistency']:.4f}")
        
        if 'min_consistency' in cross_attn:
            summary_lines.append(f"- Min Consistency: {cross_attn['min_consistency']:.4f}")
        
        if 'max_consistency' in cross_attn:
            summary_lines.append(f"- Max Consistency: {cross_attn['max_consistency']:.4f}")
        
        if 'layer_consistency' in cross_attn:
            layer_consistency = cross_attn['layer_consistency']
            
            if layer_consistency:
                # Find most and least consistent layer pairs
                most_consistent = max(layer_consistency.items(), key=lambda x: x[1])
                least_consistent = min(layer_consistency.items(), key=lambda x: x[1])
                
                summary_lines.append(f"- Most Consistent Layer Pair: {most_consistent[0][0]}-{most_consistent[0][1]} ({most_consistent[1]:.4f})")
                summary_lines.append(f"- Least Consistent Layer Pair: {least_consistent[0][0]}-{least_consistent[0][1]} ({least_consistent[1]:.4f})")
        
        if 'focused_tokens' in cross_attn and 'focused_token_strings' in cross_attn:
            token_strings = cross_attn['focused_token_strings']
            summary_lines.append(f"- Focused Tokens: {', '.join(token_strings)}")
        
        summary_lines.append("")
    
    # Add activation sensitivity summary
    if 'activation_sensitivity' in results:
        sensitivity = results['activation_sensitivity']
        
        summary_lines.append("## Activation Sensitivity")
        summary_lines.append("")
        
        # Calculate average sensitivity across layers
        avg_sensitivities = {}
        
        for layer, metrics in sensitivity.items():
            for metric_type, value in metrics.items():
                if metric_type not in avg_sensitivities:
                    avg_sensitivities[metric_type] = []
                
                avg_sensitivities[metric_type].append(value)
        
        for metric_type, values in avg_sensitivities.items():
            summary_lines.append(f"- Average {metric_type.capitalize()} Sensitivity: {np.mean(values):.4f}")
        
        summary_lines.append(f"- Layers Analyzed: {len(sensitivity)}")
        summary_lines.append("")
    
    return "\n".join(summary_lines)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Analyze transformer model attention patterns and metrics"
    )
    
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the data file (.pkl, .npy, .npz, or .json)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to save the analysis results"
    )
    parser.add_argument(
        "--token-info", "-t",
        help="Optional path to a file with token information"
    )
    parser.add_argument(
        "--focus-tokens", "-f", nargs="+",
        help="Optional list of tokens to focus the analysis on"
    )
    parser.add_argument(
        "--plot-figures", action="store_true", default=True,
        help="Generate and save visualization figures"
    )
    parser.add_argument(
        "--no-figures", dest="plot_figures", action="store_false",
        help="Disable figure generation"
    )
    parser.add_argument(
        "--metrics", "-m", nargs="+", 
        default=["attention", "activations", "cross_attention"],
        choices=["attention", "activations", "cross_attention", "sensitivity", "all"],
        help="Which metrics to compute"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load data
        logger.info(f"Loading data from {args.input}")
        data = load_data(args.input)
        
        # Extract attention data
        attention_data = extract_attention_data(data)
        if not attention_data:
            logger.error("No attention data found. Cannot proceed with analysis.")
            return 1
        
        # Extract activation data
        activation_data = extract_activation_data(data)
        
        # Load token info if provided
        token_info = None
        if args.token_info:
            try:
                token_info = load_data(args.token_info)
                logger.info(f"Loaded token info from {args.token_info}")
            except Exception as e:
                logger.warning(f"Error loading token info: {e}")
        
        # Create metrics calculator
        calculator = TransformerMetricsCalculator(use_cache=True)
        
        # Initialize results dictionary
        results = {}
        
        # Compute requested metrics
        metrics_to_compute = args.metrics
        if "all" in metrics_to_compute:
            metrics_to_compute = ["attention", "activations", "cross_attention", "sensitivity"]
        
        # Compute attention metrics
        if "attention" in metrics_to_compute:
            logger.info("Computing attention metrics...")
            attention_metrics = calculator.compute_attention_metrics(
                attention_data=attention_data,
                outputs=activation_data,
                token_info=token_info
            )
            
            # Convert attention metrics to dictionary
            results["attention_metrics"] = {
                "entropy": attention_metrics.entropy,
                "sparsity": attention_metrics.sparsity,
                "layer_entropy": attention_metrics.layer_entropy,
                "layer_sparsity": attention_metrics.layer_sparsity,
                "head_entropy": attention_metrics.head_entropy,
                "head_sparsity": attention_metrics.head_sparsity,
                "head_importance": attention_metrics.head_importance,
                "layers_analyzed": attention_metrics.layers_analyzed,
                "heads_analyzed": attention_metrics.heads_analyzed
            }
        
        # Compute activation statistics
        if "activations" in metrics_to_compute and activation_data:
            logger.info("Computing activation statistics...")
            activation_stats = calculator.compute_activation_statistics(
                activations=activation_data,
                per_token=True
            )
            
            results["activation_stats"] = activation_stats
        
        # Compute cross-attention consistency
        if "cross_attention" in metrics_to_compute:
            logger.info("Computing cross-attention consistency...")
            
            # Extract token strings from token info if available
            input_tokens = None
            if token_info and "token_strings" in token_info:
                input_tokens = token_info["token_strings"]
            
            cross_attention = calculator.compute_cross_attention_consistency(
                attention_data=attention_data,
                input_tokens=input_tokens,
                focus_tokens=args.focus_tokens
            )
            
            results["cross_attention"] = cross_attention
        
        # Compute activation sensitivity if possible
        if "sensitivity" in metrics_to_compute and activation_data:
            # Check if we have perturbed activations
            perturbed_key = None
            for key in ["perturbed_activations", "perturbed", "corrupted"]:
                if key in data:
                    perturbed_key = key
                    break
            
            if perturbed_key:
                logger.info("Computing activation sensitivity...")
                perturbed_activations = data[perturbed_key]
                
                sensitivity = calculator.compute_activation_sensitivity(
                    base_activations=activation_data,
                    perturbed_activations=perturbed_activations,
                    metric="cosine"
                )
                
                results["activation_sensitivity"] = sensitivity
        
        # Generate summary report
        summary = generate_summary_report(results)
        
        # Save results
        save_metrics_report(
            results=results,
            output_path=args.output,
            plot_figures=args.plot_figures
        )
        
        # Save summary report
        with open(Path(args.output).with_suffix('.md'), 'w') as f:
            f.write(summary)
        
        logger.info(f"Saved summary report to {Path(args.output).with_suffix('.md')}")
        
        # Print summary to console
        print("\n" + summary)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())