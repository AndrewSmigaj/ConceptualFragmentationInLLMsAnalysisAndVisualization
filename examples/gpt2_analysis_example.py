"""
Example of using the GPT-2 adapter for activation analysis.

This script demonstrates how to:
1. Load a GPT-2 model
2. Create an adapter for it
3. Extract activations from different layers
4. Analyze attention patterns
5. Visualize results
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import logging
import argparse

# Ensure the repository root is in the path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import our model adapter
from concept_fragmentation.models.transformer_adapter import (
    GPT2Adapter,
    get_transformer_adapter
)

# Import activation collector for streaming collection
from concept_fragmentation.activation import (
    ActivationCollector,
    CollectionConfig,
    ActivationFormat
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_gpt2_model(model_size: str = "small"):
    """
    Load a GPT-2 model and tokenizer.
    
    Args:
        model_size: Size of GPT-2 model ('small', 'medium', 'large', 'xl')
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        # Import required libraries
        from transformers import GPT2Model, GPT2Tokenizer
        
        # Map model size to Hugging Face model name
        model_map = {
            "small": "openai-community/gpt2",
            "medium": "openai-community/gpt2-medium",
            "large": "openai-community/gpt2-large",
            "xl": "openai-community/gpt2-xl"
        }
        
        model_name = model_map.get(model_size.lower(), "openai-community/gpt2")
        
        logger.info(f"Loading GPT-2 model: {model_name}")
        model = GPT2Model.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        return model, tokenizer
    
    except ImportError:
        logger.error("transformers library not installed. Install with: pip install transformers")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading GPT-2 model: {e}")
        sys.exit(1)


def analyze_text(model, tokenizer, text: str, adapter: Optional[GPT2Adapter] = None):
    """
    Analyze a text sample with GPT-2.
    
    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        text: Text to analyze
        adapter: Optional pre-created adapter
        
    Returns:
        Dictionary of analysis results
    """
    logger.info(f"Analyzing text: {text}")
    
    # Create adapter if not provided
    if adapter is None:
        adapter = GPT2Adapter(model, tokenizer=tokenizer)
    
    # Tokenize input
    tokens = tokenizer.encode(text, return_tensors="pt")
    token_strings = [tokenizer.decode([token_id]) for token_id in tokens[0]]
    
    logger.info(f"Tokenized into {len(token_strings)} tokens: {token_strings}")
    
    # Extract activations
    layer_outputs = adapter.get_layer_outputs(text)
    attention_patterns = adapter.get_attention_patterns(text)
    
    logger.info(f"Extracted activations from {len(layer_outputs)} layers")
    logger.info(f"Extracted attention patterns from {len(attention_patterns)} attention components")
    
    # Collect results
    results = {
        "text": text,
        "tokens": tokens,
        "token_strings": token_strings,
        "layer_outputs": layer_outputs,
        "attention_patterns": attention_patterns
    }
    
    return results


def plot_attention_patterns(results: Dict[str, Any], output_dir: str = "output", max_layers: int = 3):
    """
    Plot attention patterns from the analysis results.
    
    Args:
        results: Analysis results
        output_dir: Directory to save plots
        max_layers: Maximum number of layers to plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get token strings for axis labels
    token_strings = results["token_strings"]
    
    # Get attention patterns
    attention_patterns = results["attention_patterns"]
    
    # Sort attention patterns by layer number
    sorted_patterns = sorted(
        attention_patterns.items(),
        key=lambda x: int(x[0].split('_')[1]) if 'transformer_layer_' in x[0] else 0
    )
    
    # Limit to max_layers
    patterns_to_plot = sorted_patterns[:max_layers]
    
    # Create plots
    for layer_name, attention in patterns_to_plot:
        # Extract layer number for the filename
        if 'transformer_layer_' in layer_name:
            layer_num = layer_name.split('_')[1]
        else:
            layer_num = layer_name
        
        # Get the attention pattern for the first item in the batch
        # Reduce heads dimension by averaging across heads
        attn = attention[0].mean(dim=0).cpu().detach().numpy()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.imshow(attn, cmap='viridis')
        
        # Add labels
        plt.xlabel('Token (target)')
        plt.ylabel('Token (source)')
        plt.title(f'Attention Pattern - {layer_name}')
        
        # Add token labels if not too many
        if len(token_strings) <= 20:
            plt.xticks(np.arange(len(token_strings)), token_strings, rotation=90)
            plt.yticks(np.arange(len(token_strings)), token_strings)
        
        # Add colorbar
        plt.colorbar()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'attention_layer_{layer_num}.png'))
        plt.close()
    
    logger.info(f"Saved {len(patterns_to_plot)} attention pattern plots to {output_dir}")


def plot_layer_activations(results: Dict[str, Any], output_dir: str = "output", max_layers: int = 3):
    """
    Plot activation patterns from different layers.
    
    Args:
        results: Analysis results
        output_dir: Directory to save plots
        max_layers: Maximum number of layers to plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get token strings for axis labels
    token_strings = results["token_strings"]
    
    # Get layer outputs
    layer_outputs = results["layer_outputs"]
    
    # Filter to only include transformer layer outputs
    transformer_outputs = {
        name: tensor for name, tensor in layer_outputs.items()
        if 'transformer_layer_' in name and 'output' in name
    }
    
    # Sort by layer number
    sorted_outputs = sorted(
        transformer_outputs.items(),
        key=lambda x: int(x[0].split('_')[1]) if 'transformer_layer_' in x[0] else 0
    )
    
    # Limit to max_layers
    outputs_to_plot = sorted_outputs[:max_layers]
    
    # Create plots
    for layer_name, activations in outputs_to_plot:
        # Extract layer number for the filename
        layer_num = layer_name.split('_')[1]
        
        # Get activations for the first item in the batch
        acts = activations[0].cpu().detach().numpy()
        
        # Compute the mean activation across the hidden dimension
        mean_acts = np.mean(acts, axis=1)
        
        # Create figure for mean activations
        plt.figure(figsize=(12, 4))
        plt.plot(mean_acts)
        plt.xlabel('Token Position')
        plt.ylabel('Mean Activation')
        plt.title(f'Mean Activation by Token - {layer_name}')
        
        # Add token labels if not too many
        if len(token_strings) <= 20:
            plt.xticks(np.arange(len(token_strings)), token_strings, rotation=90)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'mean_activation_layer_{layer_num}.png'))
        plt.close()
        
        # Create heatmap of activations
        # Take first 100 features if there are many
        feature_limit = min(100, acts.shape[1])
        
        plt.figure(figsize=(12, 8))
        plt.imshow(acts[:, :feature_limit].T, aspect='auto', cmap='viridis')
        plt.xlabel('Token Position')
        plt.ylabel('Feature Index')
        plt.title(f'Activation Features - {layer_name} (first {feature_limit} features)')
        
        # Add token labels if not too many
        if len(token_strings) <= 20:
            plt.xticks(np.arange(len(token_strings)), token_strings, rotation=90)
        
        # Add colorbar
        plt.colorbar()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'activation_heatmap_layer_{layer_num}.png'))
        plt.close()
    
    logger.info(f"Saved {len(outputs_to_plot)} activation plots to {output_dir}")


def analyze_with_streaming(model, tokenizer, text: str, output_dir: str = "output"):
    """
    Analyze text using streaming activation collection.
    
    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        text: Text to analyze
        output_dir: Directory to save results
        
    Returns:
        Path to the saved activations
    """
    logger.info(f"Analyzing text with streaming activation collection: {text}")
    
    # Create adapter
    adapter = GPT2Adapter(model, tokenizer=tokenizer)
    
    # Create activation collector with streaming config
    config = CollectionConfig(
        format=ActivationFormat.NUMPY,
        stream_to_disk=True
    )
    collector = ActivationCollector(config)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    
    # Register model with collector
    collector.register_model(
        model=adapter,
        model_id="gpt2",
        activation_points=[
            f"transformer_layer_{i}_output" for i in range(12)
        ] + [
            f"transformer_layer_{i}_attention_attn_probs" for i in range(12)
        ]
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect and store activations
    output_path = os.path.join(output_dir, "gpt2_activations.pkl")
    collector.collect_and_store(
        model=adapter,
        inputs=inputs,
        output_path=output_path,
        streaming=True
    )
    
    logger.info(f"Saved streaming activations to {output_path}")
    
    return output_path


def main(args):
    """Main function."""
    # Load GPT-2 model
    model, tokenizer = load_gpt2_model(model_size=args.model_size)
    
    # Create adapter
    adapter = GPT2Adapter(model, tokenizer=tokenizer)
    
    # Analyze different text samples
    for i, text in enumerate(args.text):
        # Create output directory for this sample
        sample_dir = os.path.join(args.output_dir, f"sample_{i}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Analyze text
        results = analyze_text(model, tokenizer, text, adapter=adapter)
        
        # Plot results
        plot_attention_patterns(results, output_dir=sample_dir, max_layers=args.max_layers)
        plot_layer_activations(results, output_dir=sample_dir, max_layers=args.max_layers)
    
    # Demonstrate streaming collection if requested
    if args.streaming:
        # Use the first text sample for streaming demo
        text = args.text[0] if args.text else "Hello, world!"
        
        # Analyze with streaming
        streaming_path = analyze_with_streaming(
            model, tokenizer, text, 
            output_dir=os.path.join(args.output_dir, "streaming")
        )
        
        logger.info(f"Streaming analysis saved to: {streaming_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 Activation Analysis Example")
    
    parser.add_argument("--text", type=str, nargs="+", default=["Hello, world!", 
                                                               "The quick brown fox jumps over the lazy dog.",
                                                               "In machine learning, transformers are a type of neural network architecture."],
                        help="Text samples to analyze")
    
    parser.add_argument("--model-size", type=str, default="small", 
                        choices=["small", "medium", "large", "xl"],
                        help="GPT-2 model size")
    
    parser.add_argument("--output-dir", type=str, default="gpt2_analysis_output",
                        help="Directory to save outputs")
    
    parser.add_argument("--max-layers", type=int, default=3,
                        help="Maximum number of layers to plot")
    
    parser.add_argument("--streaming", action="store_true",
                        help="Demonstrate streaming activation collection")
    
    args = parser.parse_args()
    
    main(args)