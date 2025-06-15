"""
Example demonstrating the enhanced ActivationCollector.

This example shows how to collect activations from a simple MLP model
and a pre-trained GPT-2 model, demonstrating memory-efficient streaming
collection and processing.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time

# Import the new ActivationCollector
from concept_fragmentation.activation import (
    ActivationCollector, ActivationProcessor, ActivationStorage,
    CollectionConfig, ProcessorConfig, StorageConfig,
    ActivationFormat, ProcessingOperation
)

# Uncomment to use GPT-2 (requires transformers package)
# from transformers import GPT2Model, GPT2Tokenizer


def create_simple_model():
    """Create a simple MLP model for demonstration."""
    model = nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 5)
    )
    return model


def collect_mlp_activations():
    """Demonstrate activation collection for a simple MLP model."""
    print("\n=== Collecting activations from MLP model ===")
    
    # Create model and data
    model = create_simple_model()
    data = torch.randn(100, 10)  # 100 samples
    
    # Create collector
    config = CollectionConfig(
        device='cpu',
        format=ActivationFormat.NUMPY,
        log_dimensions=True
    )
    collector = ActivationCollector(config)
    
    # Register model
    collector.register_model(model)
    
    # Collect activations
    print("\nCollecting all activations at once...")
    start_time = time.time()
    activations = collector.collect(model, data)
    print(f"Collection took {time.time() - start_time:.2f} seconds")
    
    # Print activation information
    print("\nActivation info:")
    for layer, activation in activations.items():
        print(f"  {layer}: shape={activation.shape}, dtype={activation.dtype}")
    
    # Demonstrate streaming collection
    print("\nDemonstrating streaming collection...")
    
    # Create DataLoader for batched processing
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=20, shuffle=False)
    
    # Collect activations in streaming mode
    start_time = time.time()
    batch_generator = collector.collect(model, loader, streaming=True)
    
    # Process each batch
    batch_count = 0
    total_samples = 0
    
    for batch in batch_generator:
        batch_count += 1
        batch_metadata = batch['metadata']
        batch_activations = batch['activations']
        
        # Get first layer to determine batch size
        first_layer = next(iter(batch_activations.values()))
        batch_size = first_layer.shape[0]
        total_samples += batch_size
        
        print(f"  Batch {batch_count}: {batch_size} samples, " 
              f"{len(batch_activations)} layers")
    
    print(f"Processed {batch_count} batches with {total_samples} total samples")
    print(f"Streaming took {time.time() - start_time:.2f} seconds")
    
    return model, data


def process_activations(activations):
    """Demonstrate activation processing."""
    print("\n=== Processing activations ===")
    
    # Create processor
    config = ProcessorConfig(precision='float32')
    processor = ActivationProcessor(config)
    
    # Add dimensionality reduction
    processor.dimensionality_reduction(
        method='pca',
        n_components=2,
        fit_data=activations
    )
    
    # Add normalization
    processor.normalize(method='standard')
    
    # Process the activations
    print("Processing activations...")
    start_time = time.time()
    processed = processor.process(activations)
    print(f"Processing took {time.time() - start_time:.2f} seconds")
    
    # Print processed activation information
    print("\nProcessed activation info:")
    for layer, activation in processed.items():
        print(f"  {layer}: shape={activation.shape}, dtype={activation.dtype}")
    
    # Visualize the reduced dimensions for one layer
    visualize_layer = list(processed.keys())[1]  # Skip first layer (usually input)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        processed[visualize_layer][:, 0],
        processed[visualize_layer][:, 1],
        alpha=0.7
    )
    plt.title(f"PCA of {visualize_layer} activations")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    # Save or display the plot
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name)
        print(f"\nSaved visualization to {tmp.name}")
    
    return processed


def store_and_load_activations(activations):
    """Demonstrate activation storage and loading."""
    print("\n=== Storing and loading activations ===")
    
    # Create storage
    config = StorageConfig()
    storage = ActivationStorage(config)
    
    # Store activations
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        file_path = tmp.name
    
    print(f"Storing activations to {file_path}...")
    start_time = time.time()
    stored_path = storage.save(
        activations,
        file_path,
        metadata={"description": "Example activations"}
    )
    print(f"Storage took {time.time() - start_time:.2f} seconds")
    
    # Load activations
    print("\nLoading activations...")
    start_time = time.time()
    loaded = storage.load(stored_path)
    print(f"Loading took {time.time() - start_time:.2f} seconds")
    
    # Verify loaded data
    print("\nLoaded activation info:")
    for layer, activation in loaded['activations'].items():
        original = activations[layer]
        print(f"  {layer}: shape={activation.shape}, "
              f"matches original: {np.array_equal(activation, original)}")
    
    # Load only specific layers
    first_layer = list(activations.keys())[0]
    print(f"\nLoading only {first_layer}...")
    partial = storage.load(stored_path, layers=[first_layer])
    print(f"Partial load contains {len(partial['activations'])} layers")
    
    return loaded


def collect_gpt2_activations():
    """Demonstrate activation collection for GPT-2 (if available)."""
    try:
        from transformers import GPT2Model, GPT2Tokenizer
    except ImportError:
        print("\n=== GPT-2 collection skipped (transformers package not installed) ===")
        return None, None
    
    print("\n=== Collecting activations from GPT-2 model ===")
    
    # Load pre-trained model and tokenizer
    print("Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    
    # Prepare input
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt")
    
    # Create collector with streaming support
    config = CollectionConfig(
        device='cpu',
        format=ActivationFormat.NUMPY,
        log_dimensions=True
    )
    collector = ActivationCollector(config)
    
    # Register model with specific layer patterns
    collector.register_model(
        model,
        model_id='gpt2',
        include_patterns=['h.\d+'],  # Match transformer blocks
    )
    
    # Collect activations
    print("Collecting GPT-2 activations...")
    start_time = time.time()
    with torch.no_grad():
        activations = collector.collect(model, inputs)
    print(f"Collection took {time.time() - start_time:.2f} seconds")
    
    # Print activation information
    print("\nGPT-2 activation info:")
    for layer, activation in activations.items():
        print(f"  {layer}: shape={activation.shape}, dtype={activation.dtype}")
    
    return model, activations


def main():
    """Run the activation collection examples."""
    print("=== Enhanced ActivationCollector Examples ===")
    
    # MLP example
    model, activations = collect_mlp_activations()
    
    # Process activations
    processed = process_activations(activations)
    
    # Store and load activations
    loaded = store_and_load_activations(activations)
    
    # GPT-2 example (if available)
    gpt2_model, gpt2_activations = collect_gpt2_activations()
    
    print("\n=== All examples completed successfully ===")


if __name__ == "__main__":
    main()