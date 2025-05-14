#!/usr/bin/env python
# coding: utf-8

"""
# Concept Fragmentation in Neural Networks - Demo Workflow

This script demonstrates the complete workflow for the concept fragmentation analysis using a simple XOR dataset. It shows how to:

1. Create and train a simple neural network
2. Capture activations using the hooks module
3. Calculate fragmentation metrics
4. Visualize the results

The XOR problem is an ideal test case because it requires a non-linear decision boundary, but is simple enough to understand intuitively.
"""

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Import project modules
from concept_fragmentation.models.feedforward import FeedforwardNetwork
from concept_fragmentation.hooks.activation_hooks import capture_activations
from concept_fragmentation.metrics.cluster_entropy import cluster_entropy
from concept_fragmentation.metrics.subspace_angle import subspace_angle

def create_xor_dataset(n_repeats=25, noise_level=0.05):
    """Create an XOR dataset with optional noise."""
    # Create XOR dataset
    X = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=torch.float32)

    y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    # Expand the dataset for training
    X_expanded = X.repeat(n_repeats, 1)
    y_expanded = y.repeat(n_repeats)

    # Add small noise to make training more realistic
    if noise_level > 0:
        noise = torch.randn_like(X_expanded) * noise_level
        X_expanded = X_expanded + noise

    return X_expanded, y_expanded

def visualize_dataset(X, y):
    """Visualize the dataset."""
    plt.figure(figsize=(8, 6))
    for cls in torch.unique(y):
        mask = y == cls
        plt.scatter(X[mask, 0], X[mask, 1], s=60, alpha=0.7, label=f'Class {cls.item()}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('XOR Dataset with Noise')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_model(model, X, y, epochs=100, lr=0.01):
    """Train the model on the dataset."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).float().mean().item()
        
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy

def capture_layer_activations(model, X):
    """Capture activations at each layer."""
    with torch.no_grad(), capture_activations(model) as activations:
        _ = model(X)

    # Display available layers
    print(f"Available layers: {list(activations.keys())}")

    # Display activation shapes
    for layer_name, activation in activations.items():
        print(f"{layer_name}: {activation.shape}")
        
    return activations

def calculate_metrics(activations, labels, layer_names):
    """Calculate cluster entropy and subspace angle metrics for each layer."""
    # Calculate cluster entropy for each layer
    entropy_results = {}

    for layer_name in layer_names:
        layer_activations = activations[layer_name]
        ce = cluster_entropy(
            layer_activations, 
            labels, 
            n_clusters=2,
            return_clusters=True
        )
        entropy_results[layer_name] = ce
        
        print(f"\n{layer_name} - Cluster Entropy:")
        print(f"  Class 0: {ce['per_class'][0]:.4f}")
        print(f"  Class 1: {ce['per_class'][1]:.4f}")
        print(f"  Mean: {ce['mean']:.4f}")
        print(f"  Max: {ce['max']:.4f}")

    # Calculate subspace angle for each layer
    angle_results = {}

    for layer_name in layer_names:
        layer_activations = activations[layer_name]
        sa = subspace_angle(
            layer_activations, 
            labels,
            variance_threshold=0.9
        )
        angle_results[layer_name] = sa
        
        print(f"\n{layer_name} - Subspace Angle:")
        if (0, 1) in sa['pairwise']:
            pair_results = sa['pairwise'][(0, 1)]
            print(f"  Mean: {pair_results['mean']:.4f} degrees")
            print(f"  Min: {pair_results['min']:.4f} degrees")
            print(f"  Max: {pair_results['max']:.4f} degrees")
            if not np.isnan(pair_results['ci_95'][0]):
                print(f"  95% CI: ({pair_results['ci_95'][0]:.4f}, {pair_results['ci_95'][1]:.4f}) degrees")
        else:
            print("  No valid pairwise results")
            
    return entropy_results, angle_results

def plot_activations(layer_name, activations, labels, cluster_assignments=None):
    """Plot activations and clusters for a layer."""
    act = activations[layer_name].numpy()
    labels_np = labels.numpy()
    
    # Apply PCA if dimensions > 2
    if act.shape[1] > 2:
        pca = PCA(n_components=2)
        act_2d = pca.fit_transform(act)
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    else:
        act_2d = act
    
    plt.figure(figsize=(12, 5))
    
    # Plot activations by class
    plt.subplot(1, 2, 1)
    for cls in np.unique(labels_np):
        mask = labels_np == cls
        plt.scatter(act_2d[mask, 0], act_2d[mask, 1], s=80, alpha=0.7, label=f'Class {cls}')
    plt.title(f'{layer_name} Activations by Class')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    
    # Plot activations by cluster (if available)
    if cluster_assignments is not None:
        plt.subplot(1, 2, 2)
        
        # Class 0 clusters
        class0_mask = labels_np == 0
        clusters0 = cluster_assignments[0]
        for cluster in np.unique(clusters0):
            # Find samples in this class AND cluster
            indices = np.where(class0_mask)[0]
            cluster_mask = np.zeros_like(class0_mask)
            cluster_mask[indices[clusters0 == cluster]] = True
            
            plt.scatter(act_2d[cluster_mask, 0], act_2d[cluster_mask, 1], 
                        s=80, alpha=0.7, marker='o',
                        label=f'Class 0, Cluster {cluster}')
        
        # Class 1 clusters
        class1_mask = labels_np == 1
        clusters1 = cluster_assignments[1]
        for cluster in np.unique(clusters1):
            # Find samples in this class AND cluster
            indices = np.where(class1_mask)[0]
            cluster_mask = np.zeros_like(class1_mask)
            cluster_mask[indices[clusters1 == cluster]] = True
            
            plt.scatter(act_2d[cluster_mask, 0], act_2d[cluster_mask, 1], 
                        s=80, alpha=0.7, marker='s',
                        label=f'Class 1, Cluster {cluster}')
        
        plt.title(f'{layer_name} Activations by Cluster')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_metric_summary(layers, entropy_results, angle_results):
    """Plot summary metrics across layers."""
    entropy_means = [entropy_results[layer]['mean'] for layer in layers]
    entropy_class0 = [entropy_results[layer]['per_class'][0] for layer in layers]
    entropy_class1 = [entropy_results[layer]['per_class'][1] for layer in layers]

    angle_means = []
    for layer in layers:
        if (0, 1) in angle_results[layer]['pairwise'] and not np.isnan(angle_results[layer]['pairwise'][(0, 1)]['mean']):
            angle_means.append(angle_results[layer]['pairwise'][(0, 1)]['mean'])
        else:
            angle_means.append(np.nan)

    plt.figure(figsize=(14, 6))

    # Plot entropy
    plt.subplot(1, 2, 1)
    plt.plot(layers, entropy_means, 'o-', label='Mean Entropy', linewidth=2, markersize=10)
    plt.plot(layers, entropy_class0, 's--', label='Class 0 Entropy', linewidth=2, markersize=8)
    plt.plot(layers, entropy_class1, '^--', label='Class 1 Entropy', linewidth=2, markersize=8)
    plt.xlabel('Layer')
    plt.ylabel('Cluster Entropy')
    plt.title('Cluster Entropy by Layer')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    # Plot angle
    plt.subplot(1, 2, 2)
    plt.plot(layers, angle_means, 'o-', linewidth=2, markersize=10)
    plt.xlabel('Layer')
    plt.ylabel('Principal Angle (degrees)')
    plt.title('Subspace Angle by Layer')
    plt.ylim(0, 90)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    """Run the complete workflow."""
    print("Concept Fragmentation in Neural Networks - Demo Workflow")
    print("======================================================")
    
    # 1. Create dataset
    print("\n1. Creating XOR dataset...")
    X, y = create_xor_dataset(n_repeats=25, noise_level=0.05)
    visualize_dataset(X, y)
    
    # 2. Create and train model
    print("\n2. Creating and training neural network...")
    model = FeedforwardNetwork(
        input_dim=2,
        output_dim=2,  # Binary classification
        hidden_layer_sizes=[4, 4, 2]  # Small network for XOR
    )
    print(model)
    
    accuracy = train_model(model, X, y, epochs=100, lr=0.01)
    
    # 3. Capture activations
    print("\n3. Capturing layer activations...")
    activations = capture_layer_activations(model, X)
    
    # 4. Calculate metrics
    print("\n4. Calculating fragmentation metrics...")
    layer_names = ['layer1', 'layer2', 'layer3']
    entropy_results, angle_results = calculate_metrics(activations, y, layer_names)
    
    # 5. Visualize results
    print("\n5. Visualizing results...")
    for layer_name in layer_names:
        if 'cluster_assignments' in entropy_results[layer_name]:
            cluster_assignments = entropy_results[layer_name]['cluster_assignments']
        else:
            cluster_assignments = None
            
        plot_activations(layer_name, activations, y, cluster_assignments)
    
    # Plot summary metrics
    plot_metric_summary(layer_names, entropy_results, angle_results)
    
    print("\nConcept Fragmentation Analysis Complete")
    print("========================================")
    print("Key observations:")
    print("- Entropy generally decreases in later layers, indicating more cohesive activations")
    print("- Principal angles between class subspaces tend to increase in later layers")
    print("- Visualizations show how the network learns to separate the classes across layers")

if __name__ == "__main__":
    main() 