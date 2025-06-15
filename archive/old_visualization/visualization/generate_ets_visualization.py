"""
Generate visualizations for Explainable Threshold Similarity (ETS) clustering boundaries.

This script creates visualizations that show the threshold boundaries for key dimensions
in ETS clustering, allowing for direct interpretation of cluster membership.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any, Union
import argparse

# Define constants
CMAP = "viridis"


def generate_synthetic_ets_data(n_neurons: int = 5, 
                               n_clusters: int = 3,
                               n_samples: int = 100,
                               noise_level: float = 0.1) -> Dict[str, Any]:
    """
    Generate synthetic ETS threshold data.
    
    Args:
        n_neurons: Number of neurons (dimensions)
        n_clusters: Number of clusters
        n_samples: Number of samples
        noise_level: Level of noise to add
        
    Returns:
        Dictionary with synthetic ETS data
    """
    # Generate thresholds for each cluster
    cluster_thresholds = []
    
    for cluster_idx in range(n_clusters):
        # Generate cluster center in n-dimensional space
        cluster_center = np.random.uniform(0.2, 0.8, n_neurons)
        
        # Generate lower and upper thresholds around center
        lower_thresholds = cluster_center - np.random.uniform(0.05, 0.2, n_neurons)
        upper_thresholds = cluster_center + np.random.uniform(0.05, 0.2, n_neurons)
        
        # Ensure thresholds are in [0, 1] range
        lower_thresholds = np.clip(lower_thresholds, 0, 1)
        upper_thresholds = np.clip(upper_thresholds, 0, 1)
        
        # Ensure lower < upper
        for i in range(n_neurons):
            if lower_thresholds[i] >= upper_thresholds[i]:
                mid = (lower_thresholds[i] + upper_thresholds[i]) / 2
                lower_thresholds[i] = mid - 0.05
                upper_thresholds[i] = mid + 0.05
        
        cluster_thresholds.append({
            'cluster_id': cluster_idx,
            'lower_thresholds': lower_thresholds,
            'upper_thresholds': upper_thresholds
        })
    
    # Generate sample activations
    activations = []
    cluster_assignments = []
    
    for sample_idx in range(n_samples):
        # Choose a random cluster to sample from
        cluster_idx = np.random.randint(0, n_clusters)
        thresholds = cluster_thresholds[cluster_idx]
        
        # Generate activation within threshold bounds + noise
        lower = thresholds['lower_thresholds']
        upper = thresholds['upper_thresholds']
        
        # Generate a point within bounds
        activation = lower + np.random.uniform(0, 1, n_neurons) * (upper - lower)
        
        # Add noise that might push some dimensions outside bounds
        noise = np.random.normal(0, noise_level, n_neurons)
        activation = activation + noise
        activation = np.clip(activation, 0, 1)  # Keep in [0, 1] range
        
        activations.append(activation)
        cluster_assignments.append(cluster_idx)
    
    # Create activations array
    activations_array = np.array(activations)
    
    # Create importance scores for neurons
    importance_scores = np.random.uniform(0.1, 1.0, n_neurons)
    importance_scores = importance_scores / np.sum(importance_scores)  # Normalize
    
    return {
        'cluster_thresholds': cluster_thresholds,
        'activations': activations_array,
        'cluster_assignments': np.array(cluster_assignments),
        'neuron_importance': importance_scores
    }


def load_or_generate_ets_data(data_dir: str, dataset: str = "titanic", layer: str = "layer2") -> Dict[str, Any]:
    """
    Load ETS data if available, or generate synthetic data.
    
    Args:
        data_dir: Directory containing data files
        dataset: Dataset name
        layer: Layer name
        
    Returns:
        Dictionary with ETS data
    """
    # Check if ETS data exists
    ets_file = os.path.join(data_dir, f"{dataset}_{layer}_ets_thresholds.json")
    
    if os.path.exists(ets_file):
        with open(ets_file, 'r') as f:
            ets_data = json.load(f)
        print(f"Loaded ETS data from {ets_file}")
        return ets_data
    
    # Generate synthetic data
    print(f"No ETS data found. Generating synthetic data.")
    return generate_synthetic_ets_data()


def generate_threshold_boundary_viz(ets_data: Dict[str, Any], 
                                   output_dir: str,
                                   layer_name: str = "layer2",
                                   static: bool = False):
    """
    Generate visualization of ETS threshold boundaries.
    
    Args:
        ets_data: Dictionary with ETS data
        output_dir: Directory to save output files
        layer_name: Name of the layer being visualized
        static: Whether to generate static matplotlib viz instead of interactive Plotly
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    cluster_thresholds = ets_data['cluster_thresholds']
    activations = ets_data['activations']
    cluster_assignments = ets_data['cluster_assignments']
    neuron_importance = ets_data.get('neuron_importance', 
                                     np.ones(activations.shape[1]) / activations.shape[1])
    
    # Sort neurons by importance
    sorted_indices = np.argsort(-neuron_importance)
    top_neurons = sorted_indices[:min(10, len(sorted_indices))]  # Top 10 most important neurons
    
    if static:
        # Create static matplotlib visualization
        n_clusters = len(cluster_thresholds)
        n_neurons = len(top_neurons)
        
        # Create figure with one row per cluster
        fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 3 * n_clusters), sharex=True)
        if n_clusters == 1:
            axes = [axes]
        
        # Set up x-axis with neuron labels
        neuron_labels = [f"Neuron {i}\n({neuron_importance[i]:.3f})" for i in top_neurons]
        
        # Plot each cluster's thresholds
        for cluster_idx, thresholds in enumerate(cluster_thresholds):
            ax = axes[cluster_idx]
            
            # Get thresholds for top neurons
            lower = thresholds['lower_thresholds'][top_neurons]
            upper = thresholds['upper_thresholds'][top_neurons]
            
            # Plot ranges as bars
            for i, neuron_idx in enumerate(top_neurons):
                # Calculate range width
                width = upper[i] - lower[i]
                
                # Plot bar
                ax.barh(i, width, left=lower[i], height=0.7,
                        color=plt.cm.tab10(cluster_idx % 10), alpha=0.7,
                        label=f"Cluster {cluster_idx}" if i == 0 else "")
                
                # Add text with threshold values
                ax.text(lower[i] - 0.05, i, f"{lower[i]:.2f}", 
                        ha='right', va='center', fontsize=8)
                ax.text(upper[i] + 0.05, i, f"{upper[i]:.2f}", 
                        ha='left', va='center', fontsize=8)
            
            # Set labels
            ax.set_title(f"Cluster {cluster_idx} Threshold Boundaries")
            ax.set_yticks(range(n_neurons))
            ax.set_yticklabels(neuron_labels)
            ax.set_xlim(0, 1)
            ax.grid(True, axis='x', alpha=0.3)
            
            # Add legend
            ax.legend()
        
        # Set overall labels
        fig.suptitle(f"ETS Threshold Boundaries for {layer_name}", fontsize=16)
        fig.text(0.5, 0.01, "Activation Value", ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.05)
        
        # Save figure
        output_file = os.path.join(output_dir, f"ets_thresholds_{layer_name}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"ETS threshold visualization saved to {output_file}")
        
    else:
        # Create interactive Plotly visualization
        fig = go.Figure()
        
        # Create one subplot per cluster
        n_clusters = len(cluster_thresholds)
        
        # Create figure with one row per cluster
        fig = make_subplots(rows=n_clusters, cols=1, 
                           shared_xaxes=True,
                           subplot_titles=[f"Cluster {i} Threshold Boundaries" 
                                          for i in range(n_clusters)])
        
        # Set up x-axis with neuron labels
        neuron_labels = [f"Neuron {i}" for i in top_neurons]
        
        # Plot each cluster's thresholds
        for cluster_idx, thresholds in enumerate(cluster_thresholds):
            # Get thresholds for top neurons
            lower = thresholds['lower_thresholds'][top_neurons]
            upper = thresholds['upper_thresholds'][top_neurons]
            
            # Get samples in this cluster
            cluster_mask = cluster_assignments == cluster_idx
            cluster_samples = activations[cluster_mask]
            
            # Add threshold ranges
            for i, neuron_idx in enumerate(top_neurons):
                # Plot threshold range
                fig.add_trace(go.Bar(
                    x=[upper[i] - lower[i]],
                    y=[neuron_labels[i]],
                    orientation='h',
                    base=lower[i],
                    marker=dict(
                        color=px.colors.qualitative.Plotly[cluster_idx % len(px.colors.qualitative.Plotly)],
                        line=dict(width=1, color='black')
                    ),
                    name=f"Cluster {cluster_idx}",
                    showlegend=(i == 0),  # Show legend only once per cluster
                    legendgroup=f"cluster_{cluster_idx}",
                    text=[f"{lower[i]:.2f} - {upper[i]:.2f}"],
                    hovertemplate="Neuron: %{y}<br>Range: %{text}<br>Importance: {:.3f}".format(
                        neuron_importance[neuron_idx]
                    )
                ), row=cluster_idx+1, col=1)
                
                # Add jittered points for actual activations
                if cluster_samples.shape[0] > 0:
                    # Extract activations for this neuron
                    neuron_activations = cluster_samples[:, neuron_idx]
                    
                    # Add jitter for y-axis
                    jitter = np.random.normal(0, 0.1, len(neuron_activations))
                    
                    fig.add_trace(go.Scatter(
                        x=neuron_activations,
                        y=[neuron_labels[i]] * len(neuron_activations) + jitter,
                        mode='markers',
                        marker=dict(
                            color=px.colors.qualitative.Plotly[cluster_idx % len(px.colors.qualitative.Plotly)],
                            size=3,
                            opacity=0.5
                        ),
                        name=f"Cluster {cluster_idx} Samples",
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=cluster_idx+1, col=1)
            
            # Add text annotation for importance
            for i, neuron_idx in enumerate(top_neurons):
                fig.add_annotation(
                    x=0.02,
                    y=neuron_labels[i],
                    text=f"Imp: {neuron_importance[neuron_idx]:.3f}",
                    showarrow=False,
                    xanchor='left',
                    font=dict(size=8),
                    row=cluster_idx+1,
                    col=1
                )
        
        # Update layout
        fig.update_layout(
            title_text=f"ETS Threshold Boundaries for {layer_name}",
            height=300 * n_clusters,
            width=800,
            xaxis_title="Activation Value",
            barmode='overlay',
            bargap=0.3
        )
        
        # Set x-axis range
        fig.update_xaxes(range=[0, 1])
        
        # Save figure
        output_file = os.path.join(output_dir, f"ets_thresholds_{layer_name}.html")
        fig.write_html(output_file)
        print(f"Interactive ETS threshold visualization saved to {output_file}")
        
        # Also save as static image
        static_file = os.path.join(output_dir, f"ets_thresholds_{layer_name}.png")
        fig.write_image(static_file, width=800, height=300 * n_clusters, scale=2)
        print(f"Static image saved to {static_file}")
    
    return fig


def generate_parallel_coordinates_plot(ets_data: Dict[str, Any],
                                     output_dir: str,
                                     layer_name: str = "layer2"):
    """
    Generate parallel coordinates plot for multi-dimensional view of ETS thresholds.
    
    Args:
        ets_data: Dictionary with ETS data
        output_dir: Directory to save output files
        layer_name: Name of the layer being visualized
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    cluster_thresholds = ets_data['cluster_thresholds']
    activations = ets_data['activations']
    cluster_assignments = ets_data['cluster_assignments']
    neuron_importance = ets_data.get('neuron_importance', 
                                    np.ones(activations.shape[1]) / activations.shape[1])
    
    # Sort neurons by importance
    sorted_indices = np.argsort(-neuron_importance)
    top_neurons = sorted_indices[:min(10, len(sorted_indices))]  # Top 10 most important neurons
    
    # Create dataframe for parallel coordinates
    df_list = []
    
    # Add bounds for each cluster
    for cluster_idx, thresholds in enumerate(cluster_thresholds):
        # Get thresholds for top neurons
        lower = thresholds['lower_thresholds'][top_neurons]
        upper = thresholds['upper_thresholds'][top_neurons]
        
        # Add lower bounds
        lower_row = {f"Neuron {i} ({neuron_importance[idx]:.3f})": lower[i] 
                    for i, idx in enumerate(top_neurons)}
        lower_row['Type'] = f'Cluster {cluster_idx} Lower'
        lower_row['Cluster'] = cluster_idx
        lower_row['Bound'] = 'Lower'
        df_list.append(lower_row)
        
        # Add upper bounds
        upper_row = {f"Neuron {i} ({neuron_importance[idx]:.3f})": upper[i] 
                    for i, idx in enumerate(top_neurons)}
        upper_row['Type'] = f'Cluster {cluster_idx} Upper'
        upper_row['Cluster'] = cluster_idx
        upper_row['Bound'] = 'Upper'
        df_list.append(upper_row)
    
    # Convert to dataframe
    df = pd.DataFrame(df_list)
    
    # Create parallel coordinates plot
    fig = px.parallel_coordinates(
        df,
        color="Cluster",
        labels={col: col.split(' ')[0] + ' ' + col.split(' ')[1] for col in df.columns 
               if col not in ['Type', 'Cluster', 'Bound']},
        color_continuous_scale=px.colors.diverging.Tealrose,
        title=f"ETS Threshold Boundaries for {layer_name} - Parallel Coordinates"
    )
    
    # Add shape designation (solid for upper, dashed for lower)
    fig.update_traces(
        line = dict(
            dash = ['solid' if row['Bound'] == 'Upper' else 'dash' for idx, row in df.iterrows()]
        )
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1000,
        coloraxis_colorbar=dict(
            title="Cluster"
        )
    )
    
    # Save figure
    output_file = os.path.join(output_dir, f"ets_parallel_coords_{layer_name}.html")
    fig.write_html(output_file)
    print(f"Parallel coordinates visualization saved to {output_file}")
    
    # Also save as static image
    static_file = os.path.join(output_dir, f"ets_parallel_coords_{layer_name}.png")
    fig.write_image(static_file, width=1000, height=600, scale=2)
    print(f"Static image saved to {static_file}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate ETS threshold boundary visualizations.")
    parser.add_argument("--data_dir", type=str, default="data/cluster_paths",
                      help="Directory containing ETS data")
    parser.add_argument("--output_dir", type=str, default="results/figures",
                      help="Directory to save output figures")
    parser.add_argument("--dataset", type=str, default="titanic",
                      help="Dataset name")
    parser.add_argument("--layer", type=str, default="layer2",
                      help="Layer name")
    parser.add_argument("--static", action="store_true",
                      help="Generate static matplotlib visualizations instead of interactive Plotly")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or generate ETS data
    ets_data = load_or_generate_ets_data(args.data_dir, args.dataset, args.layer)
    
    # Generate visualizations
    generate_threshold_boundary_viz(ets_data, args.output_dir, args.layer, args.static)
    
    generate_parallel_coordinates_plot(ets_data, args.output_dir, args.layer)
    
    print("ETS threshold boundary visualizations generated successfully!")


if __name__ == "__main__":
    main()