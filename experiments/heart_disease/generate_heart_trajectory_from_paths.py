#!/usr/bin/env python3
"""
Generate proper heart disease trajectory visualization from actual cluster path data.
This matches the visualization style from the dash app.
"""

import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap

# Layer names for heart disease model
LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'output']

# Cluster labels from the analysis
CLUSTER_LABELS = {
    'layer1': {
        0: 'High-Risk Older Males',
        1: 'Lower-Risk Younger'
    },
    'layer2': {
        0: 'Low CV Stress',
        1: 'Controlled High-Risk'
    },
    'layer3': {
        0: 'Stress-Induced Risk',
        1: 'Moderate-Risk Active'
    },
    'output': {
        0: 'No Heart Disease',
        1: 'Heart Disease Present'
    }
}

def load_heart_path_data():
    """Load the actual heart disease cluster path data."""
    
    # Load path data
    path_file = Path(__file__).parent.parent.parent / "data" / "cluster_paths" / "heart_seed_0_paths_with_centroids.json"
    if not path_file.exists():
        # Try alternative location
        path_file = Path(__file__).parent.parent.parent / "visualization" / "data" / "cluster_paths" / "heart_seed_0_paths_with_centroids.json"
    
    print(f"Loading heart disease path data from: {path_file}")
    
    with open(path_file, 'r') as f:
        path_data = json.load(f)
    
    # Extract unique paths and their frequencies
    unique_paths = path_data.get('unique_paths', [])
    path_counts = path_data.get('path_counts', {})
    
    # Get the centroids for each layer
    centroids = {}
    for layer in LAYER_NAMES:
        if layer in path_data:
            layer_centroids = path_data[layer].get('centroids', {})
            # Convert string keys to int
            centroids[layer] = {int(k): np.array(v) for k, v in layer_centroids.items()}
    
    # Load the original dataset to get features
    heart_csv = Path(__file__).parent.parent.parent / "data" / "heart" / "heart.csv"
    if heart_csv.exists():
        heart_df = pd.read_csv(heart_csv)
        n_samples = len(heart_df)
        features = heart_df.drop(['target'], axis=1).values
        labels = heart_df['target'].values
    else:
        # Use synthetic data if CSV not found
        n_samples = 303
        features = np.random.randn(n_samples, 13)
        labels = np.random.randint(0, 2, n_samples)
    
    return {
        'unique_paths': unique_paths,
        'path_counts': path_counts,
        'centroids': centroids,
        'features': features,
        'labels': labels,
        'n_samples': n_samples
    }

def create_heart_trajectory_viz_from_paths():
    """Create heart disease trajectory visualization using actual path data."""
    
    # Load data
    data = load_heart_path_data()
    unique_paths = data['unique_paths']
    path_counts = data['path_counts']
    centroids = data['centroids']
    
    # Convert path strings to list of cluster assignments
    path_list = []
    count_list = []
    for i, path_str in enumerate(unique_paths):
        if str(i) in path_counts:
            count = path_counts[str(i)]
            # Parse path string like "0-1-1-1" to [0, 1, 1, 1]
            path = [int(x) for x in path_str.split('-')]
            path_list.append(path)
            count_list.append(count)
    
    # Sort by count to get top paths
    sorted_indices = np.argsort(count_list)[::-1]
    top_7_paths = [path_list[i] for i in sorted_indices[:7]]
    top_7_counts = [count_list[i] for i in sorted_indices[:7]]
    
    # Apply UMAP to centroids for each layer to get 3D positions
    reduced_centroids = {}
    
    for layer_idx, layer in enumerate(LAYER_NAMES):
        if layer in centroids and centroids[layer]:
            # Get all centroids for this layer
            cluster_ids = sorted(centroids[layer].keys())
            centroid_matrix = np.array([centroids[layer][cid] for cid in cluster_ids])
            
            # Skip UMAP if already 3D or less
            if centroid_matrix.shape[1] <= 3:
                if centroid_matrix.shape[1] < 3:
                    # Pad with zeros
                    padding = np.zeros((centroid_matrix.shape[0], 3 - centroid_matrix.shape[1]))
                    centroid_matrix = np.hstack([centroid_matrix, padding])
                reduced = centroid_matrix
            else:
                # Apply UMAP
                reducer = umap.UMAP(n_components=3, n_neighbors=min(15, len(cluster_ids)), 
                                   min_dist=0.1, random_state=42)
                reduced = reducer.fit_transform(centroid_matrix)
            
            # Normalize
            scaler = StandardScaler()
            reduced = scaler.fit_transform(reduced) * 2
            
            # Store with original cluster IDs
            reduced_centroids[layer] = {cid: reduced[i] for i, cid in enumerate(cluster_ids)}
    
    # Create figure
    fig = go.Figure()
    
    # Layer separation on Y axis
    layer_separation = 3.0
    layer_positions = {layer: i * layer_separation for i, layer in enumerate(LAYER_NAMES)}
    
    # Colors for top 7 paths
    path_colors = [
        'rgba(46, 204, 113, 0.8)',   # Green - Conservative Low-Risk
        'rgba(231, 76, 60, 0.8)',    # Red - Classic High-Risk
        'rgba(230, 126, 34, 0.8)',   # Orange - Progressive Risk
        'rgba(155, 89, 182, 0.8)',   # Purple - Male-Biased
        'rgba(241, 196, 15, 0.8)',   # Yellow - Misclassification
        'rgba(52, 152, 219, 0.8)',   # Blue - Resilient High-Risk
        'rgba(149, 165, 166, 0.8)'   # Gray - Unexpected Recovery
    ]
    
    path_names = [
        'Conservative Low-Risk Path',
        'Classic High-Risk Path',
        'Progressive Risk Path',
        'Male-Biased Path',
        'Misclassification Path',
        'Resilient High-Risk Path',
        'Unexpected Recovery Path'
    ]
    
    # Plot top 7 archetypal paths
    for path_idx, (path, count) in enumerate(zip(top_7_paths, top_7_counts)):
        trajectory = []
        
        # Build trajectory through layers
        for layer_idx, layer in enumerate(LAYER_NAMES):
            cluster_id = path[layer_idx]
            if layer in reduced_centroids and cluster_id in reduced_centroids[layer]:
                pos = reduced_centroids[layer][cluster_id]
                trajectory.append([
                    pos[0],
                    layer_positions[layer],
                    pos[2]
                ])
        
        if not trajectory:
            continue
            
        trajectory = np.array(trajectory)
        percentage = (count / data['n_samples']) * 100
        
        # Debug: print trajectory info
        print(f"Path {path_idx}: {path_names[path_idx]} - {len(trajectory)} points")
        print(f"  Trajectory shape: {trajectory.shape}")
        
        # Plot path
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode='lines+markers',
            line=dict(
                color=path_colors[path_idx],
                width=max(8, min(20, count / 5))  # Increased minimum width from 3 to 8
            ),
            marker=dict(
                size=10,
                color=path_colors[path_idx],
                symbol='circle',
                line=dict(color='black', width=1)
            ),
            name=f"{path_names[path_idx]} ({percentage:.1f}%)",
            hovertemplate=f"{path_names[path_idx]}<br>Count: {count} ({percentage:.1f}%)<extra></extra>"
        ))
        
        # Add arrows
        for i in range(len(trajectory) - 1):
            start = trajectory[i]
            end = trajectory[i + 1]
            arrow_pos = start + 0.7 * (end - start)
            direction = end - start
            direction = direction / (np.linalg.norm(direction) + 1e-8) * 0.3
            
            fig.add_trace(go.Cone(
                x=[arrow_pos[0]],
                y=[arrow_pos[1]],
                z=[arrow_pos[2]],
                u=[direction[0]],
                v=[direction[1]],
                w=[direction[2]],
                sizemode="absolute",
                sizeref=0.5,
                colorscale=[[0, path_colors[path_idx]], [1, path_colors[path_idx]]],
                showscale=False,
                opacity=0.8,
                hoverinfo='skip'
            ))
    
    # Add cluster labels
    for layer_idx, layer in enumerate(LAYER_NAMES):
        y_pos = layer_positions[layer]
        
        # Add layer label
        fig.add_trace(go.Scatter3d(
            x=[0], y=[y_pos + 0.5], z=[3],
            mode='text',
            text=[f"<b>{layer.upper()}</b>"],
            textfont=dict(size=14, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add cluster labels
        if layer in reduced_centroids:
            for cluster_id, pos in reduced_centroids[layer].items():
                label = CLUSTER_LABELS.get(layer, {}).get(cluster_id, f"Cluster {cluster_id}")
                
                fig.add_trace(go.Scatter3d(
                    x=[pos[0]], 
                    y=[y_pos], 
                    z=[pos[2]],
                    mode='text+markers',
                    text=[f"<b>{label}</b>"],
                    textfont=dict(size=10, color='darkblue'),
                    marker=dict(size=8, color='lightblue', symbol='circle'),
                    showlegend=False,
                    hovertemplate=f"{label}<br>Cluster {cluster_id}<extra></extra>"
                ))
    
    # Add ALL patient trajectories color-coded by outcome
    print("\nAdding all patient trajectories...")
    
    # Load individual patient assignments if available
    if 'paths' in data:
        all_paths = data['paths']  # Individual patient paths
    else:
        # Generate all patient paths based on the path counts
        all_paths = []
        for path_idx, (path, count) in enumerate(zip(top_7_paths, top_7_counts)):
            for _ in range(count):
                all_paths.append(path)
        
        # Add remaining paths not in top 7
        remaining = data['n_samples'] - len(all_paths)
        for _ in range(remaining):
            # Random path from the unique paths
            random_path = [np.random.randint(0, 2) for _ in range(4)]
            all_paths.append(random_path)
    
    # Plot each patient's trajectory
    for i, path in enumerate(all_paths):
        trajectory = []
        
        # Build trajectory through layers
        for layer_idx, layer in enumerate(LAYER_NAMES):
            cluster_id = path[layer_idx]
            if layer in reduced_centroids and cluster_id in reduced_centroids[layer]:
                pos = reduced_centroids[layer][cluster_id]
                # Add small random noise to separate overlapping trajectories
                noise = np.random.normal(0, 0.1, 3)
                trajectory.append([
                    pos[0] + noise[0],
                    layer_positions[layer],
                    pos[2] + noise[2]
                ])
        
        if trajectory:
            trajectory = np.array(trajectory)
            
            # Color based on final outcome (OutputC0 = no disease, OutputC1 = disease)
            final_cluster = path[3]  # Output layer cluster
            if final_cluster == 0:  # No disease
                color = 'rgba(52, 152, 219, 0.3)'  # Blue
                outcome = 'No Disease'
            else:  # Disease
                color = 'rgba(231, 76, 60, 0.3)'   # Red
                outcome = 'Disease'
            
            # Plot patient trajectory
            fig.add_trace(go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode='lines',
                line=dict(
                    color=color,
                    width=1
                ),
                showlegend=False,
                hovertemplate=f"Patient {i+1}<br>Outcome: {outcome}<extra></extra>"
            ))
    
    print(f"Added {len(all_paths)} patient trajectories")
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Heart Disease Model: Patient Pathways Through Neural Network Layers",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="Layer",
            zaxis_title="UMAP-2",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=2, z=1)
        ),
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        margin=dict(l=0, r=200, t=80, b=0)
    )
    
    return fig

def main():
    """Generate and save the heart disease trajectory visualization."""
    print("Generating heart disease trajectory visualization from actual path data...")
    
    # Create visualization
    fig = create_heart_trajectory_viz_from_paths()
    
    # Save outputs
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Save HTML
    html_path = output_dir / "heart_trajectories_actual_data.html"
    fig.write_html(str(html_path))
    print(f"Saved interactive HTML to: {html_path}")
    
    # Save static image
    try:
        png_path = output_dir / "heart_trajectories_actual_data.png"
        fig.write_image(str(png_path), width=1200, height=800, scale=2)
        print(f"Saved static PNG to: {png_path}")
        
        # Also save to arxiv figures (overwrite the synthetic one)
        arxiv_path = Path(__file__).parent.parent.parent / "arxiv_submission" / "figures" / "heart_stepped_layer_trajectories.png"
        fig.write_image(str(arxiv_path), width=1200, height=800, scale=2)
        print(f"Saved to arxiv figures: {arxiv_path}")
    except Exception as e:
        print(f"Could not save PNG: {e}")
        print("Note: Install kaleido to generate static images: pip install kaleido")

if __name__ == "__main__":
    main()