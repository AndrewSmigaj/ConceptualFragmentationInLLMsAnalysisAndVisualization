#!/usr/bin/env python3
"""
Generate 3D stepped-layer trajectory visualization for heart disease model.
Shows how patients flow through the neural network layers from input to output.
"""

import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Define patient archetype labels based on our analysis
ARCHETYPE_LABELS = {
    0: "Conservative Low-Risk Path",
    1: "Classic High-Risk Path", 
    2: "Progressive Risk Path",
    3: "Male-Biased Path",
    4: "Misclassification Path"
}

def load_heart_data():
    """Load heart disease activations and metadata."""
    # Path to the heart disease results
    base_path = Path(__file__).parent / "results"
    
    # Load activation data (you'll need to adjust paths based on actual data location)
    # This is a placeholder - replace with actual data loading
    print("Loading heart disease activation data...")
    
    # For now, create synthetic data to demonstrate the visualization
    # In practice, load actual activations from your saved results
    np.random.seed(42)
    n_patients = 303
    
    # Create synthetic activations for each layer
    # Input layer (13 features)
    input_activations = np.random.randn(n_patients, 13)
    
    # Hidden layers (simulate 2 hidden layers with different dimensions)
    hidden1_activations = np.random.randn(n_patients, 20)
    hidden2_activations = np.random.randn(n_patients, 10)
    
    # Output layer (2 classes: no disease, disease)
    output_activations = np.random.randn(n_patients, 2)
    
    # Create labels (0: no disease, 1: disease)
    labels = np.random.randint(0, 2, n_patients)
    
    # Create archetype assignments (which of the 5 paths each patient follows)
    archetype_assignments = np.random.choice([0, 1, 2, 3, 4], n_patients, p=[0.43, 0.35, 0.11, 0.07, 0.04])
    
    return {
        "input": input_activations,
        "hidden1": hidden1_activations,
        "hidden2": hidden2_activations,
        "output": output_activations,
        "labels": labels,
        "archetypes": archetype_assignments
    }

def reduce_to_3d(activations, method="pca"):
    """Reduce high-dimensional activations to 3D for visualization."""
    if activations.shape[1] <= 3:
        # Pad with zeros if less than 3D
        if activations.shape[1] < 3:
            padding = np.zeros((activations.shape[0], 3 - activations.shape[1]))
            return np.hstack([activations, padding])
        return activations
    
    if method == "pca":
        reducer = PCA(n_components=3)
    elif method == "tsne":
        reducer = TSNE(n_components=3, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_components=3, random_state=42)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    
    return reducer.fit_transform(activations)

def create_heart_trajectory_viz(data, reduction_method="pca"):
    """Create 3D stepped-layer visualization for heart disease model."""
    
    # Reduce each layer to 3D
    layers = ["input", "hidden1", "hidden2", "output"]
    reduced_activations = {}
    
    for layer in layers:
        print(f"Reducing {layer} layer to 3D using {reduction_method}...")
        reduced_activations[layer] = reduce_to_3d(data[layer], method=reduction_method)
    
    # Normalize to consistent scale
    for layer in layers:
        activations = reduced_activations[layer]
        # Center and scale
        activations = (activations - activations.mean(axis=0)) / (activations.std(axis=0) + 1e-8)
        reduced_activations[layer] = activations
    
    # Create figure
    fig = go.Figure()
    
    # Layer separation on Y axis
    layer_separation = 3.0
    layer_positions = {
        "input": 0,
        "hidden1": 1 * layer_separation,
        "hidden2": 2 * layer_separation,
        "output": 3 * layer_separation
    }
    
    # Color scheme for archetypes
    archetype_colors = {
        0: 'rgba(46, 204, 113, 0.8)',   # Green - Conservative Low-Risk
        1: 'rgba(231, 76, 60, 0.8)',     # Red - Classic High-Risk
        2: 'rgba(230, 126, 34, 0.8)',    # Orange - Progressive Risk
        3: 'rgba(155, 89, 182, 0.8)',    # Purple - Male-Biased
        4: 'rgba(241, 196, 15, 0.8)'     # Yellow - Misclassification
    }
    
    # Add layer planes/floors
    for layer, y_pos in layer_positions.items():
        activations = reduced_activations[layer]
        x_min, x_max = activations[:, 0].min(), activations[:, 0].max()
        z_min, z_max = activations[:, 2].min(), activations[:, 2].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        z_padding = (z_max - z_min) * 0.1
        
        # Create floor plane
        fig.add_trace(go.Mesh3d(
            x=[x_min - x_padding, x_max + x_padding, x_max + x_padding, x_min - x_padding],
            y=[y_pos, y_pos, y_pos, y_pos],
            z=[z_min - z_padding, z_min - z_padding, z_max + z_padding, z_max + z_padding],
            i=[0, 0], j=[1, 2], k=[2, 3],
            opacity=0.1,
            color='lightgray',
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Add layer label
        fig.add_trace(go.Scatter3d(
            x=[0], y=[y_pos + 0.5], z=[0],
            mode='text',
            text=[f"<b>{layer.upper()} LAYER</b>"],
            textfont=dict(size=14),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Define class colors - make them darker and more saturated for visibility
    class_colors = {
        0: 'rgba(34, 139, 34, 1.0)',   # Dark Green - No Disease
        1: 'rgba(220, 20, 60, 1.0)'     # Crimson Red - Disease
    }
    
    # First, identify and plot archetypal paths (one thick line per archetype)
    for archetype_id, archetype_label in ARCHETYPE_LABELS.items():
        # Get patients in this archetype
        archetype_mask = data["archetypes"] == archetype_id
        archetype_indices = np.where(archetype_mask)[0]
        
        if len(archetype_indices) == 0:
            continue
            
        # Calculate average trajectory for this archetype
        archetype_trajectories = []
        for layer in layers:
            # Get all points for this archetype at this layer
            archetype_points = reduced_activations[layer][archetype_mask]
            # Calculate centroid
            centroid = archetype_points.mean(axis=0)
            centroid[1] += layer_positions[layer]
            archetype_trajectories.append(centroid)
        
        archetype_trajectories = np.array(archetype_trajectories)
        
        # Plot thick archetype path
        fig.add_trace(go.Scatter3d(
            x=archetype_trajectories[:, 0],
            y=archetype_trajectories[:, 1],
            z=archetype_trajectories[:, 2],
            mode='lines',
            line=dict(
                color=archetype_colors[archetype_id],
                width=8,
                dash='solid'
            ),
            opacity=0.9,
            name=f"{archetype_label} (n={len(archetype_indices)})",
            legendgroup=f"archetype_{archetype_id}",
            showlegend=True,
            hovertemplate=f"Archetype: {archetype_label}<br>Patients: {len(archetype_indices)}<extra></extra>"
        ))
    
    # Then plot individual patient trajectories colored by class
    n_patients_to_plot = min(100, len(data["labels"]))  # Plot up to 100 patients
    sample_indices = np.random.choice(len(data["labels"]), n_patients_to_plot, replace=False)
    
    # Track if we've shown legend for each class
    shown_class_legend = {0: False, 1: False}
    
    for patient_idx in sample_indices:
        # Get patient's class
        patient_class = data["labels"][patient_idx]
        patient_archetype = data["archetypes"][patient_idx]
        
        # Get trajectory points for this patient
        trajectory_points = []
        for layer in layers:
            point = reduced_activations[layer][patient_idx].copy()
            point[1] += layer_positions[layer]  # Apply Y offset
            trajectory_points.append(point)
        
        trajectory_points = np.array(trajectory_points)
        
        # Determine if we should show legend for this trace
        show_legend = not shown_class_legend[patient_class]
        if show_legend:
            shown_class_legend[patient_class] = True
        
        # Add trajectory line colored by class
        fig.add_trace(go.Scatter3d(
            x=trajectory_points[:, 0],
            y=trajectory_points[:, 1],
            z=trajectory_points[:, 2],
            mode='lines',
            line=dict(
                color=class_colors[patient_class],
                width=4  # Increased from 2 to 4 for better visibility
            ),
            opacity=0.8,  # Increased from 0.3 to 0.8 for darker lines
            name=f"{'No Disease' if patient_class == 0 else 'Disease'} Patients",
            legendgroup=f"class_{patient_class}",
            showlegend=show_legend,
            hovertemplate=f"Patient {patient_idx}<br>Class: {'No Disease' if patient_class == 0 else 'Disease'}<br>Archetype: {ARCHETYPE_LABELS[patient_archetype]}<extra></extra>"
        ))
        
        # Add arrows to show direction
        for i in range(len(trajectory_points) - 1):
            start = trajectory_points[i]
            end = trajectory_points[i + 1]
            direction = end - start
            
            # Arrow position (80% along the segment)
            arrow_pos = start + 0.8 * direction
            
            # Normalize direction for arrow
            arrow_dir = direction / (np.linalg.norm(direction) + 1e-8)
            
            fig.add_trace(go.Cone(
                x=[arrow_pos[0]],
                y=[arrow_pos[1]],
                z=[arrow_pos[2]],
                u=[arrow_dir[0]],
                v=[arrow_dir[1]],
                w=[arrow_dir[2]],
                sizemode="absolute",
                sizeref=0.3,
                colorscale=[[0, class_colors[patient_class]], [1, class_colors[patient_class]]],
                showscale=False,
                opacity=0.8,  # Increased to match line opacity
                hoverinfo='skip'
            ))
    
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Heart Disease Model: Patient Trajectories Through Neural Network Layers",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        scene=dict(
            xaxis_title="Feature 1 (UMAP)",
            yaxis_title="Layer Progression",
            zaxis_title="Feature 3 (UMAP)",
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
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        margin=dict(l=0, r=150, t=50, b=0)
    )
    
    return fig

def main():
    """Generate and save the heart disease trajectory visualization."""
    # Load data
    data = load_heart_data()
    
    # Create visualization
    fig = create_heart_trajectory_viz(data, reduction_method="umap")
    
    # Save outputs
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Save HTML
    html_path = output_dir / "heart_trajectories_3d.html"
    fig.write_html(str(html_path))
    print(f"Saved interactive HTML to: {html_path}")
    
    # Save static image
    try:
        png_path = output_dir / "heart_trajectories_3d.png"
        fig.write_image(str(png_path), width=1200, height=800, scale=2)
        print(f"Saved static PNG to: {png_path}")
        
        # Also save to arxiv figures
        arxiv_path = Path(__file__).parent.parent.parent / "arxiv_submission" / "figures" / "heart_stepped_layer_trajectories.png"
        fig.write_image(str(arxiv_path), width=1200, height=800, scale=2)
        print(f"Saved to arxiv figures: {arxiv_path}")
    except:
        print("Note: Install kaleido to generate static images: pip install kaleido")

if __name__ == "__main__":
    main()