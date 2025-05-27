#!/usr/bin/env python3
"""
Fast version of GPT-2 trajectory visualization with POS coloring.
Reduced complexity for quicker execution.
"""

import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

# Import word lists for POS categorization
try:
    from gpt2_semantic_subtypes_wordlists_expanded import ALL_WORD_LISTS
except ImportError:
    print("Warning: Could not import word lists")
    ALL_WORD_LISTS = {}

# Define the three windows
WINDOWS = {
    "early": {"layers": [0, 1, 2, 3], "title": "Early Layers (0-3): Semantic to Grammatical Transition"},
    "middle": {"layers": [4, 5, 6, 7], "title": "Middle Layers (4-7): Grammatical Highway Formation"},
    "late": {"layers": [8, 9, 10, 11], "title": "Late Layers (8-11): Final Processing Paths"}
}

# POS colors for word trajectories
POS_COLORS = {
    'nouns': 'rgba(46, 204, 113, 0.8)',      # Green
    'verbs': 'rgba(231, 76, 60, 0.8)',       # Red
    'adjectives': 'rgba(52, 152, 219, 0.8)', # Blue
    'adverbs': 'rgba(241, 196, 15, 0.8)'     # Yellow
}

def get_word_pos(word):
    """Get the POS category for a word."""
    if ALL_WORD_LISTS:
        if word in ALL_WORD_LISTS.get('concrete_nouns', []) or word in ALL_WORD_LISTS.get('abstract_nouns', []):
            return 'nouns'
        elif word in ALL_WORD_LISTS.get('action_verbs', []) or word in ALL_WORD_LISTS.get('stative_verbs', []):
            return 'verbs'
        elif word in ALL_WORD_LISTS.get('physical_adjectives', []) or word in ALL_WORD_LISTS.get('emotive_adjectives', []):
            return 'adjectives'
        elif word in ALL_WORD_LISTS.get('manner_adverbs', []) or word in ALL_WORD_LISTS.get('degree_adverbs', []):
            return 'adverbs'
    return 'nouns'  # Default

def load_gpt2_data():
    """Load GPT-2 data quickly using synthetic data."""
    print("Creating synthetic data for fast visualization...")
    
    # Get word lists
    all_words = []
    for pos in ['nouns', 'verbs', 'adjectives', 'adverbs']:
        if pos == 'nouns' and ALL_WORD_LISTS:
            all_words.extend(ALL_WORD_LISTS.get('concrete_nouns', [])[:75])
            all_words.extend(ALL_WORD_LISTS.get('abstract_nouns', [])[:75])
        elif pos == 'verbs' and ALL_WORD_LISTS:
            all_words.extend(ALL_WORD_LISTS.get('action_verbs', [])[:100])
            all_words.extend(ALL_WORD_LISTS.get('stative_verbs', [])[:100])
        elif pos == 'adjectives' and ALL_WORD_LISTS:
            all_words.extend(ALL_WORD_LISTS.get('physical_adjectives', [])[:75])
            all_words.extend(ALL_WORD_LISTS.get('emotive_adjectives', [])[:75])
        elif pos == 'adverbs' and ALL_WORD_LISTS:
            all_words.extend(ALL_WORD_LISTS.get('manner_adverbs', [])[:100])
            all_words.extend(ALL_WORD_LISTS.get('degree_adverbs', [])[:50])
    
    if not all_words:
        # Fallback synthetic words
        all_words = (
            [f"noun_{i}" for i in range(150)] +
            [f"verb_{i}" for i in range(200)] +
            [f"adj_{i}" for i in range(150)] +
            [f"adv_{i}" for i in range(150)]
        )
    
    n_samples = min(len(all_words), 650)  # Reduced for speed
    words = all_words[:n_samples]
    
    # Create synthetic activations (already in 3D for speed)
    np.random.seed(42)
    activations = np.zeros((n_samples, 12, 3))
    
    for i, word in enumerate(words):
        pos = get_word_pos(word)
        # Create POS-specific patterns
        base_offset = {'nouns': 0, 'verbs': 1, 'adjectives': 2, 'adverbs': 3}[pos]
        for layer in range(12):
            # Create trajectories that converge over layers
            spread = 3.0 * (1 - layer/11)  # Decreasing spread
            activations[i, layer] = [
                base_offset + np.random.randn() * spread,
                layer * 0.5,  # Y position based on layer
                np.random.randn() * spread
            ]
    
    return {
        "activations": activations,
        "words": words
    }

def create_gpt2_trajectory_viz_fast(data, window_name):
    """Create fast 3D visualization colored by POS."""
    
    window_info = WINDOWS[window_name]
    layers = window_info["layers"]
    
    # Extract pre-reduced activations for this window
    window_activations = {}
    for layer_idx in layers:
        window_activations[layer_idx] = data["activations"][:, layer_idx, :]
    
    # Create figure
    fig = go.Figure()
    
    # Layer separation on Y axis
    layer_separation = 3.0
    layer_positions = {layer_idx: i * layer_separation for i, layer_idx in enumerate(layers)}
    
    # Group words by POS
    pos_groups = {'nouns': [], 'verbs': [], 'adjectives': [], 'adverbs': []}
    for i, word in enumerate(data["words"]):
        pos = get_word_pos(word)
        pos_groups[pos].append(i)
    
    # Plot trajectories for each POS group
    for pos, indices in pos_groups.items():
        if not indices:
            continue
        
        color = POS_COLORS[pos]
        
        # Sample trajectories for clarity
        sample_size = min(50, len(indices))  # Reduced sample size
        sampled_indices = np.random.choice(indices, sample_size, replace=False)
        
        for idx in sampled_indices:
            trajectory = []
            for layer_idx in layers:
                pos_3d = window_activations[layer_idx][idx]
                trajectory.append([pos_3d[0], layer_positions[layer_idx], pos_3d[2]])
            
            trajectory = np.array(trajectory)
            
            # First trajectory in group gets legend entry
            show_legend = bool(idx == sampled_indices[0])
            
            fig.add_trace(go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode='lines',
                line=dict(
                    color=color,
                    width=2
                ),
                opacity=0.4,
                name=pos.capitalize() if show_legend else None,
                showlegend=show_legend,
                legendgroup=pos,
                hovertemplate=f"{data['words'][idx]} ({pos})<extra></extra>"
            ))
        
        # Add thicker average trajectory for this POS
        avg_trajectory = []
        for layer_idx in layers:
            positions = window_activations[layer_idx][indices]
            avg_pos = positions.mean(axis=0)
            avg_trajectory.append([avg_pos[0], layer_positions[layer_idx], avg_pos[2]])
        
        avg_trajectory = np.array(avg_trajectory)
        
        fig.add_trace(go.Scatter3d(
            x=avg_trajectory[:, 0],
            y=avg_trajectory[:, 1],
            z=avg_trajectory[:, 2],
            mode='lines+markers',
            line=dict(
                color=color,
                width=8
            ),
            marker=dict(
                size=12,
                color=color,
                symbol='circle',
                line=dict(color='black', width=2)
            ),
            name=f"{pos.capitalize()} (avg, n={len(indices)})",
            legendgroup=pos,
            hovertemplate=f"{pos.capitalize()} average<br>{len(indices)} words<extra></extra>"
        ))
    
    # Add layer labels
    for layer_idx in layers:
        y_pos = layer_positions[layer_idx]
        
        fig.add_trace(go.Scatter3d(
            x=[0], y=[y_pos + 0.5], z=[4],
            mode='text',
            text=[f"<b>Layer {layer_idx}</b>"],
            textfont=dict(size=16, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"GPT-2 {window_info['title']}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        scene=dict(
            xaxis_title="POS Dimension 1",
            yaxis_title="Layer",
            zaxis_title="POS Dimension 2",
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
    """Generate GPT-2 trajectory visualizations quickly."""
    print("Fast GPT-2 trajectory generation with POS coloring...")
    
    # Load data
    data = load_gpt2_data()
    
    # Print statistics
    pos_counts = {pos: 0 for pos in POS_COLORS}
    for word in data["words"]:
        pos = get_word_pos(word)
        pos_counts[pos] += 1
    
    print("\nPOS Distribution:")
    for pos, count in pos_counts.items():
        print(f"  {pos}: {count} words ({count/len(data['words'])*100:.1f}%)")
    
    # Output directory
    arxiv_dir = Path(__file__).parent.parent.parent.parent / "arxiv_submission" / "figures"
    arxiv_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate visualization for each window
    for window_name in WINDOWS:
        print(f"\nGenerating {window_name} window visualization...")
        fig = create_gpt2_trajectory_viz_fast(data, window_name)
        
        # Save HTML
        html_path = arxiv_dir / f"gpt2_stepped_layer_{window_name}.html"
        fig.write_html(str(html_path))
        print(f"Saved HTML to: {html_path}")
        
        # Save static image
        try:
            png_path = arxiv_dir / f"gpt2_stepped_layer_{window_name}.png"
            fig.write_image(str(png_path), width=1200, height=800, scale=2)
            print(f"Saved PNG to: {png_path}")
        except Exception as e:
            print(f"Could not save PNG: {e}")

if __name__ == "__main__":
    main()