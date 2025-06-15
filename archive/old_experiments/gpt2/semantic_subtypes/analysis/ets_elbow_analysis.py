#!/usr/bin/env python3
"""
Elbow method analysis for ETS threshold selection using collected data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_elbow_metrics(thresholds, clusters):
    """Calculate rate of change and curvature for elbow detection."""
    # Calculate first derivative (rate of change)
    first_derivative = np.diff(clusters) / np.diff(thresholds)
    
    # Calculate second derivative (curvature)
    second_derivative = np.diff(first_derivative) / np.diff(thresholds[:-1])
    
    return first_derivative, second_derivative

def find_elbow_point(thresholds, clusters):
    """Find elbow using maximum curvature method."""
    # Normalize the data
    x = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())
    y = (clusters - clusters.min()) / (clusters.max() - clusters.min())
    
    # Calculate distances from line connecting first and last point
    # Line from (x[0], y[0]) to (x[-1], y[-1])
    distances = []
    for i in range(1, len(x)-1):
        # Distance from point to line
        num = abs((y[-1]-y[0])*x[i] - (x[-1]-x[0])*y[i] + x[-1]*y[0] - y[-1]*x[0])
        den = np.sqrt((y[-1]-y[0])**2 + (x[-1]-x[0])**2)
        distances.append(num/den)
    
    # Find maximum distance (elbow point)
    elbow_idx = np.argmax(distances) + 1
    
    return elbow_idx

def analyze_ets_elbow():
    """Analyze ETS results using elbow method."""
    # Our collected data
    data = [
        (0.95, 540),
        (0.98, 246),
        (0.985, 133),
        (0.986, 114),
        (0.987, 99),
        (0.988, 84),
        (0.989, 66),
        (0.990, 49),
        (0.991, 38),
        (0.992, 24),
        (0.995, 7),
        (0.997, 2),
        (0.998, 1),
    ]
    
    thresholds = np.array([d[0] for d in data])
    clusters = np.array([d[1] for d in data])
    
    # Find elbow point
    elbow_idx = find_elbow_point(thresholds, clusters)
    elbow_threshold = thresholds[elbow_idx]
    elbow_clusters = clusters[elbow_idx]
    
    # Calculate derivatives
    first_deriv, second_deriv = calculate_elbow_metrics(thresholds, clusters)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Clusters vs Threshold with elbow
    axes[0, 0].plot(thresholds, clusters, 'b-o', markersize=8, linewidth=2)
    axes[0, 0].plot(elbow_threshold, elbow_clusters, 'ro', markersize=12, 
                    label=f'Elbow: {elbow_threshold:.3f} ({elbow_clusters} clusters)')
    axes[0, 0].set_xlabel('Threshold Percentile')
    axes[0, 0].set_ylabel('Number of Clusters')
    axes[0, 0].set_title('ETS Clusters vs Threshold (Elbow Method)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Add annotations for key points
    for threshold, n_clusters in [(0.992, 24), (0.990, 49), (0.988, 84)]:
        idx = np.where(thresholds == threshold)[0][0]
        axes[0, 0].annotate(f'{n_clusters}', 
                           xy=(threshold, n_clusters),
                           xytext=(threshold-0.002, n_clusters+20),
                           fontsize=9,
                           ha='center')
    
    # Plot 2: Log scale
    axes[0, 1].semilogy(thresholds, clusters, 'b-o', markersize=8, linewidth=2)
    axes[0, 1].semilogy(elbow_threshold, elbow_clusters, 'ro', markersize=12)
    axes[0, 1].set_xlabel('Threshold Percentile')
    axes[0, 1].set_ylabel('Number of Clusters (log scale)')
    axes[0, 1].set_title('Log Scale View')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Rate of change
    axes[1, 0].plot(thresholds[:-1], first_deriv, 'g-o', markersize=6)
    axes[1, 0].set_xlabel('Threshold Percentile')
    axes[1, 0].set_ylabel('Rate of Change (dClusters/dThreshold)')
    axes[1, 0].set_title('First Derivative')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Distance from diagonal (elbow detection)
    x_norm = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())
    y_norm = (clusters - clusters.min()) / (clusters.max() - clusters.min())
    
    axes[1, 1].plot(thresholds, y_norm, 'b-o', markersize=6, label='Normalized clusters')
    axes[1, 1].plot([thresholds[0], thresholds[-1]], [y_norm[0], y_norm[-1]], 
                    'k--', alpha=0.5, label='Diagonal')
    axes[1, 1].plot(elbow_threshold, y_norm[elbow_idx], 'ro', markersize=12, 
                    label=f'Elbow point')
    axes[1, 1].set_xlabel('Threshold Percentile')
    axes[1, 1].set_ylabel('Normalized Clusters')
    axes[1, 1].set_title('Elbow Detection')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('ETS Elbow Analysis for Optimal Threshold Selection', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    results_dir = Path("semantic_subtypes_experiment_20250523_111112")
    plt.savefig(results_dir / 'ets_elbow_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Elbow analysis saved to: {results_dir / 'ets_elbow_analysis.png'}")
    
    # Print analysis
    print("\n" + "="*60)
    print("ETS ELBOW METHOD ANALYSIS")
    print("="*60)
    
    print(f"\nElbow point detected at:")
    print(f"  Threshold: {elbow_threshold:.3f}")
    print(f"  Clusters: {elbow_clusters}")
    
    print("\nInterpretation:")
    print("The elbow suggests the optimal balance between granularity")
    print("and interpretability. This is where the rate of cluster")
    print("reduction starts to flatten out.")
    
    print("\nKey threshold ranges:")
    print("  - 0.992: 24 clusters (similar to # unique K-means paths)")
    print("  - 0.990: 49 clusters (good middle ground)")
    print("  - 0.988: 84 clusters (captures more distinctions)")
    print(f"  - {elbow_threshold:.3f}: {elbow_clusters} clusters (elbow point)")
    
    # Additional analysis
    print("\nCluster reduction analysis:")
    for i in range(len(thresholds)-1):
        reduction = clusters[i] - clusters[i+1]
        pct_reduction = reduction / clusters[i] * 100
        if pct_reduction > 20:  # Significant drops
            print(f"  {thresholds[i]:.3f} → {thresholds[i+1]:.3f}: "
                  f"{clusters[i]} → {clusters[i+1]} "
                  f"(-{reduction}, -{pct_reduction:.1f}%)")
    
    return elbow_threshold, elbow_clusters

if __name__ == "__main__":
    optimal_threshold, optimal_clusters = analyze_ets_elbow()
    
    print("\nRecommendation:")
    print(f"Based on elbow analysis, consider using threshold {optimal_threshold:.3f}")
    print(f"This gives {optimal_clusters} clusters, balancing granularity with interpretability.")