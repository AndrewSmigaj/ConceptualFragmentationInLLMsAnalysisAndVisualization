"""
Synthetic data generation for testing Explainable Threshold Similarity (ETS).

This module creates synthetic datasets specifically designed to test various
aspects of the ETS clustering algorithm, including:
- Basic clustering functionality
- Dimension-wise thresholding
- Mixed-scale feature handling
- Explainability and semantic interpretability
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score, mutual_info_score

def generate_synthetic_ets_data(
    n_clusters: int = 3, 
    n_features: int = 10, 
    n_samples_per_cluster: int = 100,
    feature_scales: Optional[List[float]] = None, 
    cluster_std: float = 0.5, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data specifically designed to test ETS clustering.
    
    Args:
        n_clusters: Number of clusters to generate
        n_features: Number of features (dimensions)
        n_samples_per_cluster: Number of samples per cluster
        feature_scales: Optional list of scaling factors for each dimension
        cluster_std: Standard deviation of clusters
        random_state: Random seed
        
    Returns:
        Tuple of (data, labels)
    """
    np.random.seed(random_state)
    
    # Total number of samples
    n_samples = n_clusters * n_samples_per_cluster
    
    # Generate cluster centers
    centers = np.random.uniform(-10, 10, (n_clusters, n_features))
    
    # Make sure cluster centers are sufficiently separated
    min_center_distance = 5.0  # Minimum distance between cluster centers
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            while np.linalg.norm(centers[i] - centers[j]) < min_center_distance:
                centers[j] = np.random.uniform(-10, 10, n_features)
    
    # Generate data points
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_clusters):
        start_idx = i * n_samples_per_cluster
        end_idx = (i + 1) * n_samples_per_cluster
        
        # Generate points around cluster center
        X[start_idx:end_idx] = centers[i] + np.random.normal(0, cluster_std, (n_samples_per_cluster, n_features))
        y[start_idx:end_idx] = i
    
    # Apply feature scaling if provided
    if feature_scales is not None:
        if len(feature_scales) != n_features:
            raise ValueError(f"feature_scales should have length {n_features}")
        
        # Apply scaling to each dimension
        for j in range(n_features):
            X[:, j] *= feature_scales[j]
    
    return X, y

def generate_mixed_scale_dataset(
    n_clusters: int = 4, 
    n_samples_per_cluster: int = 50,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Generate a dataset with deliberately mixed feature scales.
    
    Args:
        n_clusters: Number of clusters
        n_samples_per_cluster: Samples per cluster
        random_state: Random seed
        
    Returns:
        Tuple of (data, labels, feature_scales)
    """
    # Define feature scales with dramatically different ranges
    feature_scales = [
        1.0,       # Regular scale (0-1)
        100.0,     # Large scale (0-100)
        0.01,      # Small scale (0-0.01)
        10.0,      # Medium scale (0-10)
        0.1,       # Small-medium scale (0-0.1)
        1000.0,    # Very large scale (0-1000)
        0.001,     # Very small scale (0-0.001)
        5.0,       # Medium scale (0-5)
        50.0,      # Medium-large scale (0-50)
        0.5        # Medium-small scale (0-0.5)
    ]
    
    # Generate data with these mixed scales
    data, labels = generate_synthetic_ets_data(
        n_clusters=n_clusters,
        n_features=len(feature_scales),
        n_samples_per_cluster=n_samples_per_cluster,
        feature_scales=feature_scales,
        cluster_std=0.2,  # Keep clusters tight
        random_state=random_state
    )
    
    return data, labels, feature_scales

def generate_dimension_threshold_dataset(
    n_samples: int = 300,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate dataset where clusters are separated by specific dimension thresholds.
    
    Creates a dataset where each cluster pair is distinguished by specific dimensions,
    making it perfect for testing the ETS dimension-wise threshold mechanism.
    
    Args:
        n_samples: Total number of samples
        random_state: Random seed
        
    Returns:
        Tuple of (data, labels, distinguishing_dimensions)
    """
    np.random.seed(random_state)
    
    # Define 3 clusters in 5D space
    n_features = 5
    n_clusters = 3
    samples_per_cluster = n_samples // n_clusters
    
    # Create empty arrays
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    
    # Define base values and thresholds
    base_values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    
    # Define which dimensions distinguish each cluster
    # Cluster 0: base values
    # Cluster 1: differs in dimensions 0 and 1
    # Cluster 2: differs in dimensions 2, 3, and 4
    
    cluster_shifts = {
        0: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        1: np.array([3.0, 3.0, 0.0, 0.0, 0.0]),
        2: np.array([0.0, 0.0, 3.0, 3.0, 3.0])
    }
    
    distinguishing_dimensions = {
        "0_vs_1": [0, 1],
        "0_vs_2": [2, 3, 4],
        "1_vs_2": [0, 1, 2, 3, 4]
    }
    
    # Generate data for each cluster
    for cluster_id in range(n_clusters):
        start_idx = cluster_id * samples_per_cluster
        end_idx = (cluster_id + 1) * samples_per_cluster
        
        # Generate points using base values plus cluster-specific shifts
        cluster_center = base_values + cluster_shifts[cluster_id]
        X[start_idx:end_idx] = cluster_center + np.random.normal(0, 0.3, (samples_per_cluster, n_features))
        y[start_idx:end_idx] = cluster_id
    
    return X, y, distinguishing_dimensions

def generate_semantic_dimension_dataset(
    n_samples: int = 200,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    """
    Generate dataset with semantically meaningful dimensions and thresholds.
    
    Creates a dataset where dimensions represent intuitive concepts and clusters
    are defined by meaningful thresholds on these dimensions.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed
        
    Returns:
        Tuple of (data, labels, feature_names, dimension_explanations)
    """
    np.random.seed(random_state)
    
    # Define semantic dimensions
    feature_names = [
        "height_cm",
        "weight_kg",
        "age_years",
        "income_thousands",
        "education_years",
        "travel_distance_km",
        "sleep_hours",
        "screen_time_hours"
    ]
    
    # Define realistic value ranges for each dimension
    value_ranges = {
        "height_cm": (150, 190),            # 150-190 cm
        "weight_kg": (50, 100),             # 50-100 kg
        "age_years": (18, 80),              # 18-80 years
        "income_thousands": (20, 150),      # $20k-$150k
        "education_years": (8, 20),         # 8-20 years of education
        "travel_distance_km": (1, 50),      # 1-50 km daily travel
        "sleep_hours": (4, 10),             # 4-10 hours of sleep
        "screen_time_hours": (1, 12)        # 1-12 hours of screen time
    }
    
    # Semantic cluster definitions
    clusters = [
        {
            "name": "Young Urban Professionals",
            "conditions": {
                "age_years": (22, 35),
                "income_thousands": (70, 150),
                "education_years": (16, 20),
                "travel_distance_km": (5, 20),
                "screen_time_hours": (6, 12)
            }
        },
        {
            "name": "Suburban Family",
            "conditions": {
                "age_years": (30, 50),
                "income_thousands": (50, 100),
                "education_years": (12, 18),
                "travel_distance_km": (15, 40),
                "sleep_hours": (6, 8)
            }
        },
        {
            "name": "Retired Seniors",
            "conditions": {
                "age_years": (65, 80),
                "income_thousands": (20, 60),
                "education_years": (8, 16),
                "travel_distance_km": (1, 10),
                "sleep_hours": (7, 10)
            }
        },
        {
            "name": "Health Conscious",
            "conditions": {
                "weight_kg": (50, 75),
                "sleep_hours": (7, 10),
                "screen_time_hours": (1, 4)
            }
        }
    ]
    
    # Generate random data covering the entire range
    n_features = len(feature_names)
    X = np.zeros((n_samples, n_features))
    
    for i, feature in enumerate(feature_names):
        min_val, max_val = value_ranges[feature]
        X[:, i] = np.random.uniform(min_val, max_val, n_samples)
    
    # Assign cluster labels
    y = np.full(n_samples, -1)  # -1 for unassigned
    
    for sample_idx in range(n_samples):
        # Check which cluster this sample belongs to
        for cluster_idx, cluster in enumerate(clusters):
            belongs_to_cluster = True
            
            for feature, (min_val, max_val) in cluster["conditions"].items():
                feature_idx = feature_names.index(feature)
                if not (min_val <= X[sample_idx, feature_idx] <= max_val):
                    belongs_to_cluster = False
                    break
            
            if belongs_to_cluster:
                y[sample_idx] = cluster_idx
                break
    
    # Assign remaining unassigned samples
    unassigned = (y == -1)
    n_unassigned = np.sum(unassigned)
    if n_unassigned > 0:
        y[unassigned] = np.random.randint(0, len(clusters), n_unassigned)
    
    # Create explanations for the dimensions
    dimension_explanations = {
        feature: f"Represents a person's {feature.replace('_', ' ')}"
        for feature in feature_names
    }
    
    # Add cluster explanations
    cluster_explanations = {
        i: {
            "name": cluster["name"],
            "defining_features": list(cluster["conditions"].keys())
        }
        for i, cluster in enumerate(clusters)
    }
    
    return X, y, feature_names, {
        "dimensions": dimension_explanations,
        "clusters": cluster_explanations
    }

def generate_explanation_test_cases() -> List[Dict]:
    """
    Generate specific test cases for validating ETS explanation functionality.
    
    Returns:
        List of test cases with points and expected explanation results
    """
    # Create test cases designed to validate explanation functionality
    test_cases = [
        # Case 1: Points differ in exactly one dimension
        {
            "point1": np.array([1.0, 2.0, 3.0, 4.0]),
            "point2": np.array([1.0, 2.0, 8.0, 4.0]),  # Differs only in dimension 2
            "thresholds": np.array([0.5, 0.5, 0.5, 0.5]),
            "expected_similar": False,
            "expected_distinguishing_dims": [2]
        },
        # Case 2: Points differ in multiple dimensions
        {
            "point1": np.array([1.0, 2.0, 3.0, 4.0]),
            "point2": np.array([3.0, 2.0, 8.0, 4.0]),  # Differs in dimensions 0 and 2
            "thresholds": np.array([0.5, 0.5, 0.5, 0.5]),
            "expected_similar": False,
            "expected_distinguishing_dims": [0, 2]
        },
        # Case 3: Points are similar (no distinguishing dimensions)
        {
            "point1": np.array([1.0, 2.0, 3.0, 4.0]),
            "point2": np.array([1.1, 1.9, 3.1, 4.1]),  # All within thresholds
            "thresholds": np.array([0.5, 0.5, 0.5, 0.5]),
            "expected_similar": True,
            "expected_distinguishing_dims": []
        },
        # Case 4: Edge case - right at the threshold boundary
        {
            "point1": np.array([1.0, 2.0, 3.0, 4.0]),
            "point2": np.array([1.0, 2.0, 3.5, 4.0]),  # Dimension 2 exactly at threshold
            "thresholds": np.array([0.5, 0.5, 0.5, 0.5]),
            "expected_similar": True,
            "expected_distinguishing_dims": []
        },
        # Case 5: Points with feature names
        {
            "point1": np.array([170.0, 70.0, 30.0, 50.0]),  # height, weight, age, income
            "point2": np.array([175.0, 85.0, 32.0, 55.0]),
            "thresholds": np.array([10.0, 20.0, 5.0, 10.0]),
            "feature_names": ["height_cm", "weight_kg", "age_years", "income_thousands"],
            "expected_similar": True,
            "expected_distinguishing_dims": []
        }
    ]
    
    return test_cases

def verify_explanation_correctness(
    point1: np.ndarray,
    point2: np.ndarray,
    thresholds: np.ndarray,
    explanation: Dict
) -> Tuple[bool, str]:
    """
    Verify that the ETS explanation correctly identifies the distinguishing dimensions.
    
    Args:
        point1, point2: The points being compared
        thresholds: The thresholds used
        explanation: The explanation returned by explain_ets_similarity
        
    Returns:
        is_correct: Boolean indicating if explanation is correct
        error_message: Description of any errors found
    """
    # Compute dimension-wise absolute differences
    diffs = np.abs(point1 - point2)
    
    # Verify is_similar flag
    is_similar_expected = np.all(diffs <= thresholds)
    if explanation["is_similar"] != is_similar_expected:
        return False, f"is_similar={explanation['is_similar']}, expected {is_similar_expected}"
    
    # Verify distinguishing dimensions
    expected_distinguishing_dims = [i for i, (diff, threshold) in enumerate(zip(diffs, thresholds)) 
                                   if diff > threshold]
    
    # Get indices from dimension names in explanation
    explanation_dims = []
    for dim_name in explanation["distinguishing_dimensions"]:
        # Extract index from dimension name (e.g., "Dimension 2" -> 2)
        if "Dimension" in dim_name:
            try:
                dim_idx = int(dim_name.split()[-1])
                explanation_dims.append(dim_idx)
            except ValueError:
                pass
    
    # Sort both lists for comparison
    expected_distinguishing_dims.sort()
    explanation_dims.sort()
    
    if explanation_dims != expected_distinguishing_dims:
        return False, f"distinguishing_dimensions={explanation_dims}, expected {expected_distinguishing_dims}"
    
    # Verify dimensions count
    if explanation["num_dimensions_compared"] != len(thresholds):
        return False, f"num_dimensions_compared={explanation['num_dimensions_compared']}, expected {len(thresholds)}"
    
    # Verify dimensions within threshold
    expected_within_threshold = np.sum(diffs <= thresholds)
    if explanation["num_dimensions_within_threshold"] != expected_within_threshold:
        return False, f"num_dimensions_within_threshold={explanation['num_dimensions_within_threshold']}, expected {expected_within_threshold}"
    
    return True, "Explanation is correct"

def evaluate_clustering_metrics(
    data: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray
) -> Dict:
    """
    Compute standard clustering evaluation metrics.
    
    Args:
        data: The dataset
        true_labels: Ground truth cluster labels
        predicted_labels: Predicted cluster labels
        
    Returns:
        metrics: Dictionary with ARI, silhouette score, MI, etc.
    """
    # Silhouette score requires at least 2 clusters
    n_clusters_true = len(np.unique(true_labels))
    n_clusters_pred = len(np.unique(predicted_labels))
    
    metrics = {
        "n_clusters_true": n_clusters_true,
        "n_clusters_predicted": n_clusters_pred,
        "adjusted_rand_index": adjusted_rand_score(true_labels, predicted_labels),
        "mutual_information": mutual_info_score(true_labels, predicted_labels),
    }
    
    # Add silhouette score if possible
    if n_clusters_pred >= 2:
        metrics["silhouette_score"] = silhouette_score(data, predicted_labels)
    else:
        metrics["silhouette_score"] = None
    
    return metrics

def visualize_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Visualization",
    dimensions: Tuple[int, int] = (0, 1)
) -> plt.Figure:
    """
    Create 2D visualization of the clustered data.
    
    Args:
        data: The dataset
        labels: Cluster labels
        title: Plot title
        dimensions: Tuple of dimensions to plot (for high-dimensional data)
        
    Returns:
        fig: Matplotlib figure
    """
    # If data has more than 2 dimensions, use PCA to reduce dimensions
    if data.shape[1] > 2:
        # Extract specified dimensions
        if dimensions is not None:
            dim1, dim2 = dimensions
            X_reduced = data[:, [dim1, dim2]]
        else:
            # Or use PCA
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(data)
    else:
        X_reduced = data
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get number of clusters
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        ax.scatter(
            X_reduced[mask, 0],
            X_reduced[mask, 1],
            label=f"Cluster {label}",
            alpha=0.7
        )
    
    ax.set_title(title)
    ax.set_xlabel(f"Dimension {dimensions[0]}" if dimensions else "PCA Component 1")
    ax.set_ylabel(f"Dimension {dimensions[1]}" if dimensions else "PCA Component 2")
    ax.legend()
    
    return fig

def visualize_thresholds(
    data: np.ndarray,
    thresholds: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    Visualize the computed thresholds for each dimension.
    
    Args:
        data: The dataset
        thresholds: Dimension thresholds computed by ETS
        feature_names: Optional names for dimensions
        
    Returns:
        fig: Matplotlib figure showing thresholds relative to data ranges
    """
    n_features = data.shape[1]
    
    # Set feature names if not provided
    if feature_names is None:
        feature_names = [f"Dimension {i}" for i in range(n_features)]
    
    # Compute data ranges
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_range = data_max - data_min
    
    # Compute threshold as percentage of range
    threshold_percent = thresholds / data_range * 100
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot absolute thresholds
    axes[0].bar(feature_names, thresholds)
    axes[0].set_title("Absolute Thresholds per Dimension")
    axes[0].set_ylabel("Threshold Value")
    axes[0].set_xticklabels(feature_names, rotation=45, ha="right")
    
    # Plot relative thresholds (as percentage of range)
    axes[1].bar(feature_names, threshold_percent)
    axes[1].set_title("Relative Thresholds (% of Dimension Range)")
    axes[1].set_ylabel("Threshold (% of Range)")
    axes[1].set_xticklabels(feature_names, rotation=45, ha="right")
    
    plt.tight_layout()
    return fig

def visualize_explanation(
    point1: np.ndarray,
    point2: np.ndarray,
    thresholds: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    Visualize the explanation for why two points are similar or different.
    
    Args:
        point1, point2: Points to compare
        thresholds: Dimension thresholds
        feature_names: Optional names for dimensions
        
    Returns:
        fig: Matplotlib figure showing dimension-wise differences vs thresholds
    """
    n_features = len(point1)
    
    # Set feature names if not provided
    if feature_names is None:
        feature_names = [f"Dimension {i}" for i in range(n_features)]
    
    # Compute dimension-wise absolute differences
    diffs = np.abs(point1 - point2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create indices for the bars
    indices = np.arange(n_features)
    
    # Plot differences and thresholds
    width = 0.35
    ax.bar(indices - width/2, diffs, width, label="Absolute Difference")
    ax.bar(indices + width/2, thresholds, width, label="Threshold")
    
    # Highlight dimensions that exceed threshold
    for i in range(n_features):
        if diffs[i] > thresholds[i]:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color='red')
    
    # Show whether points are similar
    is_similar = np.all(diffs <= thresholds)
    similarity_text = "Points are in the same cluster" if is_similar else "Points are in different clusters"
    ax.set_title(f"Dimension-wise Comparison: {similarity_text}")
    
    # Set labels
    ax.set_xlabel("Dimensions")
    ax.set_ylabel("Value")
    ax.set_xticks(indices)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.legend()
    
    # Add values annotation
    for i in range(n_features):
        ax.annotate(f"{point1[i]:.2f}",
                  xy=(i - width/2, diffs[i] + 0.1),
                  xytext=(0, 3),
                  textcoords="offset points",
                  ha='center', va='bottom',
                  fontsize=8)
        ax.annotate(f"{point2[i]:.2f}",
                  xy=(i - width/2, 0),
                  xytext=(0, -10),
                  textcoords="offset points",
                  ha='center', va='top',
                  fontsize=8)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Generate and visualize basic test datasets
    
    # 1. Basic clustering dataset
    print("Generating basic clustering dataset...")
    data, labels = generate_synthetic_ets_data(n_clusters=4, n_features=2)
    fig = visualize_clusters(data, labels, "Basic Clustering Dataset")
    plt.savefig("basic_clusters.png")
    
    # 2. Mixed scale dataset
    print("Generating mixed scale dataset...")
    data, labels, scales = generate_mixed_scale_dataset()
    fig = visualize_clusters(data, labels, "Mixed Scale Dataset")
    plt.savefig("mixed_scale_clusters.png")
    
    # 3. Dimension threshold dataset
    print("Generating dimension threshold dataset...")
    data, labels, dim_info = generate_dimension_threshold_dataset()
    fig = visualize_clusters(data, labels, "Dimension Threshold Dataset")
    plt.savefig("dimension_threshold_clusters.png")
    
    # 4. Semantic dimension dataset
    print("Generating semantic dimension dataset...")
    data, labels, feature_names, explanations = generate_semantic_dimension_dataset()
    fig = visualize_clusters(data, labels, "Semantic Dimension Dataset")
    plt.savefig("semantic_dimension_clusters.png")
    
    print("Done. Saved visualization plots.")