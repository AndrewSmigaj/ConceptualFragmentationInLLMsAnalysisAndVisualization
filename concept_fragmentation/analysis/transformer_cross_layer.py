"""
Cross-layer analysis for transformer models.

This module provides specialized functionality for analyzing relationships
and information flow between layers in transformer models, including attention
pattern analysis, token representation tracking, and embedding space comparisons.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import warnings
import logging
from pathlib import Path
import re
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict

from ..metrics.transformer_metrics import (
    calculate_attention_entropy,
    analyze_attention_patterns,
    calculate_activation_statistics,
    analyze_cross_attention_consistency,
    TransformerMetricsCalculator
)

from .transformer_dimensionality import (
    TransformerDimensionalityReducer,
    DimensionalityReductionResult
)

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class CrossLayerTransformerAnalysisResult:
    """
    Results from cross-layer transformer analysis.
    
    Attributes:
        layer_relationships: Dictionary mapping layer pairs to relationship metrics
        attention_flow: Dictionary with attention flow metrics
        representation_evolution: Dictionary with representation evolution metrics
        token_trajectory: Dictionary with token trajectory information
        embedding_comparisons: Dictionary with embedding space comparison metrics
        layers_analyzed: List of layer names that were analyzed
        attention_heads_analyzed: Dictionary mapping layer names to head indices
        plot_data: Dictionary with data for visualization
    """
    layer_relationships: Dict[str, Any]
    attention_flow: Dict[str, Any]
    representation_evolution: Dict[str, Any]
    token_trajectory: Dict[str, Any]
    embedding_comparisons: Dict[str, Any]
    layers_analyzed: List[str]
    attention_heads_analyzed: Dict[str, List[int]]
    plot_data: Dict[str, Any]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the analysis results.
        
        Returns:
            Dictionary with summary metrics
        """
        summary = {}
        
        # Layer relationships summary
        if self.layer_relationships:
            # Extract key metrics
            if "similarity_matrix" in self.layer_relationships:
                sim_values = []
                for _, matrix in self.layer_relationships["similarity_matrix"].items():
                    sim_values.extend(matrix.flatten().tolist())
                
                if sim_values:
                    summary["mean_layer_similarity"] = float(np.mean(sim_values))
                    summary["max_layer_similarity"] = float(np.max(sim_values))
                    summary["min_layer_similarity"] = float(np.min(sim_values))
        
        # Attention flow summary
        if self.attention_flow:
            if "layer_entropy_differences" in self.attention_flow:
                summary["mean_entropy_change"] = float(np.mean(list(
                    self.attention_flow["layer_entropy_differences"].values()
                )))
                
            if "attention_pattern_consistency" in self.attention_flow:
                summary["mean_attention_consistency"] = float(np.mean(list(
                    self.attention_flow["attention_pattern_consistency"].values()
                )))
        
        # Token trajectory summary
        if self.token_trajectory:
            if "token_stability" in self.token_trajectory:
                summary["mean_token_stability"] = float(np.mean(list(
                    self.token_trajectory["token_stability"].values()
                )))
            
            if "focused_tokens" in self.token_trajectory:
                summary["focused_tokens"] = self.token_trajectory["focused_tokens"]
        
        # Representation evolution summary
        if self.representation_evolution:
            if "semantic_shift" in self.representation_evolution:
                summary["mean_semantic_shift"] = float(np.mean(list(
                    self.representation_evolution["semantic_shift"].values()
                )))
        
        # Overall summary
        summary["layers_analyzed"] = len(self.layers_analyzed)
        summary["total_attention_heads"] = sum(len(heads) for heads in self.attention_heads_analyzed.values())
        
        return summary


class TransformerCrossLayerAnalyzer:
    """
    Analyzer for cross-layer relationships in transformer models.
    
    This class provides methods for analyzing relationships between layers
    in transformer models, focusing on attention patterns, token representations,
    and information flow across the model.
    
    Attributes:
        dim_reducer: Dimensionality reducer for high-dimensional transformer activations
        metrics_calculator: Calculator for transformer-specific metrics
        cache_dir: Directory for caching results
        use_cache: Whether to use disk caching
        random_state: Random seed for reproducibility
        _cache: In-memory cache for analysis results
    """
    
    def __init__(
        self,
        dim_reducer: Optional[TransformerDimensionalityReducer] = None,
        metrics_calculator: Optional[TransformerMetricsCalculator] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        random_state: int = 42
    ):
        """
        Initialize the transformer cross-layer analyzer.
        
        Args:
            dim_reducer: Dimensionality reducer for high-dimensional transformer activations
            metrics_calculator: Calculator for transformer-specific metrics
            cache_dir: Directory for caching results
            use_cache: Whether to use disk caching
            random_state: Random seed for reproducibility
        """
        # Create default components if not provided
        self.dim_reducer = dim_reducer or TransformerDimensionalityReducer(
            cache_dir=cache_dir,
            random_state=random_state,
            use_cache=use_cache
        )
        
        self.metrics_calculator = metrics_calculator or TransformerMetricsCalculator(
            use_cache=use_cache
        )
        
        # Set attributes
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        self.random_state = random_state
        self._cache = {}
    
    def analyze_layer_relationships(
        self,
        layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
        similarity_metric: str = "cosine",
        dimensionality_reduction: bool = True,
        n_components: int = 50
    ) -> Dict[str, Any]:
        """
        Analyze relationships between layers based on activations.
        
        Args:
            layer_activations: Dictionary mapping layer names to activation tensors
            similarity_metric: Metric to use for similarity calculation
            dimensionality_reduction: Whether to apply dimensionality reduction
            n_components: Number of components for dimensionality reduction
            
        Returns:
            Dictionary with layer relationship metrics
        """
        # First, ensure comparable dimensions across layers
        processed_activations = {}
        activation_stats = {}
        
        for layer_name, activations in layer_activations.items():
            # Convert to numpy if needed
            if isinstance(activations, torch.Tensor):
                activations = activations.detach().cpu().numpy()
            
            # Handle different shapes - we need [n_samples, n_features]
            if len(activations.shape) == 3:  # [batch_size, seq_len, hidden_dim]
                # Average across sequence dimension
                activations = np.mean(activations, axis=1)
            
            # Apply dimensionality reduction if needed
            if dimensionality_reduction and activations.shape[1] > n_components:
                result = self.dim_reducer.reduce_dimensionality(
                    activations=activations,
                    n_components=n_components,
                    method="auto",
                    layer_name=layer_name
                )
                
                if result.success:
                    activations = result.reduced_activations
                    activation_stats[layer_name] = {
                        "original_dim": result.original_dim,
                        "reduced_dim": result.reduced_dim,
                        "method": result.method
                    }
            
            processed_activations[layer_name] = activations
        
        # Compute pairwise similarities between layer representations
        layer_names = sorted(processed_activations.keys())
        similarity_matrix = {}
        
        for i, layer1 in enumerate(layer_names):
            activations1 = processed_activations[layer1]
            
            for j, layer2 in enumerate(layer_names):
                if i >= j:  # Only compute upper triangle
                    continue
                    
                activations2 = processed_activations[layer2]
                
                # Compute centroid similarities
                if similarity_metric == "cosine":
                    # Compute layer-wise centroid
                    centroid1 = np.mean(activations1, axis=0)
                    centroid2 = np.mean(activations2, axis=0)
                    
                    # Compute cosine similarity
                    norm1 = np.linalg.norm(centroid1)
                    norm2 = np.linalg.norm(centroid2)
                    
                    if norm1 > 0 and norm2 > 0:
                        sim = np.dot(centroid1, centroid2) / (norm1 * norm2)
                    else:
                        sim = 0.0
                        
                    similarity_matrix[(layer1, layer2)] = float(sim)
                
                elif similarity_metric == "cka":  # Centered Kernel Alignment
                    # Compute CKA similarity
                    from scipy.stats import spearmanr
                    
                    # Linear CKA
                    X = activations1
                    Y = activations2
                    
                    # Center the matrices
                    X_centered = X - X.mean(axis=0)
                    Y_centered = Y - Y.mean(axis=0)
                    
                    # Compute Gram matrices
                    K_X = X_centered @ X_centered.T
                    K_Y = Y_centered @ Y_centered.T
                    
                    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
                    HSIC_XY = np.sum(K_X * K_Y)
                    HSIC_XX = np.sum(K_X * K_X)
                    HSIC_YY = np.sum(K_Y * K_Y)
                    
                    # Compute CKA
                    if HSIC_XX > 0 and HSIC_YY > 0:
                        cka = HSIC_XY / np.sqrt(HSIC_XX * HSIC_YY)
                    else:
                        cka = 0.0
                        
                    similarity_matrix[(layer1, layer2)] = float(cka)
                
                elif similarity_metric == "correlation":
                    # Compute mean activation correlation
                    mean1 = np.mean(activations1, axis=0)
                    mean2 = np.mean(activations2, axis=0)
                    
                    # Compute correlation
                    corr = np.corrcoef(mean1, mean2)[0, 1]
                    similarity_matrix[(layer1, layer2)] = float(corr)
        
        # Create layer relationship graph
        G = nx.Graph()
        
        # Add nodes for each layer
        for layer_name in layer_names:
            G.add_node(layer_name, type="layer")
        
        # Add edges for layer relationships
        for (layer1, layer2), sim in similarity_matrix.items():
            G.add_edge(layer1, layer2, weight=sim, similarity=sim)
        
        # Return results
        return {
            "similarity_matrix": similarity_matrix,
            "graph": G,
            "layers": layer_names,
            "activation_stats": activation_stats,
            "similarity_metric": similarity_metric
        }
    
    def analyze_attention_flow(
        self,
        attention_data: Dict[str, Union[torch.Tensor, np.ndarray]],
        layer_order: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze attention flow across transformer layers.
        
        Args:
            attention_data: Dictionary mapping layer names to attention matrices
            layer_order: Order of layers for consistent analysis
            
        Returns:
            Dictionary with attention flow metrics
        """
        # Ensure layer order is consistent
        if layer_order is None:
            layer_order = sorted(attention_data.keys())
        
        # Compute attention entropy for each layer
        layer_entropy = {}
        
        for layer_name, attention_probs in attention_data.items():
            if layer_name not in layer_order:
                continue
                
            # Convert to numpy if needed
            if isinstance(attention_probs, torch.Tensor):
                attention_probs = attention_probs.detach().cpu().numpy()
            
            # Calculate entropy
            entropy = calculate_attention_entropy(attention_probs)
            
            # Average across batch and heads if needed
            if len(entropy.shape) == 2:  # [batch_size, n_heads]
                entropy = float(np.mean(entropy))
            elif len(entropy.shape) == 1:  # [n_heads]
                entropy = float(np.mean(entropy))
            
            layer_entropy[layer_name] = entropy
        
        # Compute entropy differences between consecutive layers
        layer_entropy_differences = {}
        
        for i in range(len(layer_order) - 1):
            layer1 = layer_order[i]
            layer2 = layer_order[i + 1]
            
            if layer1 in layer_entropy and layer2 in layer_entropy:
                diff = layer_entropy[layer2] - layer_entropy[layer1]
                layer_entropy_differences[(layer1, layer2)] = float(diff)
        
        # Compute attention pattern consistency across layers
        attention_consistency = analyze_cross_attention_consistency(attention_data)
        
        # Get attention pattern consistency between consecutive layers
        pattern_consistency = {}
        
        for i in range(len(layer_order) - 1):
            layer1 = layer_order[i]
            layer2 = layer_order[i + 1]
            
            key = (layer1, layer2)
            reverse_key = (layer2, layer1)
            
            if key in attention_consistency.get("layer_consistency", {}):
                pattern_consistency[key] = attention_consistency["layer_consistency"][key]
            elif reverse_key in attention_consistency.get("layer_consistency", {}):
                pattern_consistency[key] = attention_consistency["layer_consistency"][reverse_key]
        
        # Compute attention flow direction
        # Positive means attention gets more focused, negative means more dispersed
        attention_flow_direction = {}
        
        for (layer1, layer2), diff in layer_entropy_differences.items():
            attention_flow_direction[(layer1, layer2)] = -1 if diff > 0 else 1
        
        # Create attention flow graph
        G = nx.DiGraph()
        
        # Add nodes for each layer
        for layer_name in layer_order:
            if layer_name in layer_entropy:
                G.add_node(layer_name, entropy=layer_entropy[layer_name], type="layer")
        
        # Add directed edges for attention flow
        for (layer1, layer2), flow in attention_flow_direction.items():
            G.add_edge(
                layer1, 
                layer2, 
                flow=flow,
                entropy_diff=layer_entropy_differences.get((layer1, layer2), 0.0),
                consistency=pattern_consistency.get((layer1, layer2), 0.0)
            )
        
        # Return results
        return {
            "layer_entropy": layer_entropy,
            "layer_entropy_differences": layer_entropy_differences,
            "attention_flow_direction": attention_flow_direction,
            "attention_pattern_consistency": pattern_consistency,
            "flow_graph": G,
            "attention_consistency": attention_consistency
        }
    
    def analyze_token_trajectory(
        self,
        layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
        token_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_strings: Optional[List[str]] = None,
        focus_tokens: Optional[List[str]] = None,
        layer_order: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze how token representations evolve across layers.
        
        Args:
            layer_activations: Dictionary mapping layer names to token activations
                              [batch_size, seq_len, hidden_dim]
            token_ids: Token IDs for each position [batch_size, seq_len]
            token_strings: List of token strings
            focus_tokens: Optional list of tokens to focus the analysis on
            layer_order: Order of layers for consistent analysis
            
        Returns:
            Dictionary with token trajectory metrics
        """
        # Ensure layer order is consistent
        if layer_order is None:
            layer_order = sorted(layer_activations.keys())
        
        # Process activations to ensure consistent format
        processed_activations = {}
        
        for layer_name, activations in layer_activations.items():
            if layer_name not in layer_order:
                continue
                
            # Convert to numpy if needed
            if isinstance(activations, torch.Tensor):
                activations = activations.detach().cpu().numpy()
            
            # Ensure 3D shape [batch_size, seq_len, hidden_dim]
            if len(activations.shape) != 3:
                continue
            
            processed_activations[layer_name] = activations
        
        # If no valid activations, return empty results
        if not processed_activations:
            return {}
        
        # Get dimensions from first layer
        first_layer = next(iter(processed_activations.values()))
        batch_size, seq_len, hidden_dim = first_layer.shape
        
        # Determine tokens to focus on
        token_indices = []
        focused_token_strings = []
        
        if focus_tokens and token_strings:
            # Find indices of specified tokens
            for token in focus_tokens:
                for i, t_str in enumerate(token_strings):
                    if token.lower() in t_str.lower():
                        token_indices.append(i % seq_len)
                        focused_token_strings.append(t_str)
        
        # If no focus tokens specified or found, use the first few tokens
        if not token_indices:
            token_indices = list(range(min(5, seq_len)))
            if token_strings:
                focused_token_strings = [token_strings[i] for i in token_indices if i < len(token_strings)]
        
        # Calculate token representation stability across layers
        token_stability = {}
        semantic_shift = {}
        
        for token_idx in token_indices:
            token_key = f"token_{token_idx}"
            if token_idx < len(focused_token_strings):
                token_key = focused_token_strings[token_idx]
            
            token_stability[token_key] = {}
            semantic_shift[token_key] = {}
            
            for i in range(len(layer_order) - 1):
                layer1 = layer_order[i]
                layer2 = layer_order[i + 1]
                
                if layer1 not in processed_activations or layer2 not in processed_activations:
                    continue
                
                # Get token representations from each layer
                token_rep1 = processed_activations[layer1][0, token_idx]
                token_rep2 = processed_activations[layer2][0, token_idx]
                
                # Calculate cosine similarity between representations
                norm1 = np.linalg.norm(token_rep1)
                norm2 = np.linalg.norm(token_rep2)
                
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(token_rep1, token_rep2) / (norm1 * norm2)
                else:
                    sim = 0.0
                
                # Store similarity as stability measure
                token_stability[token_key][(layer1, layer2)] = float(sim)
                
                # Calculate semantic shift as 1 - similarity
                semantic_shift[token_key][(layer1, layer2)] = float(1.0 - sim)
        
        # Compute average token stability and semantic shift
        avg_token_stability = {}
        avg_semantic_shift = {}
        
        for i in range(len(layer_order) - 1):
            layer1 = layer_order[i]
            layer2 = layer_order[i + 1]
            
            # Collect all token stabilities for this layer pair
            stabilities = []
            shifts = []
            
            for token_key in token_stability:
                if (layer1, layer2) in token_stability[token_key]:
                    stabilities.append(token_stability[token_key][(layer1, layer2)])
                
                if (layer1, layer2) in semantic_shift[token_key]:
                    shifts.append(semantic_shift[token_key][(layer1, layer2)])
            
            # Compute averages
            if stabilities:
                avg_token_stability[(layer1, layer2)] = float(np.mean(stabilities))
            
            if shifts:
                avg_semantic_shift[(layer1, layer2)] = float(np.mean(shifts))
        
        # Create token trajectory graph
        G = nx.DiGraph()
        
        # Add nodes for each token in each layer
        for layer_name in layer_order:
            if layer_name not in processed_activations:
                continue
                
            for token_idx in token_indices:
                token_key = f"token_{token_idx}"
                if token_idx < len(focused_token_strings):
                    token_key = focused_token_strings[token_idx]
                
                node_id = f"{layer_name}_{token_key}"
                G.add_node(node_id, layer=layer_name, token=token_key)
        
        # Add edges for token trajectories
        for token_idx in token_indices:
            token_key = f"token_{token_idx}"
            if token_idx < len(focused_token_strings):
                token_key = focused_token_strings[token_idx]
            
            for i in range(len(layer_order) - 1):
                layer1 = layer_order[i]
                layer2 = layer_order[i + 1]
                
                if layer1 not in processed_activations or layer2 not in processed_activations:
                    continue
                
                # Get stability and shift for this token and layer pair
                stability = token_stability[token_key].get((layer1, layer2), 0.0)
                shift = semantic_shift[token_key].get((layer1, layer2), 1.0)
                
                # Add edge
                G.add_edge(
                    f"{layer1}_{token_key}",
                    f"{layer2}_{token_key}",
                    stability=stability,
                    shift=shift
                )
        
        # Return results
        return {
            "token_stability": token_stability,
            "semantic_shift": semantic_shift,
            "avg_token_stability": avg_token_stability,
            "avg_semantic_shift": avg_semantic_shift,
            "trajectory_graph": G,
            "focused_tokens": focused_token_strings,
            "token_indices": token_indices
        }
    
    def analyze_representation_evolution(
        self,
        layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
        class_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
        layer_order: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze how representations evolve across layers.
        
        Args:
            layer_activations: Dictionary mapping layer names to activations
            class_labels: Optional class labels for computing class separation
            layer_order: Order of layers for consistent analysis
            
        Returns:
            Dictionary with representation evolution metrics
        """
        # Ensure layer order is consistent
        if layer_order is None:
            layer_order = sorted(layer_activations.keys())
        
        # Process activations to ensure consistent format
        processed_activations = {}
        
        for layer_name, activations in layer_activations.items():
            if layer_name not in layer_order:
                continue
                
            # Convert to numpy if needed
            if isinstance(activations, torch.Tensor):
                activations = activations.detach().cpu().numpy()
            
            # Handle different shapes
            if len(activations.shape) == 3:  # [batch_size, seq_len, hidden_dim]
                # Average across sequence dimension
                activations = np.mean(activations, axis=1)
            
            processed_activations[layer_name] = activations
        
        # Calculate representation divergence across layers
        representation_divergence = {}
        
        for i in range(len(layer_order) - 1):
            layer1 = layer_order[i]
            layer2 = layer_order[i + 1]
            
            if layer1 not in processed_activations or layer2 not in processed_activations:
                continue
            
            # Get activations from each layer
            act1 = processed_activations[layer1]
            act2 = processed_activations[layer2]
            
            # Calculate average cosine distance between corresponding sample representations
            divergence = 0.0
            n_samples = min(act1.shape[0], act2.shape[0])
            
            for j in range(n_samples):
                vec1 = act1[j]
                vec2 = act2[j]
                
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(vec1, vec2) / (norm1 * norm2)
                    # Convert similarity to distance
                    dist = 1.0 - sim
                else:
                    dist = 1.0
                
                divergence += dist
            
            # Compute average divergence
            if n_samples > 0:
                divergence /= n_samples
            
            representation_divergence[(layer1, layer2)] = float(divergence)
        
        # Compute class separation if class labels provided
        class_separation = {}
        
        if class_labels is not None:
            # Convert to numpy if needed
            if isinstance(class_labels, torch.Tensor):
                class_labels = class_labels.detach().cpu().numpy()
            
            # Flatten if needed
            if len(class_labels.shape) > 1:
                class_labels = class_labels.flatten()
            
            # Compute separation for each layer
            for layer_name, activations in processed_activations.items():
                # Limit to matching samples
                n_samples = min(activations.shape[0], len(class_labels))
                act = activations[:n_samples]
                labels = class_labels[:n_samples]
                
                # Get unique classes
                unique_classes = np.unique(labels)
                
                if len(unique_classes) <= 1:
                    continue
                
                # Calculate average distance between class centroids
                class_centroids = {}
                
                for class_id in unique_classes:
                    mask = (labels == class_id)
                    if np.any(mask):
                        class_centroids[class_id] = np.mean(act[mask], axis=0)
                
                # Calculate average distance between class centroids
                total_dist = 0.0
                count = 0
                
                for i, class1 in enumerate(unique_classes):
                    if class1 not in class_centroids:
                        continue
                        
                    centroid1 = class_centroids[class1]
                    
                    for j, class2 in enumerate(unique_classes[i+1:]):
                        if class2 not in class_centroids:
                            continue
                            
                        centroid2 = class_centroids[class2]
                        
                        norm1 = np.linalg.norm(centroid1)
                        norm2 = np.linalg.norm(centroid2)
                        
                        if norm1 > 0 and norm2 > 0:
                            sim = np.dot(centroid1, centroid2) / (norm1 * norm2)
                            # Convert similarity to distance
                            dist = 1.0 - sim
                        else:
                            dist = 1.0
                        
                        total_dist += dist
                        count += 1
                
                # Compute average distance
                if count > 0:
                    class_separation[layer_name] = float(total_dist / count)
        
        # Calculate representation sparsity for each layer
        sparsity = {}
        
        for layer_name, activations in processed_activations.items():
            # Calculate fraction of near-zero values
            near_zero = np.abs(activations) < 1e-5
            sparsity[layer_name] = float(np.mean(near_zero))
        
        # Calculate representation compactness (average L2 norm)
        compactness = {}
        
        for layer_name, activations in processed_activations.items():
            norms = np.sqrt(np.sum(activations**2, axis=1))
            compactness[layer_name] = float(np.mean(norms))
        
        # Return results
        return {
            "representation_divergence": representation_divergence,
            "class_separation": class_separation,
            "sparsity": sparsity,
            "compactness": compactness
        }
    
    def analyze_embedding_comparisons(
        self,
        layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
        pca_components: int = 3,
        layer_order: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze embedding spaces with dimensionality reduction visualizations.
        
        Args:
            layer_activations: Dictionary mapping layer names to activations
            pca_components: Number of PCA components for visualization
            layer_order: Order of layers for consistent analysis
            
        Returns:
            Dictionary with embedding comparison metrics and visualizations
        """
        # Ensure layer order is consistent
        if layer_order is None:
            layer_order = sorted(layer_activations.keys())
        
        # Process activations and reduce dimensions for visualization
        embeddings = {}
        
        for layer_name, activations in layer_activations.items():
            if layer_name not in layer_order:
                continue
                
            # Convert to numpy if needed
            if isinstance(activations, torch.Tensor):
                activations = activations.detach().cpu().numpy()
            
            # Handle different shapes
            if len(activations.shape) == 3:  # [batch_size, seq_len, hidden_dim]
                # Average across sequence dimension
                activations = np.mean(activations, axis=1)
            
            # Apply PCA for visualization
            result = self.dim_reducer.reduce_dimensionality(
                activations=activations,
                n_components=pca_components,
                method="pca",
                layer_name=layer_name
            )
            
            if result.success:
                embeddings[layer_name] = result.reduced_activations
        
        # Compare embedding structures across layers
        embedding_similarities = {}
        
        for i, layer1 in enumerate(layer_order):
            if layer1 not in embeddings:
                continue
                
            emb1 = embeddings[layer1]
            
            for j, layer2 in enumerate(layer_order):
                if i >= j or layer2 not in embeddings:
                    continue
                    
                emb2 = embeddings[layer2]
                
                # Compute Procrustes similarity between embeddings
                from scipy.spatial.distance import cdist
                from scipy.linalg import orthogonal_procrustes
                
                # Align embeddings using Procrustes analysis
                R, scale = orthogonal_procrustes(emb1, emb2)
                
                # Apply transformation
                aligned_emb1 = scale * emb1 @ R
                
                # Compute average distance between aligned points
                distances = np.sqrt(np.sum((aligned_emb1 - emb2)**2, axis=1))
                avg_distance = float(np.mean(distances))
                
                # Convert distance to similarity
                similarity = float(np.exp(-avg_distance))
                
                embedding_similarities[(layer1, layer2)] = similarity
        
        # Return results
        return {
            "embeddings": embeddings,
            "embedding_similarities": embedding_similarities,
            "pca_components": pca_components
        }
    
    def analyze_transformer_cross_layer(
        self,
        layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
        attention_data: Dict[str, Union[torch.Tensor, np.ndarray]],
        token_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_strings: Optional[List[str]] = None,
        class_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
        focus_tokens: Optional[List[str]] = None,
        layer_order: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> CrossLayerTransformerAnalysisResult:
        """
        Perform comprehensive cross-layer analysis for transformer models.
        
        Args:
            layer_activations: Dictionary mapping layer names to activations
            attention_data: Dictionary mapping layer names to attention matrices
            token_ids: Token IDs for each position [batch_size, seq_len]
            token_strings: List of token strings
            class_labels: Optional class labels
            focus_tokens: Optional list of tokens to focus the analysis on
            layer_order: Order of layers for consistent analysis
            config: Configuration dictionary with parameters for each analysis
            
        Returns:
            CrossLayerTransformerAnalysisResult with comprehensive analysis results
        """
        # Initialize default config
        default_config = {
            "similarity_metric": "cosine",
            "dimensionality_reduction": True,
            "n_components": 50,
            "pca_components": 3
        }
        
        # Update with user config
        if config:
            default_config.update(config)
        
        # Ensure consistent layer order
        if layer_order is None:
            # Combine keys from both dictionaries
            all_keys = set(layer_activations.keys()) | set(attention_data.keys())
            layer_order = sorted(all_keys)
        
        # Analyze layer relationships
        try:
            layer_relationships = self.analyze_layer_relationships(
                layer_activations=layer_activations,
                similarity_metric=default_config["similarity_metric"],
                dimensionality_reduction=default_config["dimensionality_reduction"],
                n_components=default_config["n_components"]
            )
        except Exception as e:
            logger.error(f"Error analyzing layer relationships: {e}")
            layer_relationships = {"error": str(e)}
        
        # Analyze attention flow
        try:
            attention_flow = self.analyze_attention_flow(
                attention_data=attention_data,
                layer_order=layer_order
            )
        except Exception as e:
            logger.error(f"Error analyzing attention flow: {e}")
            attention_flow = {"error": str(e)}
        
        # Analyze token trajectory
        try:
            token_trajectory = self.analyze_token_trajectory(
                layer_activations=layer_activations,
                token_ids=token_ids,
                token_strings=token_strings,
                focus_tokens=focus_tokens,
                layer_order=layer_order
            )
        except Exception as e:
            logger.error(f"Error analyzing token trajectory: {e}")
            token_trajectory = {"error": str(e)}
        
        # Analyze representation evolution
        try:
            representation_evolution = self.analyze_representation_evolution(
                layer_activations=layer_activations,
                class_labels=class_labels,
                layer_order=layer_order
            )
        except Exception as e:
            logger.error(f"Error analyzing representation evolution: {e}")
            representation_evolution = {"error": str(e)}
        
        # Analyze embedding comparisons
        try:
            embedding_comparisons = self.analyze_embedding_comparisons(
                layer_activations=layer_activations,
                pca_components=default_config["pca_components"],
                layer_order=layer_order
            )
        except Exception as e:
            logger.error(f"Error analyzing embedding comparisons: {e}")
            embedding_comparisons = {"error": str(e)}
        
        # Determine layers and attention heads analyzed
        layers_analyzed = []
        attention_heads_analyzed = {}
        
        # From layer activations
        for layer_name in layer_activations:
            if layer_name in layer_order and layer_name not in layers_analyzed:
                layers_analyzed.append(layer_name)
        
        # From attention data
        for layer_name, attention_probs in attention_data.items():
            if layer_name in layer_order:
                if layer_name not in layers_analyzed:
                    layers_analyzed.append(layer_name)
                
                # Determine number of attention heads
                if isinstance(attention_probs, torch.Tensor):
                    attention_probs = attention_probs.detach().cpu().numpy()
                
                if len(attention_probs.shape) == 4:  # [batch_size, n_heads, seq_len, seq_len]
                    n_heads = attention_probs.shape[1]
                elif len(attention_probs.shape) == 3:  # [n_heads, seq_len, seq_len]
                    n_heads = attention_probs.shape[0]
                else:
                    n_heads = 1
                
                attention_heads_analyzed[layer_name] = list(range(n_heads))
        
        # Prepare plot data for visualization
        plot_data = {}
        
        # Layer similarity matrix for heatmap
        if "similarity_matrix" in layer_relationships:
            similarity_data = []
            
            for (layer1, layer2), similarity in layer_relationships["similarity_matrix"].items():
                similarity_data.append({
                    "layer1": layer1,
                    "layer2": layer2,
                    "similarity": similarity
                })
            
            plot_data["layer_similarity"] = similarity_data
        
        # Attention flow direction for flow diagram
        if "attention_flow_direction" in attention_flow:
            flow_data = []
            
            for (layer1, layer2), direction in attention_flow["attention_flow_direction"].items():
                flow_data.append({
                    "source": layer1,
                    "target": layer2,
                    "direction": direction,
                    "entropy_diff": attention_flow["layer_entropy_differences"].get((layer1, layer2), 0.0),
                    "consistency": attention_flow.get("attention_pattern_consistency", {}).get((layer1, layer2), 0.0)
                })
            
            plot_data["attention_flow"] = flow_data
        
        # Token trajectories for trajectory visualization
        if "token_stability" in token_trajectory:
            trajectory_data = []
            
            for token_key, stability_values in token_trajectory["token_stability"].items():
                for (layer1, layer2), stability in stability_values.items():
                    shift = token_trajectory["semantic_shift"][token_key].get((layer1, layer2), 1.0 - stability)
                    
                    trajectory_data.append({
                        "token": token_key,
                        "source": layer1,
                        "target": layer2,
                        "stability": stability,
                        "shift": shift
                    })
            
            plot_data["token_trajectories"] = trajectory_data
        
        # Representation evolution for divergence visualization
        if "representation_divergence" in representation_evolution:
            divergence_data = []
            
            for (layer1, layer2), divergence in representation_evolution["representation_divergence"].items():
                divergence_data.append({
                    "source": layer1,
                    "target": layer2,
                    "divergence": divergence
                })
            
            plot_data["representation_divergence"] = divergence_data
        
        # Class separation for bar chart
        if "class_separation" in representation_evolution:
            separation_data = []
            
            for layer_name, separation in representation_evolution["class_separation"].items():
                separation_data.append({
                    "layer": layer_name,
                    "separation": separation
                })
            
            plot_data["class_separation"] = separation_data
        
        # Create result
        result = CrossLayerTransformerAnalysisResult(
            layer_relationships=layer_relationships,
            attention_flow=attention_flow,
            representation_evolution=representation_evolution,
            token_trajectory=token_trajectory,
            embedding_comparisons=embedding_comparisons,
            layers_analyzed=layers_analyzed,
            attention_heads_analyzed=attention_heads_analyzed,
            plot_data=plot_data
        )
        
        return result
    
    def clear_cache(self):
        """
        Clear the in-memory cache.
        """
        self._cache = {}
        
        # Also clear caches of child components
        self.metrics_calculator.clear_cache()
        
        # Note: We don't clear the dim_reducer's disk cache here
        # as that might affect other code using the same cache directory