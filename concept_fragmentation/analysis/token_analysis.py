"""
Token-level and sequence-level analysis for transformer models.

This module provides classes and functions for analyzing transformer models
at both the token level (per token activations) and sequence level (aggregated
activations across sequences). It extends the clustering and path tracking
capabilities to work with transformer-based models like GPT-2.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
import logging
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings

# Import the enhanced dimensionality reduction module
from .transformer_dimensionality import (
    TransformerDimensionalityReducer,
    DimensionalityReductionResult
)

from ..models.model_interfaces import ModelInterface, SequenceModelInterface, AttentionModelInterface

# Setup logger
logger = logging.getLogger(__name__)


class TokenLevelAnalysis:
    """
    Analyzes activations at the token level.
    
    This class provides methods for clustering and tracking token-level
    activations across layers in transformer models, enabling fine-grained
    analysis of how individual tokens are processed.
    
    Attributes:
        max_k: Maximum number of clusters to try
        random_state: Random seed for reproducibility
        use_cache: Whether to cache results
        dim_reducer: TransformerDimensionalityReducer for handling high-dimensional activations
    """
    
    def __init__(
        self,
        max_k: int = 10,
        random_state: int = 42,
        use_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize token-level analysis.
        
        Args:
            max_k: Maximum number of clusters to try
            random_state: Random seed for reproducibility
            use_cache: Whether to cache results
            cache_dir: Directory for caching results
        """
        self.max_k = max_k
        self.random_state = random_state
        self.use_cache = use_cache
        self._cache = {}
        
        # Initialize dimensionality reducer for transformer activations
        self.dim_reducer = TransformerDimensionalityReducer(
            cache_dir=cache_dir,
            random_state=random_state,
            use_cache=use_cache
        )
    
    def cluster_token_activations(
        self,
        activations: Union[torch.Tensor, np.ndarray],
        layer_name: str,
        method: str = "kmeans",
        n_clusters: Optional[int] = None,
        dimensionality_reduction: bool = False,
        n_components: int = 50,
        token_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_metadata: Optional[Dict[str, Any]] = None,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Cluster activations at the token level.
        
        Args:
            activations: Token activations [batch_size, seq_len, hidden_dim]
                         or [seq_len, hidden_dim] if batch_size = 1
            layer_name: Name of the layer
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Number of clusters (None for auto-selection)
            dimensionality_reduction: Whether to reduce dimensionality
            n_components: Number of components for dimensionality reduction
            token_ids: Optional token IDs for each position
            token_metadata: Optional metadata for tokens (e.g., token strings)
            force_recompute: Whether to force recomputation ignoring cache
            
        Returns:
            Dictionary with clustering results
        """
        # Check cache
        cache_key = f"cluster_{layer_name}_{method}"
        if self.use_cache and cache_key in self._cache and not force_recompute:
            return self._cache[cache_key]
        
        # Convert to numpy array for scikit-learn
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        
        # Handle different input shapes
        original_shape = activations.shape
        
        # If we have a batch dimension, reshape to [batch_size * seq_len, hidden_dim]
        if len(original_shape) == 3:  # [batch_size, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = original_shape
            activations = activations.reshape(-1, hidden_dim)
            reshaped = True
        elif len(original_shape) == 2:  # [seq_len, hidden_dim]
            seq_len, hidden_dim = original_shape
            batch_size = 1
            reshaped = False
        else:
            raise ValueError(f"Unexpected activation shape: {original_shape}")
        
        # Apply dimensionality reduction if requested
        if dimensionality_reduction and hidden_dim > n_components:
            logger.info(f"Reducing dimensionality from {hidden_dim} to {n_components}")
            try:
                # Use enhanced dimensionality reduction for transformer activations
                reduction_result = self.dim_reducer.reduce_dimensionality(
                    activations=activations,
                    n_components=n_components,
                    method="auto",
                    layer_name=layer_name
                )
                
                if reduction_result.success:
                    activations = reduction_result.reduced_activations
                    explained_variance = reduction_result.explained_variance
                    
                    if explained_variance is not None:
                        logger.info(f"{reduction_result.method} explained variance: {explained_variance:.4f}")
                    
                    logger.info(f"Dimensionality reduced from {hidden_dim} to {n_components} using {reduction_result.method}")
                else:
                    logger.warning(f"Dimensionality reduction failed: {reduction_result.error_message}, proceeding with original dimensions")
            except Exception as e:
                logger.warning(f"Dimensionality reduction failed: {e}, proceeding with original dimensions")
        
        # Determine number of clusters if not specified
        if n_clusters is None and method == "kmeans":
            logger.info(f"Determining optimal number of clusters for {layer_name}")
            n_clusters, _, _ = self._find_optimal_clusters(activations)
        elif n_clusters is None:
            n_clusters = min(5, max(2, seq_len // 10))  # Reasonable default
        
        # Apply clustering
        logger.info(f"Clustering tokens for {layer_name} with {n_clusters} clusters")
        
        if method == "kmeans":
            cluster_labels, centers = self._apply_kmeans(activations, n_clusters)
        elif method == "dbscan":
            cluster_labels, centers = self._apply_dbscan(activations)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        # Compute silhouette score if multiple clusters
        silhouette = -1.0
        if len(set(cluster_labels)) > 1 and min(np.bincount(cluster_labels)) > 1:
            try:
                silhouette = silhouette_score(activations, cluster_labels)
            except Exception as e:
                logger.warning(f"Error computing silhouette score: {e}")
        
        # Prepare token information
        token_info = []
        
        # If we have token metadata, add it to results
        if token_ids is not None:
            # Convert to numpy if needed
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.detach().cpu().numpy()
            
            # Handle token_ids with batch dimension
            if len(token_ids.shape) == 2:  # [batch_size, seq_len]
                token_ids = token_ids.reshape(-1)
            
            # Create token info entries
            for i, (token_id, cluster_id) in enumerate(zip(token_ids, cluster_labels)):
                batch_idx = i // seq_len if reshaped else 0
                seq_idx = i % seq_len if reshaped else i
                
                token_entry = {
                    "token_id": int(token_id),
                    "cluster_id": int(cluster_id),
                    "batch_idx": batch_idx,
                    "seq_idx": seq_idx
                }
                
                # Add token string if available
                if token_metadata and "token_strings" in token_metadata:
                    token_str_idx = batch_idx * seq_len + seq_idx
                    if token_str_idx < len(token_metadata["token_strings"]):
                        token_entry["token_string"] = token_metadata["token_strings"][token_str_idx]
                
                token_info.append(token_entry)
        
        # Collect results
        results = {
            "layer_name": layer_name,
            "n_clusters": n_clusters,
            "cluster_labels": cluster_labels,
            "cluster_centers": centers,
            "activations_shape": original_shape,
            "silhouette_score": silhouette,
            "method": method,
            "token_info": token_info,
            "dimensionality_reduced": dimensionality_reduction,
            "n_components": n_components if dimensionality_reduction else hidden_dim
        }
        
        # Cache results
        if self.use_cache:
            self._cache[cache_key] = results
        
        return results
    
    def track_token_paths(
        self,
        layer_activations: Dict[str, torch.Tensor],
        token_ids: Optional[torch.Tensor] = None,
        layer_order: Optional[List[str]] = None,
        cluster_results: Optional[Dict[str, Dict[str, Any]]] = None,
        token_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Track how tokens move through activation space across layers.
        
        Args:
            layer_activations: Dictionary mapping layer names to activations
                             [batch_size, seq_len, hidden_dim]
            token_ids: Token IDs for each position [batch_size, seq_len]
            layer_order: Order of layers for path tracking (None for auto)
            cluster_results: Optional pre-computed clustering results
            token_metadata: Optional metadata for tokens (e.g., token strings)
            
        Returns:
            Dictionary with token path information
        """
        # Determine layer order if not provided
        if layer_order is None:
            # Try to infer from layer names (e.g., transformer_layer_0, transformer_layer_1)
            layers = list(layer_activations.keys())
            
            # Sort by layer index if pattern is consistent
            try:
                layer_order = sorted(
                    layers,
                    key=lambda name: int(name.split('_')[-1]) if name.split('_')[-1].isdigit() else 0
                )
            except (IndexError, ValueError):
                # Fall back to alphabetical order
                layer_order = sorted(layers)
        
        # Cluster each layer if not pre-computed
        if cluster_results is None:
            cluster_results = {}
            for layer_name in layer_order:
                activations = layer_activations[layer_name]
                
                cluster_results[layer_name] = self.cluster_token_activations(
                    activations=activations,
                    layer_name=layer_name,
                    token_ids=token_ids,
                    token_metadata=token_metadata
                )
        
        # Initialize path tracking data
        batch_size = None
        seq_len = None
        
        # Determine dimensions
        for layer_name in layer_order:
            if layer_name in layer_activations:
                activations = layer_activations[layer_name]
                if isinstance(activations, torch.Tensor):
                    batch_size = activations.shape[0]
                    seq_len = activations.shape[1]
                    break
                elif isinstance(activations, np.ndarray):
                    batch_size = activations.shape[0]
                    seq_len = activations.shape[1]
                    break
        
        if batch_size is None or seq_len is None:
            raise ValueError("Could not determine batch size and sequence length")
        
        # Initialize token paths
        token_paths = np.zeros((batch_size, seq_len, len(layer_order)), dtype=np.int32)
        
        # Fill in token paths
        for layer_idx, layer_name in enumerate(layer_order):
            if layer_name not in cluster_results:
                logger.warning(f"Missing clustering results for layer {layer_name}")
                continue
                
            cluster_result = cluster_results[layer_name]
            cluster_labels = cluster_result["cluster_labels"]
            
            # Reshape cluster labels if needed
            if len(cluster_labels) == batch_size * seq_len:
                labels_reshaped = cluster_labels.reshape(batch_size, seq_len)
            else:
                # If we can't reshape, skip this layer
                logger.warning(f"Cluster labels shape mismatch for layer {layer_name}")
                continue
                
            # Store in token paths
            token_paths[:, :, layer_idx] = labels_reshaped
        
        # For each token, create a human-readable path
        human_readable_paths = []
        
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                token_path = token_paths[batch_idx, seq_idx]
                
                # Create path string (e.g., "L0C2->L1C3->L2C1")
                path_parts = []
                
                for layer_idx, cluster_id in enumerate(token_path):
                    path_parts.append(f"L{layer_idx}C{cluster_id}")
                
                path_str = "->".join(path_parts)
                
                token_entry = {
                    "batch_idx": batch_idx,
                    "seq_idx": seq_idx,
                    "path": path_str,
                    "cluster_ids": token_path.tolist()
                }
                
                # Add token information if available
                if token_ids is not None:
                    token_entry["token_id"] = int(token_ids[batch_idx, seq_idx])
                
                if token_metadata and "token_strings" in token_metadata:
                    token_str_idx = batch_idx * seq_len + seq_idx
                    if token_str_idx < len(token_metadata["token_strings"]):
                        token_entry["token_string"] = token_metadata["token_strings"][token_str_idx]
                
                human_readable_paths.append(token_entry)
        
        # Collect results
        results = {
            "layer_order": layer_order,
            "token_paths": token_paths,
            "human_readable_paths": human_readable_paths,
            "cluster_results": cluster_results
        }
        
        return results
    
    def _find_optimal_clusters(
        self,
        activations: np.ndarray
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            activations: Activation values [n_samples, n_features]
            
        Returns:
            Tuple of (optimal_k, cluster_centers, cluster_labels)
        """
        # Logic from compute_clusters_for_layer in cluster_paths.py
        best_k = 2
        best_score = -1.0
        best_labels = None
        best_centers = None
        
        # Try different k values
        for k in range(2, min(self.max_k, activations.shape[0]//2) + 1):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = kmeans.fit_predict(activations)
            
            # Skip if we have clusters with only one point (silhouette undefined)
            # or if only one cluster is formed
            if len(set(labels)) < 2 or np.min(np.bincount(labels)) < 2:
                continue
                
            try:
                score = silhouette_score(activations, labels)
                if score > best_score:
                    best_k = k
                    best_score = score
                    best_labels = labels
                    best_centers = kmeans.cluster_centers_
            except Exception as e:
                warnings.warn(f"Error computing silhouette for k={k}: {str(e)}")
                continue
        
        # Fall back to k=2 if no valid clustering found
        if best_centers is None:
            # Attempt to force k=2 if possible
            if activations.shape[0] >= 4:
                num_clusters_fallback = 2
            elif activations.shape[0] >= 2:
                num_clusters_fallback = 1
            else:
                return 0, np.array([]), np.array([])

            if num_clusters_fallback == 1 and activations.shape[0] > 0:
                best_labels = np.zeros(activations.shape[0], dtype=int)
                best_centers = np.mean(activations, axis=0, keepdims=True)
                best_k = 1
            elif num_clusters_fallback == 2:
                kmeans = KMeans(n_clusters=2, n_init=10, random_state=self.random_state)
                try:
                    best_labels = kmeans.fit_predict(activations)
                    # Check if k-means actually produced 2 clusters
                    if len(np.unique(best_labels)) < 2:
                        # If k-means collapsed to 1 cluster, set labels to 0s and center to mean
                        best_labels = np.zeros(activations.shape[0], dtype=int)
                        best_centers = np.mean(activations, axis=0, keepdims=True)
                        best_k = 1
                    else:
                        best_centers = kmeans.cluster_centers_
                        best_k = 2
                except Exception as e:
                    warnings.warn(f"Error during fallback KMeans (k=2): {e}")
                    return 0, np.array([]), np.array([])
            else:
                return 0, np.array([]), np.array([])
        
        return best_k, best_centers, best_labels
    
    def _apply_kmeans(
        self,
        activations: np.ndarray,
        n_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply K-means clustering.
        
        Args:
            activations: Activation values [n_samples, n_features]
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (cluster_labels, cluster_centers)
        """
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_state)
        labels = kmeans.fit_predict(activations)
        centers = kmeans.cluster_centers_
        
        return labels, centers
    
    def _apply_dbscan(
        self,
        activations: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply DBSCAN clustering.
        
        Args:
            activations: Activation values [n_samples, n_features]
            eps: The maximum distance between samples
            min_samples: The minimum number of samples in a neighborhood
            
        Returns:
            Tuple of (cluster_labels, cluster_centers)
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(activations)
        
        # Handle noise points (-1 label) by assigning them to a new cluster
        if -1 in labels:
            labels = labels.copy()  # Make a copy to avoid modifying the original
            labels[labels == -1] = labels.max() + 1
        
        # Compute cluster centers
        n_clusters = len(set(labels))
        centers = np.zeros((n_clusters, activations.shape[1]))
        
        for i in range(n_clusters):
            if np.sum(labels == i) > 0:
                centers[i] = np.mean(activations[labels == i], axis=0)
        
        return labels, centers


class SequenceLevelAnalysis:
    """
    Analyzes activations at the sequence level.
    
    This class provides methods for aggregating token-level activations
    to sequence-level representations, enabling analysis of entire
    sequences rather than individual tokens.
    
    Attributes:
        token_analyzer: Optional TokenLevelAnalysis instance
        max_k: Maximum number of clusters to try
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        token_analyzer: Optional[TokenLevelAnalysis] = None,
        max_k: int = 10,
        random_state: int = 42
    ):
        """
        Initialize sequence-level analysis.
        
        Args:
            token_analyzer: TokenLevelAnalysis instance for token-level analysis
            max_k: Maximum number of clusters to try
            random_state: Random seed for reproducibility
        """
        self.token_analyzer = token_analyzer or TokenLevelAnalysis(
            max_k=max_k,
            random_state=random_state
        )
        self.max_k = max_k
        self.random_state = random_state
    
    def aggregate_token_activations(
        self,
        token_activations: Union[torch.Tensor, np.ndarray],
        method: str = "mean",
        attention_weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Aggregate token-level activations to sequence level.
        
        Args:
            token_activations: Token activations [batch_size, seq_len, hidden_dim]
            method: Aggregation method ('mean', 'max', 'attention_pooling', 'cls', 'final')
            attention_weights: Optional attention weights [batch_size, seq_len]
            token_mask: Optional mask for padding tokens [batch_size, seq_len]
            
        Returns:
            Sequence-level activations [batch_size, hidden_dim]
        """
        is_torch = isinstance(token_activations, torch.Tensor)
        
        # Convert numpy to torch if needed for easier processing
        if not is_torch:
            token_activations = torch.tensor(token_activations)
            if attention_weights is not None:
                attention_weights = torch.tensor(attention_weights)
            if token_mask is not None:
                token_mask = torch.tensor(token_mask)
        
        # Apply token mask if provided
        if token_mask is not None:
            # Expand mask to broadcast over hidden dimension
            mask = token_mask.unsqueeze(-1)
            # Apply mask (replace masked tokens with zeros)
            token_activations = token_activations * mask
        
        # Apply aggregation method
        if method == "mean":
            if token_mask is not None:
                # Compute mean only over unmasked tokens
                # Sum over sequence dimension and divide by number of unmasked tokens
                seq_representations = token_activations.sum(dim=1) / token_mask.sum(dim=1, keepdim=True)
            else:
                # Simple mean over sequence dimension
                seq_representations = token_activations.mean(dim=1)
                
        elif method == "max":
            # Max pooling over sequence dimension
            seq_representations, _ = token_activations.max(dim=1)
            
        elif method == "attention_pooling":
            if attention_weights is None:
                raise ValueError("Attention weights required for attention_pooling method")
            
            # Ensure attention weights sum to 1
            attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True)
            
            # Expand attention weights to broadcast over hidden dimension
            weights = attention_weights.unsqueeze(-1)
            
            # Apply attention weights and sum
            seq_representations = (token_activations * weights).sum(dim=1)
            
        elif method == "cls":
            # Use first token representation (like BERT's [CLS] token)
            seq_representations = token_activations[:, 0, :]
            
        elif method == "final":
            # Use final token representation
            if token_mask is not None:
                # Get the last non-masked position for each sequence
                seq_lengths = token_mask.sum(dim=1)
                batch_indices = torch.arange(token_activations.size(0))
                last_indices = seq_lengths - 1
                seq_representations = token_activations[batch_indices, last_indices]
            else:
                # Use the last token
                seq_representations = token_activations[:, -1, :]
                
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
        
        # Convert back to numpy if input was numpy
        if not is_torch:
            seq_representations = seq_representations.numpy()
        
        return seq_representations
    
    def cluster_sequences(
        self,
        layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
        method: str = "kmeans",
        aggregation_method: str = "mean",
        n_clusters: Optional[int] = None,
        attention_weights: Optional[Dict[str, Union[torch.Tensor, np.ndarray]]] = None,
        token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        sequence_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Cluster sequences based on aggregated token activations.
        
        Args:
            layer_activations: Dictionary mapping layer names to token activations
                             [batch_size, seq_len, hidden_dim]
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            aggregation_method: Method to aggregate tokens ('mean', 'max', 'attention_pooling')
            n_clusters: Number of clusters (None for auto-selection)
            attention_weights: Optional dictionary of attention weights per layer
            token_mask: Optional mask for padding tokens
            sequence_metadata: Optional metadata for sequences
            
        Returns:
            Dictionary mapping layer names to clustering results
        """
        sequence_clusters = {}
        
        # Process each layer
        for layer_name, token_activations in layer_activations.items():
            # Get attention weights for this layer if available
            layer_attention = None
            if attention_weights and layer_name in attention_weights:
                layer_attention = attention_weights[layer_name]
            
            # Aggregate tokens to sequence level
            seq_representations = self.aggregate_token_activations(
                token_activations=token_activations,
                method=aggregation_method,
                attention_weights=layer_attention,
                token_mask=token_mask
            )
            
            # Apply clustering
            layer_clusters = self.token_analyzer.cluster_token_activations(
                activations=seq_representations,
                layer_name=f"{layer_name}_seq",
                method=method,
                n_clusters=n_clusters
            )
            
            # Add sequence metadata if available
            if sequence_metadata:
                layer_clusters["sequence_metadata"] = sequence_metadata
            
            sequence_clusters[layer_name] = layer_clusters
        
        return sequence_clusters
    
    def track_sequence_paths(
        self,
        layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
        layer_order: Optional[List[str]] = None,
        cluster_results: Optional[Dict[str, Dict[str, Any]]] = None,
        aggregation_method: str = "mean",
        attention_weights: Optional[Dict[str, Union[torch.Tensor, np.ndarray]]] = None,
        token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        sequence_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Track how sequences move through activation space across layers.
        
        Args:
            layer_activations: Dictionary mapping layer names to token activations
            layer_order: Order of layers for path tracking (None for auto)
            cluster_results: Optional pre-computed clustering results
            aggregation_method: Method to aggregate tokens
            attention_weights: Optional dictionary of attention weights per layer
            token_mask: Optional mask for padding tokens
            sequence_metadata: Optional metadata for sequences
            
        Returns:
            Dictionary with sequence path information
        """
        # Determine layer order if not provided
        if layer_order is None:
            # Try to infer from layer names (e.g., transformer_layer_0, transformer_layer_1)
            layers = list(layer_activations.keys())
            
            # Sort by layer index if pattern is consistent
            try:
                layer_order = sorted(
                    layers,
                    key=lambda name: int(name.split('_')[-1]) if name.split('_')[-1].isdigit() else 0
                )
            except (IndexError, ValueError):
                # Fall back to alphabetical order
                layer_order = sorted(layers)
        
        # Cluster sequences for each layer if not pre-computed
        if cluster_results is None:
            cluster_results = self.cluster_sequences(
                layer_activations=layer_activations,
                aggregation_method=aggregation_method,
                attention_weights=attention_weights,
                token_mask=token_mask,
                sequence_metadata=sequence_metadata
            )
        
        # Determine batch size
        batch_size = None
        for layer_name in layer_order:
            if layer_name in layer_activations:
                activations = layer_activations[layer_name]
                if isinstance(activations, torch.Tensor):
                    batch_size = activations.shape[0]
                    break
                elif isinstance(activations, np.ndarray):
                    batch_size = activations.shape[0]
                    break
        
        if batch_size is None:
            raise ValueError("Could not determine batch size")
        
        # Initialize sequence paths
        sequence_paths = np.zeros((batch_size, len(layer_order)), dtype=np.int32)
        
        # Fill in sequence paths
        for layer_idx, layer_name in enumerate(layer_order):
            if layer_name not in cluster_results:
                logger.warning(f"Missing clustering results for layer {layer_name}")
                continue
                
            cluster_result = cluster_results[layer_name]
            cluster_labels = cluster_result["cluster_labels"]
            
            # Ensure the right shape for batch processing
            if len(cluster_labels) != batch_size:
                # If shapes don't match, skip this layer
                logger.warning(f"Cluster labels shape mismatch for layer {layer_name}")
                continue
                
            # Store in sequence paths
            sequence_paths[:, layer_idx] = cluster_labels
        
        # Create human-readable paths
        human_readable_paths = []
        
        for batch_idx in range(batch_size):
            sequence_path = sequence_paths[batch_idx]
            
            # Create path string (e.g., "L0C2->L1C3->L2C1")
            path_parts = []
            
            for layer_idx, cluster_id in enumerate(sequence_path):
                path_parts.append(f"L{layer_idx}C{cluster_id}")
            
            path_str = "->".join(path_parts)
            
            sequence_entry = {
                "batch_idx": batch_idx,
                "path": path_str,
                "cluster_ids": sequence_path.tolist()
            }
            
            # Add sequence metadata if available
            if sequence_metadata and "sequence_info" in sequence_metadata:
                if batch_idx < len(sequence_metadata["sequence_info"]):
                    sequence_entry.update(sequence_metadata["sequence_info"][batch_idx])
            
            human_readable_paths.append(sequence_entry)
        
        # Collect results
        results = {
            "layer_order": layer_order,
            "sequence_paths": sequence_paths,
            "human_readable_paths": human_readable_paths,
            "cluster_results": cluster_results
        }
        
        return results


class TokenClusteringStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """
    Pipeline stage for token-level clustering.
    
    This stage applies clustering to token-level activations 
    from transformer models, extending the standard clustering
    stage to handle token representations.
    """
    
    def __init__(
        self,
        analyzer: Optional[TokenLevelAnalysis] = None,
        filter_layers: Optional[str] = None,
        max_k: int = 10,
        use_dimensionality_reduction: bool = True,
        n_components: int = 50,
        reduction_method: str = "auto",
        use_progressive_reduction: bool = True,
        cache_dir: Optional[str] = None,
        name: str = "TokenClustering"
    ):
        """
        Initialize token clustering stage.
        
        Args:
            analyzer: Optional TokenLevelAnalysis instance
            filter_layers: Optional regex to filter layer names
            max_k: Maximum number of clusters to try
            use_dimensionality_reduction: Whether to reduce dimensionality
            n_components: Number of components for dimensionality reduction
            reduction_method: Method to use for dimensionality reduction
            use_progressive_reduction: Whether to use progressive reduction
            cache_dir: Directory for caching results
            name: Name for this stage
        """
        super().__init__(name=name)
        self.analyzer = analyzer or TokenLevelAnalysis(max_k=max_k, cache_dir=cache_dir)
        self.filter_layers = filter_layers
        self.use_dimensionality_reduction = use_dimensionality_reduction
        self.n_components = n_components
        self.reduction_method = reduction_method
        self.use_progressive_reduction = use_progressive_reduction
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process token activations through clustering.
        
        The input should be a dictionary with:
        - 'activations': Dictionary mapping layer names to activation tensors
                        [batch_size, seq_len, hidden_dim]
        - 'token_ids': Optional token IDs [batch_size, seq_len]
        - 'token_metadata': Optional metadata for tokens
        
        Returns:
            Dictionary with the input data and added token clusters
        """
        result = data.copy()
        
        # Check for required inputs
        if 'activations' not in data:
            raise ValueError("Input must contain 'activations' key")
        
        activations = data['activations']
        token_ids = data.get('token_ids')
        token_metadata = data.get('token_metadata')
        
        # Filter layers if pattern provided
        if self.filter_layers:
            import re
            pattern = re.compile(self.filter_layers)
            filtered_activations = {
                name: tensor for name, tensor in activations.items()
                if pattern.search(name)
            }
        else:
            filtered_activations = activations
        
        # Process each layer
        token_clusters = {}
        
        for layer_name, layer_activations in filtered_activations.items():
            # Skip non-tensor activations or metadata
            if not isinstance(layer_activations, (torch.Tensor, np.ndarray)):
                continue
                
            # Check tensor shape (must be [batch_size, seq_len, hidden_dim])
            if len(layer_activations.shape) != 3:
                logger.warning(f"Skipping layer {layer_name} with shape {layer_activations.shape}")
                continue
            
            # Apply clustering
            cluster_result = self.analyzer.cluster_token_activations(
                activations=layer_activations,
                layer_name=layer_name,
                token_ids=token_ids,
                token_metadata=token_metadata,
                dimensionality_reduction=self.use_dimensionality_reduction,
                n_components=self.n_components
            )
            
            token_clusters[layer_name] = cluster_result
        
        # Add results to output
        result['token_clusters'] = token_clusters
        
        return result


class SequenceClusteringStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """
    Pipeline stage for sequence-level clustering.
    
    This stage applies clustering to sequence-level representations
    derived from token-level activations.
    """
    
    def __init__(
        self,
        analyzer: Optional[SequenceLevelAnalysis] = None,
        filter_layers: Optional[str] = None,
        aggregation_method: str = "mean",
        max_k: int = 10,
        name: str = "SequenceClustering"
    ):
        """
        Initialize sequence clustering stage.
        
        Args:
            analyzer: Optional SequenceLevelAnalysis instance
            filter_layers: Optional regex to filter layer names
            aggregation_method: Method to aggregate tokens
            max_k: Maximum number of clusters to try
            name: Name for this stage
        """
        super().__init__(name=name)
        self.analyzer = analyzer or SequenceLevelAnalysis(max_k=max_k)
        self.filter_layers = filter_layers
        self.aggregation_method = aggregation_method
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process token activations through sequence clustering.
        
        The input should be a dictionary with:
        - 'activations': Dictionary mapping layer names to activation tensors
                        [batch_size, seq_len, hidden_dim]
        - 'attention_weights': Optional dictionary of attention weights
        - 'token_mask': Optional mask for padding tokens
        - 'sequence_metadata': Optional metadata for sequences
        
        Returns:
            Dictionary with the input data and added sequence clusters
        """
        result = data.copy()
        
        # Check for required inputs
        if 'activations' not in data:
            raise ValueError("Input must contain 'activations' key")
        
        activations = data['activations']
        attention_weights = data.get('attention_weights')
        token_mask = data.get('token_mask')
        sequence_metadata = data.get('sequence_metadata')
        
        # Filter layers if pattern provided
        if self.filter_layers:
            import re
            pattern = re.compile(self.filter_layers)
            filtered_activations = {
                name: tensor for name, tensor in activations.items()
                if pattern.search(name)
            }
        else:
            filtered_activations = activations
        
        # Filter attention weights if available
        filtered_attention = None
        if attention_weights and self.filter_layers:
            filtered_attention = {
                name: weights for name, weights in attention_weights.items()
                if pattern.search(name)
            }
        else:
            filtered_attention = attention_weights
        
        # Apply sequence clustering
        sequence_clusters = self.analyzer.cluster_sequences(
            layer_activations=filtered_activations,
            aggregation_method=self.aggregation_method,
            attention_weights=filtered_attention,
            token_mask=token_mask,
            sequence_metadata=sequence_metadata
        )
        
        # Add results to output
        result['sequence_clusters'] = sequence_clusters
        
        return result


# Import here to avoid circular imports
from ..pipeline.pipeline import PipelineStageBase