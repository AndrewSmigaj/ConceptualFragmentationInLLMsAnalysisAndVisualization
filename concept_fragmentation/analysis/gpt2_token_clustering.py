"""
Token-level clustering for GPT-2 activations.

This module provides specialized clustering approaches for token-level
representations in GPT-2 models, allowing for tracking concept flow
at the token level across model layers.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import logging
from collections import Counter, defaultdict
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dataclasses import dataclass

# Import dimensionality reduction
from concept_fragmentation.analysis.gpt2_dimensionality import GPT2DimensionalityReducer

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class TokenClusteringResult:
    """
    Results from token-level clustering.
    
    Attributes:
        token_clusters: Dictionary mapping token indices to cluster assignments
        combined_clusters: Combined clustering across batch and sequence dimensions
        cluster_centers: Cluster centroids in the embedding space
        position_clusters: Dictionary mapping positions to cluster assignments
        metadata: Additional metadata about the clustering
        success: Whether the clustering was successful
        error: Error message if clustering failed
    """
    token_clusters: Dict[int, np.ndarray]
    combined_clusters: np.ndarray
    cluster_centers: np.ndarray
    position_clusters: Optional[Dict[int, np.ndarray]] = None
    metadata: Dict[str, Any] = None
    success: bool = True
    error: Optional[str] = None


class GPT2TokenClusterer:
    """
    Specialized clusterer for token-level representations in GPT-2.
    
    This class provides methods for clustering token representations across
    layers in GPT-2 models, tracking token identity, and analyzing how
    concepts evolve at the token level.
    
    Attributes:
        n_clusters: Number of clusters for clustering
        dim_reducer: Dimensionality reducer for preprocessing
        random_state: Random seed for reproducibility
        token_aware: Whether to include token identity in clustering
        use_cache: Whether to use caching
        _cache: In-memory cache for clustering results
    """
    
    def __init__(
        self,
        n_clusters: int = 8,
        dim_reducer: Optional[GPT2DimensionalityReducer] = None,
        random_state: int = 42,
        token_aware: bool = True,
        use_cache: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the token clusterer.
        
        Args:
            n_clusters: Number of clusters for clustering
            dim_reducer: Dimensionality reducer for preprocessing
            random_state: Random seed for reproducibility
            token_aware: Whether to include token identity in clustering
            use_cache: Whether to use caching
            verbose: Whether to log verbose information
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.token_aware = token_aware
        self.use_cache = use_cache
        self.verbose = verbose
        
        # Create default dimensionality reducer if not provided
        if dim_reducer is None:
            self.dim_reducer = GPT2DimensionalityReducer(
                random_state=random_state,
                use_cache=use_cache,
                verbose=verbose
            )
        else:
            self.dim_reducer = dim_reducer
        
        # Initialize cache
        self._cache = {}
    
    def cluster_token_activations(
        self,
        activations: Union[torch.Tensor, np.ndarray],
        token_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        method: str = "combined",
        dimensionality_reduction: bool = True,
        n_components: int = 50,
        layer_name: Optional[str] = None
    ) -> TokenClusteringResult:
        """
        Cluster token-level activations.
        
        Args:
            activations: Token activations [batch_size, seq_len, hidden_dim]
            token_ids: Optional token IDs for token-aware clustering [batch_size, seq_len]
            token_mask: Optional mask for padding tokens [batch_size, seq_len]
            method: Clustering method ('combined', 'per_position', 'per_token_type')
            dimensionality_reduction: Whether to apply dimensionality reduction
            n_components: Number of components for dimensionality reduction
            layer_name: Optional layer name for context
            
        Returns:
            TokenClusteringResult with clustering results
        """
        # Handle inputs
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().numpy()
        
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.detach().cpu().numpy()
        
        # Validate activations shape
        if len(activations.shape) != 3:
            error = f"Expected 3D activations [batch_size, seq_len, hidden_dim], got {activations.shape}"
            logger.error(error)
            return TokenClusteringResult(
                token_clusters={},
                combined_clusters=np.array([]),
                cluster_centers=np.array([]),
                success=False,
                error=error
            )
        
        # Get dimensions
        batch_size, seq_len, hidden_dim = activations.shape
        
        # Create default token mask if not provided
        if token_mask is None:
            token_mask = np.ones((batch_size, seq_len), dtype=bool)
        
        # Select clustering method
        if method == "combined":
            return self._cluster_combined(
                activations=activations,
                token_ids=token_ids,
                token_mask=token_mask,
                dimensionality_reduction=dimensionality_reduction,
                n_components=n_components,
                layer_name=layer_name
            )
        elif method == "per_position":
            return self._cluster_per_position(
                activations=activations,
                token_ids=token_ids,
                token_mask=token_mask,
                dimensionality_reduction=dimensionality_reduction,
                n_components=n_components,
                layer_name=layer_name
            )
        elif method == "per_token_type":
            return self._cluster_per_token_type(
                activations=activations,
                token_ids=token_ids,
                token_mask=token_mask,
                dimensionality_reduction=dimensionality_reduction,
                n_components=n_components,
                layer_name=layer_name
            )
        else:
            error = f"Unknown clustering method: {method}"
            logger.error(error)
            return TokenClusteringResult(
                token_clusters={},
                combined_clusters=np.array([]),
                cluster_centers=np.array([]),
                success=False,
                error=error
            )
    
    def _cluster_combined(
        self,
        activations: np.ndarray,
        token_ids: Optional[np.ndarray],
        token_mask: np.ndarray,
        dimensionality_reduction: bool,
        n_components: int,
        layer_name: Optional[str]
    ) -> TokenClusteringResult:
        """
        Cluster all token activations together.
        
        Args:
            activations: Token activations [batch_size, seq_len, hidden_dim]
            token_ids: Optional token IDs [batch_size, seq_len]
            token_mask: Mask for padding tokens [batch_size, seq_len]
            dimensionality_reduction: Whether to apply dimensionality reduction
            n_components: Number of components for dimensionality reduction
            layer_name: Optional layer name for context
            
        Returns:
            TokenClusteringResult with combined clustering
        """
        batch_size, seq_len, hidden_dim = activations.shape
        
        # Flatten activations to [batch_size * seq_len, hidden_dim]
        flat_activations = activations.reshape(-1, hidden_dim)
        
        # Flatten mask to [batch_size * seq_len]
        flat_mask = token_mask.reshape(-1)
        
        # Only use valid (non-padding) tokens
        valid_activations = flat_activations[flat_mask]
        
        # Apply dimensionality reduction if needed
        if dimensionality_reduction and valid_activations.shape[1] > n_components:
            logger.info(f"Reducing dimensionality from {valid_activations.shape[1]} to {n_components}")
            
            result = self.dim_reducer.reduce_dimensionality(
                activations=valid_activations,
                n_components=n_components,
                method="auto",
                layer_name=layer_name
            )
            
            if result.success:
                valid_activations = result.reduced_activations
            else:
                logger.warning(f"Dimensionality reduction failed: {result.error_message}")
        
        # Cluster the activations
        logger.info(f"Clustering {valid_activations.shape[0]} token activations into {self.n_clusters} clusters")
        
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        valid_clusters = kmeans.fit_predict(valid_activations)
        
        # Generate complete cluster assignment
        combined_clusters = np.zeros((batch_size * seq_len), dtype=int)
        combined_clusters[flat_mask] = valid_clusters
        
        # Reshape to [batch_size, seq_len]
        combined_clusters = combined_clusters.reshape(batch_size, seq_len)
        
        # Create token-level clusters
        token_clusters = {}
        
        for tok_idx in range(seq_len):
            # Extract clusters for this token position across all batches
            token_clusters[tok_idx] = combined_clusters[:, tok_idx]
        
        # Return result
        return TokenClusteringResult(
            token_clusters=token_clusters,
            combined_clusters=combined_clusters,
            cluster_centers=kmeans.cluster_centers_,
            metadata={
                "method": "combined",
                "n_clusters": self.n_clusters,
                "n_valid_tokens": valid_activations.shape[0],
                "dimensionality_reduction": dimensionality_reduction,
                "n_components": n_components if dimensionality_reduction else hidden_dim
            }
        )
    
    def _cluster_per_position(
        self,
        activations: np.ndarray,
        token_ids: Optional[np.ndarray],
        token_mask: np.ndarray,
        dimensionality_reduction: bool,
        n_components: int,
        layer_name: Optional[str]
    ) -> TokenClusteringResult:
        """
        Cluster tokens separately for each position.
        
        Args:
            activations: Token activations [batch_size, seq_len, hidden_dim]
            token_ids: Optional token IDs [batch_size, seq_len]
            token_mask: Mask for padding tokens [batch_size, seq_len]
            dimensionality_reduction: Whether to apply dimensionality reduction
            n_components: Number of components for dimensionality reduction
            layer_name: Optional layer name for context
            
        Returns:
            TokenClusteringResult with per-position clustering
        """
        batch_size, seq_len, hidden_dim = activations.shape
        
        # Initialize results
        token_clusters = {}
        position_clusters = {}
        all_centers = []
        
        # Process each position separately
        for pos in range(seq_len):
            # Extract activations for this position
            pos_activations = activations[:, pos, :]
            
            # Extract mask for this position
            pos_mask = token_mask[:, pos]
            
            # Only use valid tokens
            valid_activations = pos_activations[pos_mask]
            
            # Skip if no valid tokens
            if valid_activations.shape[0] == 0:
                token_clusters[pos] = np.zeros(batch_size, dtype=int)
                continue
            
            # Apply dimensionality reduction if needed
            if dimensionality_reduction and valid_activations.shape[1] > n_components:
                result = self.dim_reducer.reduce_dimensionality(
                    activations=valid_activations,
                    n_components=n_components,
                    method="auto",
                    layer_name=f"{layer_name}_pos{pos}" if layer_name else f"pos{pos}"
                )
                
                if result.success:
                    valid_activations = result.reduced_activations
            
            # Adjust number of clusters if needed
            actual_clusters = min(self.n_clusters, valid_activations.shape[0])
            
            if actual_clusters < self.n_clusters:
                logger.warning(f"Reduced clusters from {self.n_clusters} to {actual_clusters} for position {pos}")
            
            # Cluster the activations
            kmeans = KMeans(
                n_clusters=actual_clusters,
                random_state=self.random_state,
                n_init=10
            )
            
            valid_clusters = kmeans.fit_predict(valid_activations)
            
            # Generate complete cluster assignment
            pos_clusters = np.zeros(batch_size, dtype=int)
            pos_clusters[pos_mask] = valid_clusters
            
            # Store results
            token_clusters[pos] = pos_clusters
            position_clusters[pos] = valid_clusters
            all_centers.append(kmeans.cluster_centers_)
        
        # Combine results into a [batch_size, seq_len] array
        combined_clusters = np.zeros((batch_size, seq_len), dtype=int)
        
        for pos, clusters in token_clusters.items():
            combined_clusters[:, pos] = clusters
        
        # Combine all cluster centers
        all_centers_array = np.vstack(all_centers) if all_centers else np.array([])
        
        # Return result
        return TokenClusteringResult(
            token_clusters=token_clusters,
            combined_clusters=combined_clusters,
            cluster_centers=all_centers_array,
            position_clusters=position_clusters,
            metadata={
                "method": "per_position",
                "n_clusters": self.n_clusters,
                "positions_processed": len(token_clusters),
                "dimensionality_reduction": dimensionality_reduction,
                "n_components": n_components if dimensionality_reduction else hidden_dim
            }
        )
    
    def _cluster_per_token_type(
        self,
        activations: np.ndarray,
        token_ids: Optional[np.ndarray],
        token_mask: np.ndarray,
        dimensionality_reduction: bool,
        n_components: int,
        layer_name: Optional[str]
    ) -> TokenClusteringResult:
        """
        Cluster tokens based on token type.
        
        Args:
            activations: Token activations [batch_size, seq_len, hidden_dim]
            token_ids: Token IDs [batch_size, seq_len]
            token_mask: Mask for padding tokens [batch_size, seq_len]
            dimensionality_reduction: Whether to apply dimensionality reduction
            n_components: Number of components for dimensionality reduction
            layer_name: Optional layer name for context
            
        Returns:
            TokenClusteringResult with per-token-type clustering
        """
        batch_size, seq_len, hidden_dim = activations.shape
        
        # Require token_ids for this method
        if token_ids is None:
            error = "token_ids required for per_token_type clustering"
            logger.error(error)
            return TokenClusteringResult(
                token_clusters={},
                combined_clusters=np.array([]),
                cluster_centers=np.array([]),
                success=False,
                error=error
            )
        
        # Flatten arrays
        flat_activations = activations.reshape(-1, hidden_dim)
        flat_token_ids = token_ids.reshape(-1)
        flat_mask = token_mask.reshape(-1)
        
        # Get unique token types
        unique_tokens = np.unique(flat_token_ids[flat_mask])
        
        # Initialize results
        token_type_clusters = {}
        all_centers = []
        
        # Process each token type separately
        for token_type in unique_tokens:
            # Get indices for this token type
            token_indices = np.where((flat_token_ids == token_type) & flat_mask)[0]
            
            # Skip if no tokens of this type
            if len(token_indices) == 0:
                continue
            
            # Extract activations for this token type
            token_activations = flat_activations[token_indices]
            
            # Apply dimensionality reduction if needed
            if dimensionality_reduction and token_activations.shape[1] > n_components:
                result = self.dim_reducer.reduce_dimensionality(
                    activations=token_activations,
                    n_components=n_components,
                    method="auto",
                    layer_name=f"{layer_name}_token{token_type}" if layer_name else f"token{token_type}"
                )
                
                if result.success:
                    token_activations = result.reduced_activations
            
            # Adjust number of clusters if needed
            actual_clusters = min(self.n_clusters, token_activations.shape[0])
            
            if actual_clusters < self.n_clusters:
                logger.warning(f"Reduced clusters from {self.n_clusters} to {actual_clusters} for token type {token_type}")
            
            # Cluster the activations
            kmeans = KMeans(
                n_clusters=actual_clusters,
                random_state=self.random_state,
                n_init=10
            )
            
            type_clusters = kmeans.fit_predict(token_activations)
            
            # Store results
            token_type_clusters[token_type] = {
                "indices": token_indices,
                "clusters": type_clusters,
                "centers": kmeans.cluster_centers_
            }
            
            all_centers.append(kmeans.cluster_centers_)
        
        # Combine results into a [batch_size * seq_len] array
        combined_clusters = np.zeros(batch_size * seq_len, dtype=int)
        
        # Assign clusters for each token type
        for token_type, data in token_type_clusters.items():
            combined_clusters[data["indices"]] = data["clusters"]
        
        # Reshape to [batch_size, seq_len]
        combined_clusters = combined_clusters.reshape(batch_size, seq_len)
        
        # Create token-level clusters
        token_clusters = {}
        
        for tok_idx in range(seq_len):
            # Extract clusters for this token position across all batches
            token_clusters[tok_idx] = combined_clusters[:, tok_idx]
        
        # Combine all cluster centers
        all_centers_array = np.vstack(all_centers) if all_centers else np.array([])
        
        # Return result
        return TokenClusteringResult(
            token_clusters=token_clusters,
            combined_clusters=combined_clusters,
            cluster_centers=all_centers_array,
            metadata={
                "method": "per_token_type",
                "n_clusters": self.n_clusters,
                "token_types_processed": len(token_type_clusters),
                "dimensionality_reduction": dimensionality_reduction,
                "n_components": n_components if dimensionality_reduction else hidden_dim,
                "token_type_map": {str(k): len(v["indices"]) for k, v in token_type_clusters.items()}
            }
        )
    
    def get_token_paths(
        self,
        layer_clusters: Dict[str, TokenClusteringResult],
        layer_order: Optional[List[str]] = None
    ) -> Tuple[Dict[int, np.ndarray], List[str]]:
        """
        Extract token paths through layers.
        
        Args:
            layer_clusters: Dictionary mapping layer names to clustering results
            layer_order: Optional custom layer ordering
            
        Returns:
            Dictionary mapping token indices to paths, and list of layer names
        """
        # Ensure layer order is consistent
        if layer_order is None:
            layer_order = sorted(layer_clusters.keys())
        
        # Get dimensions from first layer
        first_layer = layer_clusters[layer_order[0]]
        batch_size, seq_len = first_layer.combined_clusters.shape
        
        # Initialize paths for each token position
        token_paths = {}
        
        for tok_idx in range(seq_len):
            # Create array for this token's path
            token_path = np.zeros((batch_size, len(layer_order)), dtype=int)
            
            # Fill in the path
            for layer_idx, layer_name in enumerate(layer_order):
                if layer_name in layer_clusters:
                    result = layer_clusters[layer_name]
                    if tok_idx in result.token_clusters:
                        token_path[:, layer_idx] = result.token_clusters[tok_idx]
            
            # Store this token's path
            token_paths[tok_idx] = token_path
        
        return token_paths, layer_order
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache = {}
        
        # Clear dimensionality reducer cache
        if hasattr(self.dim_reducer, "clear_cache"):
            self.dim_reducer.clear_cache()


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create example data
    batch_size = 4
    seq_len = 16
    hidden_dim = 768  # GPT-2 small hidden dim
    n_layers = 3
    
    # Generate random activations for multiple layers
    layer_activations = {}
    for i in range(n_layers):
        layer_name = f"layer{i}"
        layer_activations[layer_name] = np.random.normal(size=(batch_size, seq_len, hidden_dim))
    
    # Generate random token IDs
    token_ids = np.random.randint(0, 1000, size=(batch_size, seq_len))
    
    # Create token clusterer
    clusterer = GPT2TokenClusterer(
        n_clusters=8,
        token_aware=True,
        verbose=True
    )
    
    # Cluster tokens for each layer
    layer_clusters = {}
    
    for layer_name, activations in layer_activations.items():
        print(f"\nClustering layer: {layer_name}")
        
        # Try different clustering methods
        for method in ["combined", "per_position", "per_token_type"]:
            print(f"\nMethod: {method}")
            
            result = clusterer.cluster_token_activations(
                activations=activations,
                token_ids=token_ids,
                method=method,
                dimensionality_reduction=True,
                n_components=32
            )
            
            if result.success:
                print(f"Success: {len(result.token_clusters)} token positions processed")
                print(f"Combined clusters shape: {result.combined_clusters.shape}")
                print(f"Cluster centers shape: {result.cluster_centers.shape}")
                if result.metadata:
                    print(f"Metadata: {result.metadata}")
            else:
                print(f"Failed: {result.error}")
        
        # Store combined method results
        result = clusterer.cluster_token_activations(
            activations=activations,
            token_ids=token_ids,
            method="combined",
            dimensionality_reduction=True,
            n_components=32
        )
        
        layer_clusters[layer_name] = result
    
    # Extract token paths
    token_paths, layer_names = clusterer.get_token_paths(layer_clusters)
    
    print("\nToken paths extracted:")
    for tok_idx, paths in list(token_paths.items())[:3]:  # Show first 3 tokens
        print(f"Token {tok_idx}: Paths shape {paths.shape}")
        print(f"Example path for batch 0: {paths[0]}")