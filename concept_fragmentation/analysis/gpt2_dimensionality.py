"""
GPT-2 specific dimensionality reduction techniques.

This module extends the transformer_dimensionality module with GPT-2 specific
optimizations for handling the high-dimensional representations in GPT-2 models.
It provides specialized preprocessing for GPT-2 hidden states and optimized
reduction techniques for GPT-2's representation spaces.
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
import warnings
from pathlib import Path
import hashlib
import pickle
import os

# Import base reducer
from concept_fragmentation.analysis.transformer_dimensionality import (
    TransformerDimensionalityReducer,
    DimensionalityReductionResult,
    SKLEARN_AVAILABLE,
    UMAP_AVAILABLE
)

# Conditionally import sklearn components
if SKLEARN_AVAILABLE:
    from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
    from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
    from sklearn.preprocessing import StandardScaler

# Initialize logger
logger = logging.getLogger(__name__)


class GPT2DimensionalityReducer(TransformerDimensionalityReducer):
    """
    Dimensionality reduction optimized for GPT-2 model activations.
    
    This class extends TransformerDimensionalityReducer with GPT-2 specific
    optimizations, including handling of positional encoding, specialized
    preprocessing for causal language models, and optimized reduction methods
    for GPT-2's representational spaces.
    
    Attributes:
        cache_dir: Directory for caching reduction results
        random_state: Random seed for reproducibility
        use_cache: Whether to use disk caching
        verbose: Whether to log verbose information
        remove_positional: Whether to remove positional encoding effects
        standardize: Whether to standardize activations before reduction
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        random_state: int = 42,
        use_cache: bool = True,
        verbose: bool = False,
        remove_positional: bool = True,
        standardize: bool = True,
        gpt2_model_size: str = "small"  # small, medium, large, xl
    ):
        """
        Initialize the GPT-2 dimensionality reducer.
        
        Args:
            cache_dir: Directory for caching results
            random_state: Random seed for reproducibility
            use_cache: Whether to use disk caching
            verbose: Whether to log verbose information
            remove_positional: Whether to remove positional encoding effects
            standardize: Whether to standardize activations before reduction
            gpt2_model_size: GPT-2 model size (small, medium, large, xl)
        """
        # Initialize base class
        super().__init__(
            cache_dir=cache_dir,
            random_state=random_state,
            use_cache=use_cache,
            verbose=verbose
        )
        
        # Set GPT-2 specific attributes
        self.remove_positional = remove_positional
        self.standardize = standardize
        self.gpt2_model_size = gpt2_model_size
        
        # Override cache directory to be GPT-2 specific
        if cache_dir is None:
            self.cache_dir = Path("cache") / "gpt2_dimensionality_reduction" / gpt2_model_size
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Log specific parameters
        if self.verbose:
            logger.info(f"Initialized GPT2DimensionalityReducer for model size: {gpt2_model_size}")
            logger.info(f"Remove positional: {remove_positional}, Standardize: {standardize}")
    
    def _get_gpt2_hidden_dim(self) -> int:
        """
        Get the hidden dimension for the configured GPT-2 model size.
        
        Returns:
            Hidden dimension size
        """
        hidden_dims = {
            "small": 768,
            "medium": 1024,
            "large": 1280,
            "xl": 1600
        }
        
        return hidden_dims.get(self.gpt2_model_size.lower(), 768)
    
    def _get_cache_key(
        self, 
        activations: np.ndarray,
        method: str,
        n_components: int,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Generate a GPT-2 specific cache key for the given parameters.
        
        Args:
            activations: Input activations
            method: Dimensionality reduction method
            n_components: Number of components
            parameters: Additional parameters
            
        Returns:
            Cache key string
        """
        # Create a GPT-2 specific key incorporating model size
        base_key = super()._get_cache_key(activations, method, n_components, parameters)
        
        # Add GPT-2 specific parameters to the key
        gpt2_params = f"gpt2_{self.gpt2_model_size}_pos{int(self.remove_positional)}_std{int(self.standardize)}"
        
        return f"{gpt2_params}_{base_key}"
    
    def preprocess_gpt2_activations(
        self,
        activations: Union[torch.Tensor, np.ndarray],
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Preprocess GPT-2 hidden states before dimensionality reduction.
        
        Args:
            activations: Input activations [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
            layer_name: Optional layer name for context
            
        Returns:
            Preprocessed activations
        """
        # Convert to numpy if it's a torch tensor
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        
        # Get original shape
        original_shape = activations.shape
        
        # Ensure we're working with 2D data [n_samples, n_features]
        if len(original_shape) == 3:
            batch_size, seq_len, hidden_dim = original_shape
            activations_2d = activations.reshape(-1, hidden_dim)
        else:
            activations_2d = activations
        
        # Remove positional encoding effects if requested
        if self.remove_positional and len(original_shape) == 3:
            # For each position in the sequence, calculate the mean across batch
            position_means = np.zeros((seq_len, hidden_dim))
            
            for pos in range(seq_len):
                # Extract all batch items at this position
                pos_activations = activations[:, pos, :]
                # Calculate mean
                position_means[pos] = np.mean(pos_activations, axis=0)
            
            # Subtract position means
            for b in range(batch_size):
                for pos in range(seq_len):
                    # Subtract position-specific mean
                    activations[b, pos, :] -= position_means[pos]
            
            # Reshape to 2D
            activations_2d = activations.reshape(-1, hidden_dim)
        
        # Standardize activations if requested
        if self.standardize and SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            activations_2d = scaler.fit_transform(activations_2d)
        
        if self.verbose:
            logger.info(f"Preprocessed activations: {original_shape} -> {activations_2d.shape}")
        
        return activations_2d
    
    def reduce_dimensionality(
        self,
        activations: Union[torch.Tensor, np.ndarray],
        n_components: int,
        method: str = "auto",
        parameters: Dict[str, Any] = None,
        layer_name: Optional[str] = None,
        force_recompute: bool = False
    ) -> DimensionalityReductionResult:
        """
        Reduce dimensionality of GPT-2 activations using selected method.
        
        Args:
            activations: Input activations (either numpy array or torch tensor)
            n_components: Target number of dimensions
            method: Reduction method ('auto', 'pca', 'umap', 'truncated_svd', etc.)
            parameters: Additional parameters for the reduction method
            layer_name: Optional layer name for context (used in caching)
            force_recompute: Whether to force recomputation ignoring cache
            
        Returns:
            DimensionalityReductionResult with reduction results
        """
        # Preprocess activations specifically for GPT-2
        original_shape = activations.shape
        original_is_torch = isinstance(activations, torch.Tensor)
        
        # Apply GPT-2 specific preprocessing
        activations_2d = self.preprocess_gpt2_activations(
            activations=activations,
            layer_name=layer_name
        )
        
        # Override the method for very high-dimensional GPT-2 models
        if method == "auto":
            method = self._select_best_gpt2_method(
                activations=activations_2d,
                n_components=n_components
            )
        
        # Call the parent method with preprocessed activations
        result = super().reduce_dimensionality(
            activations=activations_2d,
            n_components=n_components,
            method=method,
            parameters=parameters,
            layer_name=layer_name,
            force_recompute=force_recompute
        )
        
        # Reshape result back to original shape if needed
        if result.success and len(original_shape) == 3:
            batch_size, seq_len, _ = original_shape
            result_activations = result.reduced_activations.reshape(
                batch_size, seq_len, result.reduced_dim
            )
            result.reduced_activations = result_activations
        
        # Convert back to torch if the input was a torch tensor
        if result.success and original_is_torch:
            result.reduced_activations = torch.tensor(
                result.reduced_activations,
                dtype=torch.float32
            )
        
        return result
    
    def _select_best_gpt2_method(
        self,
        activations: np.ndarray,
        n_components: int
    ) -> str:
        """
        Automatically select the best method for GPT-2 activations.
        
        Args:
            activations: Input activations
            n_components: Target number of dimensions
            
        Returns:
            Name of the selected method
        """
        n_samples, n_features = activations.shape
        gpt2_dim = self._get_gpt2_hidden_dim()
        
        # For very high-dimensional data (XL model), use TruncatedSVD or progressive reduction
        if n_features >= 1280 and "truncated_svd" in self._methods:
            return "truncated_svd"
        
        # For standard GPT-2 dimensions (small/medium), prefer PCA for interpretability
        elif n_features == gpt2_dim and "pca" in self._methods:
            return "pca"
        
        # For very low-dimensional projections (visualization), use UMAP
        elif n_components <= 3 and "umap" in self._methods:
            return "umap"
        
        # Fallback to random projection
        else:
            return "random_projection"
    
    def progressive_gpt2_reduction(
        self,
        activations: Union[torch.Tensor, np.ndarray],
        target_dim: int = 10,
        intermediate_dim: Optional[int] = None,
        initial_method: str = "truncated_svd",
        secondary_method: str = "pca",
        layer_name: Optional[str] = None,
        parameters: Dict[str, Any] = None
    ) -> DimensionalityReductionResult:
        """
        Progressive dimensionality reduction for very high-dimensional GPT-2 spaces.
        
        This performs reduction in two steps:
        1. Initial reduction to an intermediate dimensionality (e.g., 256)
        2. Secondary reduction to the target dimensionality
        
        Args:
            activations: Input activations
            target_dim: Final target dimensionality
            intermediate_dim: Intermediate dimensionality (auto if None)
            initial_method: Method for initial reduction
            secondary_method: Method for secondary reduction
            layer_name: Optional layer name for context
            parameters: Additional parameters
            
        Returns:
            DimensionalityReductionResult with final reduction results
        """
        # Start with GPT-2 preprocessing
        original_shape = activations.shape
        original_is_torch = isinstance(activations, torch.Tensor)
        
        # Apply GPT-2 specific preprocessing
        preprocessed = self.preprocess_gpt2_activations(
            activations=activations,
            layer_name=layer_name
        )
        
        # Get dimensions from preprocessed data
        n_samples, n_features = preprocessed.shape
        
        # Determine intermediate dimensionality if not provided
        if intermediate_dim is None:
            # Use square root of the original dimensionality (empirically good)
            intermediate_dim = min(256, int(np.sqrt(n_features * target_dim)))
            
            # Ensure intermediate_dim is at least double the target
            intermediate_dim = max(intermediate_dim, target_dim * 2)
            
            # Cap at 256 for GPU memory efficiency
            intermediate_dim = min(intermediate_dim, 256)
        
        if self.verbose:
            logger.info(f"Progressive reduction: {n_features} -> {intermediate_dim} -> {target_dim}")
            logger.info(f"Methods: {initial_method} -> {secondary_method}")
        
        # Step 1: Initial reduction to intermediate dimensionality
        initial_result = super().reduce_dimensionality(
            activations=preprocessed,
            n_components=intermediate_dim,
            method=initial_method,
            parameters=parameters,
            layer_name=f"{layer_name}_initial" if layer_name else None
        )
        
        if not initial_result.success:
            logger.warning("Initial reduction failed, returning preprocessed activations")
            # Return original data with failure flag
            return DimensionalityReductionResult(
                reduced_activations=preprocessed,
                reducer=None,
                original_dim=n_features,
                reduced_dim=n_features,
                method="failed",
                n_components=n_features,
                parameters={},
                success=False,
                error_message=initial_result.error_message
            )
        
        # Step 2: Secondary reduction to target dimensionality
        secondary_result = super().reduce_dimensionality(
            activations=initial_result.reduced_activations,
            n_components=target_dim,
            method=secondary_method,
            parameters=parameters,
            layer_name=f"{layer_name}_secondary" if layer_name else None
        )
        
        if not secondary_result.success:
            logger.warning("Secondary reduction failed, returning initial reduction")
            # Return initial reduction with partial success
            return DimensionalityReductionResult(
                reduced_activations=initial_result.reduced_activations,
                reducer=initial_result.reducer,
                original_dim=n_features,
                reduced_dim=intermediate_dim,
                method=f"{initial_method}_partial",
                n_components=intermediate_dim,
                parameters=parameters or {},
                success=True,
                error_message=secondary_result.error_message
            )
        
        # Successful progressive reduction
        # Create combined result
        final_result = DimensionalityReductionResult(
            reduced_activations=secondary_result.reduced_activations,
            reducer=(initial_result.reducer, secondary_result.reducer),
            original_dim=n_features,
            reduced_dim=target_dim,
            method=f"{initial_method}+{secondary_method}",
            n_components=target_dim,
            parameters=parameters or {},
            success=True,
            error_message=None
        )
        
        # Reshape result back to original shape if needed
        if len(original_shape) == 3:
            batch_size, seq_len, _ = original_shape
            result_activations = final_result.reduced_activations.reshape(
                batch_size, seq_len, final_result.reduced_dim
            )
            final_result.reduced_activations = result_activations
        
        # Convert back to torch if the input was a torch tensor
        if original_is_torch:
            final_result.reduced_activations = torch.tensor(
                final_result.reduced_activations,
                dtype=torch.float32
            )
        
        return final_result


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create example data similar to GPT-2 small hidden states
    batch_size = 4
    seq_len = 16
    hidden_dim = 768  # GPT-2 small hidden dim
    
    # Generate random activations
    activations = np.random.normal(size=(batch_size, seq_len, hidden_dim))
    
    # Create GPT-2 dimensionality reducer
    reducer = GPT2DimensionalityReducer(
        gpt2_model_size="small",
        verbose=True
    )
    
    # Reduce dimensionality
    result = reducer.reduce_dimensionality(
        activations=activations,
        n_components=32,
        method="auto"
    )
    
    print(f"\nReduction summary:")
    print(f"Original dimensions: {hidden_dim}")
    print(f"Reduced dimensions: {result.reduced_dim}")
    print(f"Method: {result.method}")
    print(f"Reduction shape: {result.reduced_activations.shape}")
    
    # Try progressive reduction
    prog_result = reducer.progressive_gpt2_reduction(
        activations=activations,
        target_dim=10
    )
    
    print(f"\nProgressive reduction summary:")
    print(f"Original dimensions: {hidden_dim}")
    print(f"Reduced dimensions: {prog_result.reduced_dim}")
    print(f"Method: {prog_result.method}")
    print(f"Reduction shape: {prog_result.reduced_activations.shape}")