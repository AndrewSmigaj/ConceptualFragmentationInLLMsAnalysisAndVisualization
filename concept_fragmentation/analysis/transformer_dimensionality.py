"""
Enhanced dimensionality reduction techniques for transformer model activations.

This module provides specialized dimensionality reduction techniques optimized for
high-dimensional transformer model activations. It extends the standard PCA and UMAP
approaches with transformer-specific optimizations and fallback mechanisms.
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
import warnings
from pathlib import Path
import os
import hashlib
import pickle

# Conditionally import dimensionality reduction methods
try:
    from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
    from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available, some dimensionality reduction methods will be disabled")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available, install with: pip install umap-learn")

# Initialize logger
logger = logging.getLogger(__name__)


@dataclass
class DimensionalityReductionResult:
    """
    Data class for dimensionality reduction results.
    
    Attributes:
        reduced_activations: Reduced-dimension activations
        reducer: The fitted reducer object
        original_dim: Original dimensionality
        reduced_dim: Reduced dimensionality
        explained_variance: Explained variance if applicable
        method: Method used for reduction
        n_components: Number of components used
        parameters: Additional parameters used
        reconstruction_error: Reconstruction error if computed
        success: Whether the reduction was successful
        error_message: Error message if reduction failed
    """
    reduced_activations: np.ndarray
    reducer: Any
    original_dim: int
    reduced_dim: int
    explained_variance: Optional[float] = None
    method: str = "unknown"
    n_components: int = 0
    parameters: Dict[str, Any] = None
    reconstruction_error: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


class TransformerDimensionalityReducer:
    """
    Dimensionality reduction optimized for transformer model activations.
    
    This class provides methods for reducing the dimensionality of high-dimensional
    transformer activations, with built-in caching, progressive fallback mechanisms,
    and automatic parameter tuning for transformer representations.
    
    Attributes:
        cache_dir: Directory for caching reduction results
        random_state: Random seed for reproducibility
        use_cache: Whether to use disk caching
        verbose: Whether to log verbose information
        _methods: Available dimensionality reduction methods
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        random_state: int = 42,
        use_cache: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the dimensionality reducer.
        
        Args:
            cache_dir: Directory for caching results (created if doesn't exist)
            random_state: Random seed for reproducibility
            use_cache: Whether to use disk caching
            verbose: Whether to log verbose information
        """
        self.random_state = random_state
        self.use_cache = use_cache
        self.verbose = verbose
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path("cache") / "dimensionality_reduction"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Available methods (determined by installed libraries)
        self._methods = ["random_projection"]  # Always available
        
        if SKLEARN_AVAILABLE:
            self._methods.extend(["pca", "truncated_svd", "kernel_pca"])
        
        if UMAP_AVAILABLE:
            self._methods.append("umap")
            
        # Log available methods
        if self.verbose:
            logger.info(f"Available dimensionality reduction methods: {', '.join(self._methods)}")
    
    def _get_cache_key(
        self, 
        activations: np.ndarray,
        method: str,
        n_components: int,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Generate a cache key for the given parameters.
        
        Args:
            activations: Input activations
            method: Dimensionality reduction method
            n_components: Number of components
            parameters: Additional parameters
            
        Returns:
            Cache key string
        """
        # Create a signature from data shape, method, and params
        act_shape = activations.shape
        
        # Create a hash from a sample of the data to detect changes
        # without storing the full data
        if activations.size > 1000:
            # Sample a subset of points to avoid memory issues
            rng = np.random.RandomState(self.random_state)
            indices = rng.choice(activations.size, 1000, replace=False)
            sample = activations.ravel()[indices]
        else:
            sample = activations.ravel()
            
        # Create a hash from the sample statistics
        sample_hash = hashlib.md5()
        sample_hash.update(str(sample.mean()).encode())
        sample_hash.update(str(sample.std()).encode())
        sample_hash.update(str(act_shape).encode())
        
        # Add method and parameters to the hash
        param_str = str(sorted(parameters.items())) if parameters else ""
        sample_hash.update(f"{method}_{n_components}_{param_str}".encode())
        
        return f"{method}_{n_components}_{sample_hash.hexdigest()}"
    
    def _save_to_cache(
        self,
        cache_key: str,
        result: DimensionalityReductionResult
    ) -> None:
        """
        Save result to cache.
        
        Args:
            cache_key: Cache key string
            result: Dimensionality reduction result
        """
        if not self.use_cache:
            return
            
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
                
            if self.verbose:
                logger.info(f"Saved result to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def _load_from_cache(
        self,
        cache_key: str
    ) -> Optional[DimensionalityReductionResult]:
        """
        Load result from cache.
        
        Args:
            cache_key: Cache key string
            
        Returns:
            Cached result or None if not found
        """
        if not self.use_cache:
            return None
            
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, "rb") as f:
                result = pickle.load(f)
                
            if self.verbose:
                logger.info(f"Loaded result from cache: {cache_path}")
                
            return result
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None
    
    def reduce_dimensionality(
        self,
        activations: Union[np.ndarray, torch.Tensor],
        n_components: int = 50,
        method: str = "auto",
        layer_name: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        force_recompute: bool = False
    ) -> DimensionalityReductionResult:
        """
        Reduce the dimensionality of activations.
        
        Args:
            activations: Input activations [n_samples, n_features]
            n_components: Target number of dimensions
            method: Reduction method ('auto', 'pca', 'umap', 'random_projection', etc.)
            layer_name: Optional layer name for specialized parameters
            additional_params: Additional method-specific parameters
            force_recompute: Whether to force recomputation ignoring cache
            
        Returns:
            DimensionalityReductionResult with reduced activations and metadata
        """
        # Convert to numpy if needed
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        
        # Ensure 2D shape [n_samples, n_features]
        original_shape = activations.shape
        
        if len(original_shape) > 2:
            # Reshape [batch, seq_len, hidden_dim] to [batch*seq_len, hidden_dim]
            original_batch = original_shape[0]
            original_seq_len = original_shape[1]
            activations = activations.reshape(-1, original_shape[-1])
            reshaped = True
        elif len(original_shape) < 2:
            # Add a second dimension for single samples
            activations = activations.reshape(1, -1)
            reshaped = True
        else:
            reshaped = False
        
        # Store original dimensions
        n_samples, n_features = activations.shape
        
        # If already low-dimensional, just return original
        if n_features <= n_components:
            if self.verbose:
                logger.info(f"Original dimension {n_features} already <= {n_components}. Skipping reduction.")
            
            return DimensionalityReductionResult(
                reduced_activations=activations.copy(),
                reducer=None,
                original_dim=n_features,
                reduced_dim=n_features,
                method="identity",
                n_components=n_features,
                parameters={},
                success=True
            )
        
        # Prepare parameters
        params = additional_params.copy() if additional_params else {}
        
        # Auto-select method based on data characteristics
        if method == "auto":
            method = self._select_best_method(activations, n_components)
        
        # Check if method is available
        if method not in self._methods:
            avail_methods = ", ".join(self._methods)
            logger.warning(f"Method {method} not available. Use one of: {avail_methods}")
            
            # Fall back to a default method
            if "pca" in self._methods:
                method = "pca"
            elif "random_projection" in self._methods:
                method = "random_projection"
            else:
                # Last resort fallback to simple random sampling
                method = "random_projection"
        
        # Generate cache key
        cache_key = self._get_cache_key(activations, method, n_components, params)
        
        # Check cache
        if not force_recompute and self.use_cache:
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                # Reshape result back to original shape if needed
                if reshaped and len(original_shape) > 2:
                    # Reshape back to [batch, seq_len, reduced_dim]
                    result_activations = cached_result.reduced_activations.reshape(
                        original_batch, original_seq_len, cached_result.reduced_dim
                    )
                    cached_result.reduced_activations = result_activations
                
                return cached_result
        
        # Compute new reduction
        try:
            # Apply the selected method
            if method == "pca":
                result = self._reduce_with_pca(activations, n_components, params)
            elif method == "umap":
                result = self._reduce_with_umap(activations, n_components, params)
            elif method == "truncated_svd":
                result = self._reduce_with_truncated_svd(activations, n_components, params)
            elif method == "kernel_pca":
                result = self._reduce_with_kernel_pca(activations, n_components, params)
            elif method == "random_projection":
                result = self._reduce_with_random_projection(activations, n_components, params)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Reshape result back to original shape if needed
            if reshaped and len(original_shape) > 2:
                # Reshape back to [batch, seq_len, reduced_dim]
                result_activations = result.reduced_activations.reshape(
                    original_batch, original_seq_len, result.reduced_dim
                )
                result.reduced_activations = result_activations
            
            # Cache the result
            if self.use_cache:
                # Store the 2D version in cache to avoid issues with cache key
                result_to_cache = DimensionalityReductionResult(
                    reduced_activations=(
                        activations if result.success is False else 
                        result.reduced_activations.reshape(-1, result.reduced_dim)
                    ),
                    **{k: v for k, v in vars(result).items() if k != 'reduced_activations'}
                )
                self._save_to_cache(cache_key, result_to_cache)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during dimensionality reduction with {method}: {e}")
            
            # Return original activations on failure
            return DimensionalityReductionResult(
                reduced_activations=activations.copy(),
                reducer=None,
                original_dim=n_features,
                reduced_dim=n_features,
                method="failed",
                n_components=n_features,
                parameters={},
                success=False,
                error_message=str(e)
            )
    
    def _select_best_method(
        self,
        activations: np.ndarray,
        n_components: int
    ) -> str:
        """
        Automatically select the best dimensionality reduction method.
        
        Args:
            activations: Input activations
            n_components: Target number of dimensions
            
        Returns:
            Name of the selected method
        """
        n_samples, n_features = activations.shape
        
        # For very high-dimensional data with few samples, use TruncatedSVD
        if n_samples < 1000 and n_features > 10000 and "truncated_svd" in self._methods:
            return "truncated_svd"
        
        # For medium to large datasets with high dimensions, use PCA
        elif (n_samples >= 1000 or n_features <= 10000) and "pca" in self._methods:
            return "pca"
        
        # If UMAP is available and we're reducing to very low dimensions, use it
        elif n_components <= 3 and "umap" in self._methods:
            return "umap"
        
        # Fallback to random projection, which works in all scenarios
        else:
            return "random_projection"
    
    def _reduce_with_pca(
        self,
        activations: np.ndarray,
        n_components: int,
        params: Dict[str, Any]
    ) -> DimensionalityReductionResult:
        """
        Reduce dimensionality using PCA.
        
        Args:
            activations: Input activations
            n_components: Target number of dimensions
            params: Additional parameters
            
        Returns:
            DimensionalityReductionResult with PCA results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for PCA")
        
        # Set default parameters
        pca_params = {
            "random_state": self.random_state,
            "whiten": params.get("whiten", False),
            "svd_solver": params.get("svd_solver", "auto")
        }
        
        # Use svd_solver=randomized for high-dimensional data
        if activations.shape[1] > 500:
            pca_params["svd_solver"] = "randomized"
        
        # Initialize PCA
        pca = PCA(n_components=n_components, **pca_params)
        
        # Apply PCA
        start_time = None
        if self.verbose:
            import time
            start_time = time.time()
            logger.info(f"Applying PCA to reduce from {activations.shape[1]} to {n_components} dimensions")
        
        reduced_activations = pca.fit_transform(activations)
        
        if self.verbose and start_time is not None:
            import time
            logger.info(f"PCA completed in {time.time() - start_time:.2f} seconds")
        
        # Compute explained variance
        explained_variance = float(pca.explained_variance_ratio_.sum())
        
        if self.verbose:
            logger.info(f"PCA explained variance: {explained_variance:.4f}")
        
        return DimensionalityReductionResult(
            reduced_activations=reduced_activations,
            reducer=pca,
            original_dim=activations.shape[1],
            reduced_dim=n_components,
            explained_variance=explained_variance,
            method="pca",
            n_components=n_components,
            parameters=pca_params
        )
    
    def _reduce_with_truncated_svd(
        self,
        activations: np.ndarray,
        n_components: int,
        params: Dict[str, Any]
    ) -> DimensionalityReductionResult:
        """
        Reduce dimensionality using TruncatedSVD.
        
        Args:
            activations: Input activations
            n_components: Target number of dimensions
            params: Additional parameters
            
        Returns:
            DimensionalityReductionResult with TruncatedSVD results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for TruncatedSVD")
        
        # Set default parameters
        svd_params = {
            "random_state": self.random_state,
            "algorithm": params.get("algorithm", "randomized"),
            "n_iter": params.get("n_iter", 5)
        }
        
        # Initialize TruncatedSVD
        svd = TruncatedSVD(n_components=n_components, **svd_params)
        
        # Apply TruncatedSVD
        if self.verbose:
            logger.info(f"Applying TruncatedSVD to reduce from {activations.shape[1]} to {n_components} dimensions")
        
        reduced_activations = svd.fit_transform(activations)
        
        # Compute explained variance
        explained_variance = float(svd.explained_variance_ratio_.sum())
        
        if self.verbose:
            logger.info(f"TruncatedSVD explained variance: {explained_variance:.4f}")
        
        return DimensionalityReductionResult(
            reduced_activations=reduced_activations,
            reducer=svd,
            original_dim=activations.shape[1],
            reduced_dim=n_components,
            explained_variance=explained_variance,
            method="truncated_svd",
            n_components=n_components,
            parameters=svd_params
        )
    
    def _reduce_with_kernel_pca(
        self,
        activations: np.ndarray,
        n_components: int,
        params: Dict[str, Any]
    ) -> DimensionalityReductionResult:
        """
        Reduce dimensionality using KernelPCA.
        
        Args:
            activations: Input activations
            n_components: Target number of dimensions
            params: Additional parameters
            
        Returns:
            DimensionalityReductionResult with KernelPCA results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for KernelPCA")
        
        # Set default parameters
        kernel_pca_params = {
            "random_state": self.random_state,
            "kernel": params.get("kernel", "rbf"),
            "gamma": params.get("gamma", None),
            "fit_inverse_transform": params.get("fit_inverse_transform", False)
        }
        
        # Initialize KernelPCA
        kpca = KernelPCA(n_components=n_components, **kernel_pca_params)
        
        # Apply KernelPCA
        if self.verbose:
            logger.info(f"Applying KernelPCA to reduce from {activations.shape[1]} to {n_components} dimensions")
        
        reduced_activations = kpca.fit_transform(activations)
        
        return DimensionalityReductionResult(
            reduced_activations=reduced_activations,
            reducer=kpca,
            original_dim=activations.shape[1],
            reduced_dim=n_components,
            explained_variance=None,  # KernelPCA doesn't provide explained variance
            method="kernel_pca",
            n_components=n_components,
            parameters=kernel_pca_params
        )
    
    def _reduce_with_umap(
        self,
        activations: np.ndarray,
        n_components: int,
        params: Dict[str, Any]
    ) -> DimensionalityReductionResult:
        """
        Reduce dimensionality using UMAP.
        
        Args:
            activations: Input activations
            n_components: Target number of dimensions
            params: Additional parameters
            
        Returns:
            DimensionalityReductionResult with UMAP results
        """
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP is required for UMAP reduction")
        
        # Set default parameters
        umap_params = {
            "random_state": self.random_state,
            "n_neighbors": params.get("n_neighbors", 15),
            "min_dist": params.get("min_dist", 0.1),
            "metric": params.get("metric", "euclidean")
        }
        
        # Initialize UMAP
        umap_reducer = umap.UMAP(n_components=n_components, **umap_params)
        
        # Apply UMAP
        if self.verbose:
            logger.info(f"Applying UMAP to reduce from {activations.shape[1]} to {n_components} dimensions")
        
        reduced_activations = umap_reducer.fit_transform(activations)
        
        return DimensionalityReductionResult(
            reduced_activations=reduced_activations,
            reducer=umap_reducer,
            original_dim=activations.shape[1],
            reduced_dim=n_components,
            explained_variance=None,  # UMAP doesn't provide explained variance
            method="umap",
            n_components=n_components,
            parameters=umap_params
        )
    
    def _reduce_with_random_projection(
        self,
        activations: np.ndarray,
        n_components: int,
        params: Dict[str, Any]
    ) -> DimensionalityReductionResult:
        """
        Reduce dimensionality using RandomProjection.
        
        Args:
            activations: Input activations
            n_components: Target number of dimensions
            params: Additional parameters
            
        Returns:
            DimensionalityReductionResult with RandomProjection results
        """
        if not SKLEARN_AVAILABLE:
            # If sklearn is not available, implement a simple random projection
            if self.verbose:
                logger.info("scikit-learn not available, using numpy-based random projection")
            
            # Create a random projection matrix
            rng = np.random.RandomState(self.random_state)
            projection_matrix = rng.normal(size=(activations.shape[1], n_components))
            
            # Normalize columns
            projection_matrix /= np.sqrt(np.sum(projection_matrix**2, axis=0, keepdims=True))
            
            # Apply projection
            reduced_activations = activations @ projection_matrix
            
            return DimensionalityReductionResult(
                reduced_activations=reduced_activations,
                reducer=projection_matrix,
                original_dim=activations.shape[1],
                reduced_dim=n_components,
                explained_variance=None,
                method="numpy_random_projection",
                n_components=n_components,
                parameters={"random_state": self.random_state}
            )
        
        # Set default parameters
        rp_params = {
            "random_state": self.random_state,
            "eps": params.get("eps", 0.1),
            "dense_output": params.get("dense_output", True)
        }
        
        # Choose between Gaussian and Sparse random projection
        use_sparse = params.get("use_sparse", False)
        
        if use_sparse:
            # Sparse random projection is faster and scales better
            density = params.get("density", "auto")
            rp = SparseRandomProjection(n_components=n_components, density=density, **rp_params)
        else:
            # Gaussian random projection preserves distances better
            rp = GaussianRandomProjection(n_components=n_components, **rp_params)
        
        # Apply random projection
        if self.verbose:
            logger.info(f"Applying {'Sparse' if use_sparse else 'Gaussian'} RandomProjection to reduce from {activations.shape[1]} to {n_components} dimensions")
        
        reduced_activations = rp.fit_transform(activations)
        
        return DimensionalityReductionResult(
            reduced_activations=reduced_activations,
            reducer=rp,
            original_dim=activations.shape[1],
            reduced_dim=n_components,
            explained_variance=None,  # Random projection doesn't provide explained variance
            method="sparse_random_projection" if use_sparse else "gaussian_random_projection",
            n_components=n_components,
            parameters=rp_params
        )
    
    def progressive_dimensionality_reduction(
        self,
        activations: Union[np.ndarray, torch.Tensor],
        target_dim: int = 50,
        initial_method: str = "pca",
        secondary_method: str = "umap",
        pca_variance_threshold: float = 0.95,
        layer_name: Optional[str] = None,
        force_recompute: bool = False
    ) -> DimensionalityReductionResult:
        """
        Apply multi-stage dimensionality reduction for extremely high-dimensional data.
        
        This method first applies an initial reduction (like PCA) to capture the
        principal components, then applies a secondary method (like UMAP) to
        further reduce to the final target dimensions.
        
        Args:
            activations: Input activations
            target_dim: Final target dimensionality
            initial_method: Method for initial reduction (usually PCA)
            secondary_method: Method for secondary reduction (usually UMAP)
            pca_variance_threshold: Minimum explained variance to preserve in initial step
            layer_name: Optional layer name for specialized parameters
            force_recompute: Whether to force recomputation ignoring cache
            
        Returns:
            DimensionalityReductionResult for the full reduction pipeline
        """
        # Convert to numpy if needed
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        
        # Ensure 2D shape [n_samples, n_features]
        original_shape = activations.shape
        
        if len(original_shape) > 2:
            # Reshape [batch, seq_len, hidden_dim] to [batch*seq_len, hidden_dim]
            activations = activations.reshape(-1, original_shape[-1])
            reshaped = True
        else:
            reshaped = False
        
        # Original dimensions
        n_samples, n_features = activations.shape
        
        # Step 1: Initial reduction (e.g., PCA)
        if self.verbose:
            logger.info(f"Progressive dimensionality reduction: Step 1 - {initial_method}")
        
        # Determine intermediate dimension
        if initial_method == "pca" and SKLEARN_AVAILABLE:
            # For PCA, we can adapt the components to preserve variance
            # First, create a small-scale PCA to estimate components needed
            from sklearn.decomposition import PCA
            
            try:
                # Use a subset of data to estimate required components
                max_samples_for_estimate = min(1000, n_samples)
                sample_indices = np.random.choice(n_samples, max_samples_for_estimate, replace=False)
                
                # Estimate with a small PCA
                pca_estimate = PCA(n_components=min(200, n_features), random_state=self.random_state)
                pca_estimate.fit(activations[sample_indices])
                
                # Find how many components to reach target variance
                cumulative_variance = np.cumsum(pca_estimate.explained_variance_ratio_)
                required_components = np.argmax(cumulative_variance >= pca_variance_threshold) + 1
                
                # Set an upper bound to avoid too many dimensions
                intermediate_dim = min(required_components, 200)
                
                if self.verbose:
                    logger.info(f"Estimated {intermediate_dim} components needed to preserve {pca_variance_threshold:.2f} variance")
            
            except Exception as e:
                logger.warning(f"Error estimating PCA components: {e}")
                # Fallback to a reasonable intermediate dimension
                intermediate_dim = min(200, n_features // 2)
        else:
            # For other methods, use a heuristic
            intermediate_dim = min(200, n_features // 2)
        
        # Ensure intermediate_dim is at least target_dim
        intermediate_dim = max(intermediate_dim, target_dim)
        
        # Apply initial dimensionality reduction
        initial_result = self.reduce_dimensionality(
            activations=activations,
            n_components=intermediate_dim,
            method=initial_method,
            layer_name=layer_name,
            force_recompute=force_recompute
        )
        
        # Check if successful
        if not initial_result.success:
            logger.warning(f"Initial dimensionality reduction failed: {initial_result.error_message}")
            # Return the original data with error details
            return DimensionalityReductionResult(
                reduced_activations=activations.copy(),
                reducer=None,
                original_dim=n_features,
                reduced_dim=n_features,
                method="failed_progressive",
                n_components=n_features,
                parameters={},
                success=False,
                error_message=f"Initial {initial_method} failed: {initial_result.error_message}"
            )
        
        # If we already reached target_dim, just return the result
        if initial_result.reduced_dim <= target_dim:
            return initial_result
        
        # Step 2: Secondary reduction (e.g., UMAP)
        if self.verbose:
            logger.info(f"Progressive dimensionality reduction: Step 2 - {secondary_method}")
        
        # Apply secondary dimensionality reduction
        secondary_result = self.reduce_dimensionality(
            activations=initial_result.reduced_activations,
            n_components=target_dim,
            method=secondary_method,
            layer_name=layer_name,
            force_recompute=force_recompute
        )
        
        # Check if successful
        if not secondary_result.success:
            logger.warning(f"Secondary dimensionality reduction failed: {secondary_result.error_message}")
            # Return the initial reduction result
            return initial_result
        
        # Create a result that includes both steps
        progressive_result = DimensionalityReductionResult(
            reduced_activations=secondary_result.reduced_activations,
            reducer=[initial_result.reducer, secondary_result.reducer],
            original_dim=n_features,
            reduced_dim=target_dim,
            explained_variance=initial_result.explained_variance,
            method=f"{initial_method}+{secondary_method}",
            n_components=target_dim,
            parameters={
                "initial_method": initial_method,
                "intermediate_dim": intermediate_dim,
                "secondary_method": secondary_method,
                "target_dim": target_dim,
                "initial_params": initial_result.parameters,
                "secondary_params": secondary_result.parameters
            },
            success=True
        )
        
        # Reshape back to original dimensions if needed
        if reshaped and len(original_shape) > 2:
            progressive_result.reduced_activations = progressive_result.reduced_activations.reshape(
                original_shape[0], original_shape[1], target_dim
            )
        
        return progressive_result


class DimensionalityReductionPipelineStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """
    Pipeline stage for dimensionality reduction.
    
    This stage applies dimensionality reduction to activations from transformer models,
    using the TransformerDimensionalityReducer class to handle high-dimensional spaces.
    
    Attributes:
        reducer: TransformerDimensionalityReducer instance
        n_components: Target number of dimensions
        method: Reduction method to use
        filter_layers: Optional regex to filter layer names
        progressive: Whether to use progressive reduction
        use_cache: Whether to use disk caching
    """
    
    def __init__(
        self,
        n_components: int = 50,
        method: str = "auto",
        filter_layers: Optional[str] = None,
        progressive: bool = True,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        random_state: int = 42,
        name: str = "DimensionalityReduction"
    ):
        """
        Initialize dimensionality reduction stage.
        
        Args:
            n_components: Target number of dimensions
            method: Reduction method to use
            filter_layers: Optional regex to filter layer names
            progressive: Whether to use progressive reduction
            use_cache: Whether to use disk caching
            cache_dir: Directory for caching results
            random_state: Random seed for reproducibility
            name: Name for this stage
        """
        super().__init__(name=name)
        self.reducer = TransformerDimensionalityReducer(
            cache_dir=cache_dir,
            random_state=random_state,
            use_cache=use_cache
        )
        self.n_components = n_components
        self.method = method
        self.filter_layers = filter_layers
        self.progressive = progressive
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process activations through dimensionality reduction.
        
        The input should be a dictionary with:
        - 'activations': Dictionary mapping layer names to activation tensors
                        [batch_size, seq_len, hidden_dim] or [n_samples, hidden_dim]
        
        Returns:
            Dictionary with the input data and added reduced activations
        """
        result = data.copy()
        
        # Check for required inputs
        if 'activations' not in data:
            raise ValueError("Input must contain 'activations' key")
        
        activations = data['activations']
        
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
        reduced_activations = {}
        reduction_metadata = {}
        
        for layer_name, layer_activations in filtered_activations.items():
            # Skip non-tensor activations or metadata
            if not isinstance(layer_activations, (torch.Tensor, np.ndarray)):
                continue
            
            # Check if high-dimensional
            if isinstance(layer_activations, torch.Tensor):
                hidden_dim = layer_activations.shape[-1]
            else:
                hidden_dim = layer_activations.shape[-1]
            
            # Apply dimensionality reduction
            if hidden_dim > self.n_components:
                if self.progressive:
                    # Use progressive reduction for very high-dimensional data
                    reduction_result = self.reducer.progressive_dimensionality_reduction(
                        activations=layer_activations,
                        target_dim=self.n_components,
                        layer_name=layer_name
                    )
                else:
                    # Use single-step reduction
                    reduction_result = self.reducer.reduce_dimensionality(
                        activations=layer_activations,
                        n_components=self.n_components,
                        method=self.method,
                        layer_name=layer_name
                    )
                
                reduced_activations[layer_name] = reduction_result.reduced_activations
                
                # Store metadata
                reduction_metadata[layer_name] = {
                    "original_dim": reduction_result.original_dim,
                    "reduced_dim": reduction_result.reduced_dim,
                    "method": reduction_result.method,
                    "success": reduction_result.success
                }
                
                # Add explained variance if available
                if reduction_result.explained_variance is not None:
                    reduction_metadata[layer_name]["explained_variance"] = reduction_result.explained_variance
            else:
                # No reduction needed
                reduced_activations[layer_name] = layer_activations
                reduction_metadata[layer_name] = {
                    "original_dim": hidden_dim,
                    "reduced_dim": hidden_dim,
                    "method": "identity",
                    "success": True
                }
        
        # Add results to output
        result['reduced_activations'] = reduced_activations
        result['reduction_metadata'] = reduction_metadata
        
        return result


# Import here to avoid circular imports
from ..pipeline.pipeline import PipelineStageBase