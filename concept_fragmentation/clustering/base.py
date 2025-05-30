"""Base classes for clustering algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
from .exceptions import ClustererError, ClusteringNotFittedError


class BaseClusterer(ABC):
    """Abstract base class for all clustering algorithms.
    
    This class defines the interface that all clustering algorithms must implement.
    It follows the scikit-learn API conventions for consistency.
    
    Attributes:
        n_clusters: Number of clusters
        fitted: Whether the clusterer has been fitted to data
    """
    
    def __init__(self, n_clusters: int, random_state: Optional[int] = None):
        """Initialize the clusterer.
        
        Args:
            n_clusters: Number of clusters to form
            random_state: Random seed for reproducibility
            
        Raises:
            ValueError: If n_clusters is less than 2
        """
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
            
        self._n_clusters = n_clusters
        self._random_state = random_state
        self._fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseClusterer':
        """Fit the clusterer to data.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            self: The fitted clusterer instance
            
        Raises:
            ClustererError: If fitting fails
            InvalidDataError: If input data is invalid
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for samples.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            labels: Cluster labels for each sample
            
        Raises:
            ClusteringNotFittedError: If clusterer hasn't been fitted
            InvalidDataError: If input data is invalid
        """
        pass
    
    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit clusterer and predict labels in one step.
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
            
        Returns:
            labels: Cluster labels for each sample
            
        Raises:
            ClustererError: If fitting or prediction fails
            InvalidDataError: If input data is invalid
        """
        pass
    
    @property
    def n_clusters(self) -> int:
        """Return number of clusters."""
        return self._n_clusters
    
    @property
    def fitted(self) -> bool:
        """Return whether the clusterer has been fitted."""
        return self._fitted
    
    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        """Validate input data.
        
        Args:
            X: Input data
            
        Returns:
            X: Validated data as numpy array
            
        Raises:
            InvalidDataError: If data is invalid
        """
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except Exception as e:
                raise InvalidDataError(f"Cannot convert input to array: {e}")
                
        if X.ndim != 2:
            raise InvalidDataError(f"Expected 2D array, got {X.ndim}D")
            
        if X.shape[0] == 0:
            raise InvalidDataError("Cannot cluster empty dataset")
            
        if X.shape[0] < self.n_clusters:
            raise InvalidDataError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}"
            )
            
        return X
    
    def _check_fitted(self) -> None:
        """Check if clusterer has been fitted.
        
        Raises:
            ClusteringNotFittedError: If not fitted
        """
        if not self.fitted:
            raise ClusteringNotFittedError(
                f"{self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' before using this method."
            )