"""Base classes for cluster labeling strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .exceptions import LabelerError


class BaseLabeler(ABC):
    """Abstract base class for cluster labeling strategies.
    
    This class defines the interface for all labeling strategies,
    including consistent labeling, semantic labeling, and others.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the labeler.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._labels = {}
        
    @abstractmethod
    def label_clusters(self, 
                      cluster_data: Dict[str, Any],
                      tokens: List[str],
                      **kwargs) -> Dict[str, Dict[str, Any]]:
        """Generate labels for clusters.
        
        Args:
            cluster_data: Dictionary containing cluster information
                - Should include cluster assignments by layer
                - May include cluster centroids, sizes, etc.
            tokens: List of token strings
            **kwargs: Additional labeling parameters
            
        Returns:
            Dictionary mapping layer_cluster keys to label information:
                {
                    "layer_0": {
                        "L0_C0": {
                            "label": "Primary Label (Secondary)",
                            "primary": "Primary Label",
                            "secondary": "Secondary",
                            "confidence": 0.95,
                            "metadata": {...}
                        },
                        ...
                    },
                    ...
                }
                
        Raises:
            LabelerError: If labeling fails
            InvalidClusterDataError: If cluster data is invalid
        """
        pass
    
    @abstractmethod
    def get_label_consistency(self, labels: Dict[str, Dict[str, Any]]) -> float:
        """Calculate label consistency score.
        
        Consistency measures how well similar clusters receive similar labels
        across different layers.
        
        Args:
            labels: Labels dictionary from label_clusters
            
        Returns:
            Consistency score between 0 and 1
            
        Raises:
            LabelerError: If consistency calculation fails
        """
        pass
    
    def get_cluster_similarity(self,
                              cluster1_tokens: List[str],
                              cluster2_tokens: List[str]) -> float:
        """Calculate similarity between two clusters based on their tokens.
        
        Args:
            cluster1_tokens: Tokens in first cluster
            cluster2_tokens: Tokens in second cluster
            
        Returns:
            Similarity score between 0 and 1
        """
        if not cluster1_tokens or not cluster2_tokens:
            return 0.0
            
        set1 = set(cluster1_tokens)
        set2 = set(cluster2_tokens)
        
        # Jaccard similarity
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def validate_cluster_data(self, cluster_data: Dict[str, Any]) -> None:
        """Validate cluster data structure.
        
        Args:
            cluster_data: Cluster data to validate
            
        Raises:
            InvalidClusterDataError: If data is invalid
        """
        if not isinstance(cluster_data, dict):
            raise InvalidClusterDataError("Cluster data must be a dictionary")
            
        # Check for required keys based on specific labeler needs
        self._validate_specific_requirements(cluster_data)
        
    def _validate_specific_requirements(self, cluster_data: Dict[str, Any]) -> None:
        """Validate specific requirements for this labeler.
        
        Subclasses should override this to add specific validation.
        
        Args:
            cluster_data: Cluster data to validate
        """
        pass
    
    @property
    def labels(self) -> Dict[str, Dict[str, Any]]:
        """Return generated labels."""
        return self._labels
    
    def save_labels(self, output_path: str) -> None:
        """Save labels to file.
        
        Args:
            output_path: Path to save labels
        """
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self._labels, f, indent=2)
            
    def load_labels(self, input_path: str) -> Dict[str, Dict[str, Any]]:
        """Load labels from file.
        
        Args:
            input_path: Path to load labels from
            
        Returns:
            Loaded labels dictionary
        """
        import json
        with open(input_path, 'r', encoding='utf-8') as f:
            self._labels = json.load(f)
        return self._labels