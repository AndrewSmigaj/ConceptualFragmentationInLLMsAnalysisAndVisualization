"""
Window detection utilities for Layer Window Manager.
Wraps existing metrics from concept_fragmentation for boundary detection.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

# Import existing metrics - no reimplementation
from concept_fragmentation.metrics import (
    representation_stability,
    cross_layer_metrics,
    transformer_metrics
)

logger = logging.getLogger(__name__)


class WindowDetectionMetrics:
    """Wrapper for existing metrics used in window boundary detection."""
    
    @staticmethod
    def compute_boundary_metrics(
        activations_dict: Dict[str, np.ndarray],
        clusters_dict: Optional[Dict[str, Dict]] = None,
        compute_all: bool = False
    ) -> Dict[str, Any]:
        """
        Compute metrics that help identify layer boundaries.
        
        Args:
            activations_dict: Layer name -> activation tensor
            clusters_dict: Layer name -> cluster information (optional)
            compute_all: Whether to compute all available metrics
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        try:
            # Always compute representation stability (most useful for boundaries)
            stability_result = representation_stability.compute_representation_stability(
                activations_dict=activations_dict,
                normalize=True
            )
            
            # Extract layer-to-layer stability scores
            metrics['stability'] = WindowDetectionMetrics._extract_layer_scores(
                stability_result, 
                list(activations_dict.keys())
            )
            
        except Exception as e:
            logger.warning(f"Could not compute stability metrics: {e}")
            metrics['stability'] = None
        
        # Compute cluster-based metrics if available
        if clusters_dict and compute_all:
            try:
                # Path density shows connectivity between layers
                density_scores = []
                layer_names = list(clusters_dict.keys())
                
                for i in range(len(layer_names) - 1):
                    layer1, layer2 = layer_names[i], layer_names[i+1]
                    if layer1 in clusters_dict and layer2 in clusters_dict:
                        density, _ = cross_layer_metrics.compute_path_density(
                            {layer1: clusters_dict[layer1], layer2: clusters_dict[layer2]},
                            min_overlap=0.1
                        )
                        # Extract the single density value between these layers
                        if (layer1, layer2) in density:
                            density_scores.append(density[(layer1, layer2)])
                        else:
                            density_scores.append(0.0)
                
                metrics['density'] = density_scores
                
            except Exception as e:
                logger.warning(f"Could not compute density metrics: {e}")
                metrics['density'] = None
        
        return metrics
    
    @staticmethod
    def _extract_layer_scores(
        stability_result: Dict[Tuple[str, str], float],
        layer_names: List[str]
    ) -> List[float]:
        """Extract sequential layer stability scores from pairwise results."""
        scores = []
        
        for i in range(len(layer_names) - 1):
            key = (layer_names[i], layer_names[i+1])
            if key in stability_result:
                # Lower stability = higher change = potential boundary
                # Invert so peaks indicate boundaries
                scores.append(1.0 - stability_result[key])
            else:
                scores.append(0.0)
        
        return scores
    
    @staticmethod
    def normalize_metrics_for_display(
        metrics: Dict[str, List[float]], 
        target_range: Tuple[float, float] = (0.0, 1.0)
    ) -> Dict[str, List[float]]:
        """
        Normalize metrics to a common range for visualization.
        
        Args:
            metrics: Dictionary of metric arrays
            target_range: Target range for normalization
            
        Returns:
            Normalized metrics dictionary
        """
        normalized = {}
        min_val, max_val = target_range
        
        for name, values in metrics.items():
            if values is None or len(values) == 0:
                normalized[name] = values
                continue
                
            arr = np.array(values)
            if np.std(arr) == 0:
                # All values are the same
                normalized[name] = [0.5 * (min_val + max_val)] * len(values)
            else:
                # Min-max normalization
                arr_min, arr_max = np.min(arr), np.max(arr)
                arr_norm = (arr - arr_min) / (arr_max - arr_min)
                arr_norm = arr_norm * (max_val - min_val) + min_val
                normalized[name] = arr_norm.tolist()
        
        return normalized
    
    @staticmethod
    def aggregate_metrics_for_deep_networks(
        metrics: Dict[str, List[float]], 
        target_points: int = 50
    ) -> Dict[str, List[float]]:
        """
        Aggregate metrics for visualization of deep networks.
        Uses moving average to reduce number of points while preserving patterns.
        
        Args:
            metrics: Dictionary of metric arrays
            target_points: Target number of points for visualization
            
        Returns:
            Aggregated metrics dictionary
        """
        aggregated = {}
        
        for name, values in metrics.items():
            if values is None or len(values) <= target_points:
                aggregated[name] = values
                continue
            
            # Calculate window size for moving average
            window_size = max(2, len(values) // target_points)
            
            # Apply moving average
            arr = np.array(values)
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(arr, kernel, mode='valid')
            
            # Subsample to get target number of points
            indices = np.linspace(0, len(smoothed) - 1, target_points, dtype=int)
            aggregated[name] = smoothed[indices].tolist()
        
        return aggregated
    
    @staticmethod
    def detect_peaks(
        values: List[float], 
        min_prominence: float = 0.2,
        min_distance: int = 2
    ) -> List[int]:
        """
        Detect peaks in metric values that could indicate boundaries.
        
        Args:
            values: Metric values
            min_prominence: Minimum peak prominence (0-1 range)
            min_distance: Minimum distance between peaks (in layers)
            
        Returns:
            List of peak indices (potential boundaries)
        """
        if not values or len(values) < 3:
            return []
        
        arr = np.array(values)
        
        # Simple peak detection
        peaks = []
        for i in range(1, len(arr) - 1):
            # Check if local maximum
            if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                # Check prominence
                left_min = np.min(arr[max(0, i-min_distance):i])
                right_min = np.min(arr[i+1:min(len(arr), i+min_distance+1)])
                prominence = arr[i] - max(left_min, right_min)
                
                if prominence >= min_prominence:
                    peaks.append(i)
        
        return peaks