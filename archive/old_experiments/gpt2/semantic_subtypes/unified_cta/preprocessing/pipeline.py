"""
Preprocessing pipeline with standardization, PCA, and Procrustes alignment
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes

from logging_config import setup_logging

logger = setup_logging(__name__)


class PreprocessingPipeline:
    """
    Preprocessing pipeline for activation data.
    
    Steps:
    1. Standardization (z-score normalization)
    2. PCA dimensionality reduction
    3. Procrustes alignment between successive layers
    """
    
    def __init__(self, n_components: int = 128, align_layers: bool = True):
        """
        Initialize preprocessing pipeline.
        
        Args:
            n_components: Number of PCA components to keep
            align_layers: Whether to apply Procrustes alignment
        """
        self.n_components = n_components
        self.align_layers = align_layers
        self.scalers = {}  # One per layer
        self.pcas = {}      # One per layer
        self.procrustes_transforms = {}  # R matrices for alignment
        
        logger.info(f"Initialized preprocessing pipeline: "
                   f"n_components={n_components}, align_layers={align_layers}")
    
    def fit_transform(self, activations_by_layer: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Fit preprocessing pipeline and transform activations.
        
        Args:
            activations_by_layer: Dict mapping layer names to activation matrices
                                Each matrix is (n_samples, n_features)
                                
        Returns:
            Preprocessed activations with same structure
        """
        logger.info(f"Starting preprocessing for {len(activations_by_layer)} layers")
        
        preprocessed = {}
        prev_layer_data = None
        
        # Process layers in order
        layer_names = sorted(activations_by_layer.keys(), 
                           key=lambda x: int(x.split('_')[1]))
        
        for layer_name in layer_names:
            layer_idx = int(layer_name.split('_')[1])
            activations = activations_by_layer[layer_name]
            
            logger.info(f"Processing {layer_name}: shape {activations.shape}")
            
            # Step 1: Standardization
            scaler = StandardScaler()
            standardized = scaler.fit_transform(activations)
            self.scalers[layer_name] = scaler
            
            # Step 2: PCA
            n_components_actual = min(self.n_components, 
                                    min(activations.shape) - 1)
            pca = PCA(n_components=n_components_actual)
            reduced = pca.fit_transform(standardized)
            self.pcas[layer_name] = pca
            
            explained_var = pca.explained_variance_ratio_.sum()
            logger.info(f"  PCA: {activations.shape[1]} -> {n_components_actual} dims, "
                       f"explained variance: {explained_var:.3f}")
            
            # Step 3: Procrustes alignment (except for first layer)
            if self.align_layers and prev_layer_data is not None and layer_idx > 0:
                # Align current layer to previous layer
                R, scale = orthogonal_procrustes(reduced, prev_layer_data)
                aligned = reduced @ R
                self.procrustes_transforms[layer_name] = R
                
                # Measure alignment quality
                alignment_error = np.mean(np.square(aligned - prev_layer_data))
                logger.info(f"  Procrustes alignment error: {alignment_error:.6f}")
                
                preprocessed[layer_name] = aligned
                prev_layer_data = aligned
            else:
                preprocessed[layer_name] = reduced
                prev_layer_data = reduced
                
        logger.info("Preprocessing complete")
        return preprocessed
    
    def transform(self, activations_by_layer: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transform new data using fitted preprocessing.
        
        Args:
            activations_by_layer: New activation data
            
        Returns:
            Preprocessed activations
        """
        if not self.scalers:
            raise ValueError("Pipeline must be fitted before transform")
            
        preprocessed = {}
        
        for layer_name, activations in activations_by_layer.items():
            if layer_name not in self.scalers:
                logger.warning(f"Layer {layer_name} not seen during fit, skipping")
                continue
                
            # Apply saved transformations
            standardized = self.scalers[layer_name].transform(activations)
            reduced = self.pcas[layer_name].transform(standardized)
            
            # Apply Procrustes if available
            if layer_name in self.procrustes_transforms:
                aligned = reduced @ self.procrustes_transforms[layer_name]
                preprocessed[layer_name] = aligned
            else:
                preprocessed[layer_name] = reduced
                
        return preprocessed
    
    def save(self, filepath: str):
        """Save fitted pipeline to disk."""
        save_data = {
            'n_components': self.n_components,
            'align_layers': self.align_layers,
            'scalers': self.scalers,
            'pcas': self.pcas,
            'procrustes_transforms': self.procrustes_transforms
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
            
        logger.info(f"Saved preprocessing pipeline to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PreprocessingPipeline':
        """Load fitted pipeline from disk."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
            
        pipeline = cls(
            n_components=save_data['n_components'],
            align_layers=save_data['align_layers']
        )
        pipeline.scalers = save_data['scalers']
        pipeline.pcas = save_data['pcas']
        pipeline.procrustes_transforms = save_data['procrustes_transforms']
        
        logger.info(f"Loaded preprocessing pipeline from {filepath}")
        return pipeline


def load_and_preprocess_activations(
    activations_file: str = "activations_by_layer.pkl",
    n_components: int = 128,
    align_layers: bool = True,
    save_path: Optional[str] = None
) -> Tuple[Dict[str, np.ndarray], PreprocessingPipeline]:
    """
    Convenience function to load and preprocess activations.
    
    Args:
        activations_file: Path to pickled activations
        n_components: Number of PCA components
        align_layers: Whether to apply Procrustes
        save_path: Optional path to save preprocessed data
        
    Returns:
        Tuple of (preprocessed_data, fitted_pipeline)
    """
    logger.info(f"Loading activations from {activations_file}")
    
    # Load raw activations
    with open(activations_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract activation arrays
    activations_by_layer = {}
    for layer_name, layer_data in data.items():
        if layer_name.startswith('layer_') and 'activations' in layer_data:
            activations_by_layer[layer_name] = layer_data['activations']
    
    logger.info(f"Loaded {len(activations_by_layer)} layers")
    
    # Create and fit pipeline
    pipeline = PreprocessingPipeline(n_components, align_layers)
    preprocessed = pipeline.fit_transform(activations_by_layer)
    
    # Save if requested
    if save_path:
        save_data = {
            'preprocessed': preprocessed,
            'pipeline_config': {
                'n_components': n_components,
                'align_layers': align_layers
            }
        }
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"Saved preprocessed data to {save_path}")
    
    return preprocessed, pipeline