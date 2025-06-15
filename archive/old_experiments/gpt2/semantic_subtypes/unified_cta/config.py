"""
Configuration for unified CTA pipeline.
Uses existing configuration patterns from concept_fragmentation.config
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import sys

# Add parent directories to path for imports
root_dir = Path(__file__).parent.parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import existing config patterns - no reimplementation
from concept_fragmentation.config.config_classes import MetricsConfig
from concept_fragmentation.config.config_manager import ConfigManager

from logging_config import setup_logging

logger = setup_logging(__name__)


@dataclass
class LayerSpecificConfig:
    """Configuration that varies by layer depth."""
    k_range: Tuple[int, int] = (2, 8)  # (min_k, max_k) for gap statistic
    ets_percentile: float = 70.0  # ETS threshold percentile
    coverage_target: float = 0.7  # Micro-clustering coverage target
    purity_target: float = 0.6   # Micro-clustering purity target
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'k_range': self.k_range,
            'ets_percentile': self.ets_percentile,
            'coverage_target': self.coverage_target,
            'purity_target': self.purity_target
        }


@dataclass
class PreprocessingConfig:
    """Preprocessing pipeline configuration."""
    standardize: bool = True
    pca_components: int = 128
    apply_procrustes: bool = True
    procrustes_scaling: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'standardize': self.standardize,
            'pca_components': self.pca_components,
            'apply_procrustes': self.apply_procrustes,
            'procrustes_scaling': self.procrustes_scaling
        }


@dataclass
class QualityConfig:
    """Quality thresholds and validation settings."""
    min_silhouette_score: float = 0.3
    min_coverage: float = 0.7
    min_purity: float = 0.6
    max_cluster_size_ratio: float = 10.0
    preprocessing_score_threshold: float = 0.9
    
    def to_dict(self) -> Dict:
        return {
            'min_silhouette_score': self.min_silhouette_score,
            'min_coverage': self.min_coverage,
            'min_purity': self.min_purity,
            'max_cluster_size_ratio': self.max_cluster_size_ratio,
            'preprocessing_score_threshold': self.preprocessing_score_threshold
        }


@dataclass
class UnifiedCTAConfig:
    """
    Main configuration for unified CTA pipeline.
    Follows existing config patterns from concept_fragmentation.
    """
    
    # Data paths
    activations_path: str = "../activations_by_layer.pkl"
    word_list_path: str = "../data/gpt2_semantic_subtypes_curated.json"
    output_dir: str = "results/unified_cta_config"
    
    # Layer configurations (12 layers for GPT-2)
    layer_configs: Dict[int, LayerSpecificConfig] = field(default_factory=dict)
    
    # Pipeline stages
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    
    # Pipeline control
    layers_to_process: List[int] = field(default_factory=lambda: list(range(12)))
    enable_micro_clustering: bool = True
    enable_path_analysis: bool = True
    enable_interpretability: bool = True
    enable_visualization: bool = True
    
    # Misc settings
    random_seed: int = 42
    max_words_for_viz: int = 100
    save_intermediate: bool = True
    
    def __post_init__(self):
        """Initialize layer-specific configs with smart defaults."""
        if not self.layer_configs:
            # Create layer-specific configs based on depth
            for layer in range(12):
                if layer <= 2:  # Early layers
                    config = LayerSpecificConfig(
                        k_range=(2, 6),
                        ets_percentile=80.0,  # More generous for simple features
                        coverage_target=0.8,
                        purity_target=0.7
                    )
                elif layer <= 5:  # Middle layers
                    config = LayerSpecificConfig(
                        k_range=(3, 8),
                        ets_percentile=70.0,
                        coverage_target=0.7,
                        purity_target=0.6
                    )
                elif layer <= 8:  # Late-middle layers
                    config = LayerSpecificConfig(
                        k_range=(4, 10),
                        ets_percentile=65.0,  # Stricter for complex semantics
                        coverage_target=0.6,
                        purity_target=0.5
                    )
                else:  # Final layers
                    config = LayerSpecificConfig(
                        k_range=(3, 8),
                        ets_percentile=70.0,
                        coverage_target=0.7,
                        purity_target=0.6
                    )
                
                self.layer_configs[layer] = config
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization following existing patterns."""
        return {
            'activations_path': self.activations_path,
            'word_list_path': self.word_list_path,
            'output_dir': self.output_dir,
            'layer_configs': {
                str(layer): config.to_dict() 
                for layer, config in self.layer_configs.items()
            },
            'preprocessing': self.preprocessing.to_dict(),
            'quality': self.quality.to_dict(),
            'layers_to_process': self.layers_to_process,
            'enable_micro_clustering': self.enable_micro_clustering,
            'enable_path_analysis': self.enable_path_analysis,
            'enable_interpretability': self.enable_interpretability,
            'enable_visualization': self.enable_visualization,
            'random_seed': self.random_seed,
            'max_words_for_viz': self.max_words_for_viz,
            'save_intermediate': self.save_intermediate
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UnifiedCTAConfig':
        """Create config from dictionary following existing patterns."""
        # Extract layer configs
        layer_configs = {}
        if 'layer_configs' in data:
            for layer_str, config_data in data['layer_configs'].items():
                layer = int(layer_str)
                layer_configs[layer] = LayerSpecificConfig(**config_data)
        
        # Extract other configs
        preprocessing = PreprocessingConfig()
        if 'preprocessing' in data:
            preprocessing = PreprocessingConfig(**data['preprocessing'])
        
        quality = QualityConfig()
        if 'quality' in data:
            quality = QualityConfig(**data['quality'])
        
        # Create main config
        config_data = data.copy()
        config_data.pop('layer_configs', None)
        config_data.pop('preprocessing', None)
        config_data.pop('quality', None)
        
        config = cls(
            layer_configs=layer_configs,
            preprocessing=preprocessing,
            quality=quality,
            **config_data
        )
        
        return config
    
    def save(self, path: Path):
        """Save configuration to JSON file following existing patterns."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved configuration to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'UnifiedCTAConfig':
        """Load configuration from JSON file following existing patterns."""
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls.from_dict(data)
        logger.info(f"Loaded configuration from {path}")
        return config
    
    def get_layer_config(self, layer: int) -> LayerSpecificConfig:
        """Get configuration for specific layer."""
        if layer in self.layer_configs:
            return self.layer_configs[layer]
        else:
            # Fallback to default
            logger.warning(f"No config for layer {layer}, using default")
            return LayerSpecificConfig()


def create_default_config() -> UnifiedCTAConfig:
    """
    Factory function for default configuration.
    Follows existing pattern from concept_fragmentation.config.defaults
    """
    return UnifiedCTAConfig()


def create_config_for_experiment(experiment_type: str = "full") -> UnifiedCTAConfig:
    """
    Create configuration for specific experiment type.
    Follows existing factory pattern.
    """
    base_config = create_default_config()
    
    if experiment_type == "quick_test":
        # Fast configuration for testing
        base_config.layers_to_process = [0, 5, 11]  # Just 3 layers
        base_config.max_words_for_viz = 50
        base_config.save_intermediate = False
        
        # Looser quality thresholds for testing
        base_config.quality.min_silhouette_score = 0.2
        base_config.quality.min_coverage = 0.5
        
    elif experiment_type == "high_quality":
        # Strict configuration for publication
        base_config.quality.min_silhouette_score = 0.4
        base_config.quality.min_coverage = 0.8
        base_config.quality.min_purity = 0.7
        
        # More conservative clustering
        for layer_config in base_config.layer_configs.values():
            layer_config.ets_percentile = 60.0  # Stricter thresholds
            layer_config.coverage_target = 0.8
            layer_config.purity_target = 0.7
    
    return base_config