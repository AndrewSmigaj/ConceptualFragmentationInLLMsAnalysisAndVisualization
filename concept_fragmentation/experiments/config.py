"""Configuration management for experiments."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
import json


@dataclass
class ExperimentConfig:
    """Configuration for experiments.
    
    This class defines all parameters needed to run an experiment,
    including model settings, clustering parameters, and output paths.
    """
    
    # Basic settings
    name: str
    description: str = ""
    model: str = "gpt2"
    dataset: str = "tokens"
    
    # Clustering parameters
    k_values: List[int] = field(default_factory=lambda: [10])
    layers: List[int] = field(default_factory=lambda: list(range(12)))
    
    # Data parameters
    n_tokens: Optional[int] = 10000
    token_type: str = "all"  # "all", "frequent", "specific"
    
    # Output settings
    output_dir: str = "./results"
    save_activations: bool = True
    save_visualizations: bool = True
    
    # Random seed for reproducibility
    random_seed: Optional[int] = 42
    
    # Advanced settings
    batch_size: int = 32
    device: str = "cuda"
    use_cache: bool = True
    cache_dir: Optional[str] = "./cache"
    
    # Visualization settings
    viz_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            ExperimentConfig instance
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file.
        
        Args:
            path: Path to JSON configuration file
            
        Returns:
            ExperimentConfig instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
            
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return asdict(self)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
            
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            path: Path to save JSON file
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate k values
        if not self.k_values:
            raise ValueError("k_values cannot be empty")
            
        for k in self.k_values:
            if k < 2:
                raise ValueError(f"k must be >= 2, got {k}")
                
        # Validate layers
        if not self.layers:
            raise ValueError("layers cannot be empty")
            
        # Validate model
        valid_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "bert", "custom"]
        if self.model not in valid_models and not self.model.startswith("custom:"):
            raise ValueError(f"Invalid model: {self.model}")
            
        # Validate paths
        output_path = Path(self.output_dir)
        if not output_path.parent.exists():
            raise ValueError(f"Parent directory of output_dir does not exist: {output_path.parent}")
            
    def get_experiment_id(self) -> str:
        """Generate unique experiment ID.
        
        Returns:
            Experiment ID string
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        k_str = "_".join(map(str, self.k_values))
        return f"{self.name}_k{k_str}_{timestamp}"
        
    def copy(self, **kwargs) -> 'ExperimentConfig':
        """Create a copy of the configuration with updated values.
        
        Args:
            **kwargs: Values to update
            
        Returns:
            New ExperimentConfig instance
        """
        data = self.to_dict()
        data.update(kwargs)
        return ExperimentConfig(**data)