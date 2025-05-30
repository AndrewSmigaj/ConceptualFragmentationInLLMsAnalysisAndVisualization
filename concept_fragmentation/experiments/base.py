"""Base class for experiments."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """Abstract base class for all experiments.
    
    This class provides a framework for running experiments with
    consistent structure, logging, and result management.
    """
    
    def __init__(self, config: 'ExperimentConfig'):
        """Initialize the experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results = {}
        self.artifacts = {}
        self.start_time = None
        self.end_time = None
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Set up experiment-specific logging."""
        log_file = self.output_dir / f"{self.config.name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
    def run(self) -> Dict[str, Any]:
        """Run the complete experiment pipeline.
        
        Returns:
            Dictionary of experiment results
        """
        logger.info(f"Starting experiment: {self.config.name}")
        self.start_time = datetime.now()
        
        try:
            # Setup phase
            logger.info("Setting up experiment...")
            self.setup()
            
            # Execution phase
            logger.info("Running experiment...")
            self.results = self.execute()
            
            # Analysis phase
            logger.info("Analyzing results...")
            analysis_results = self.analyze()
            self.results['analysis'] = analysis_results
            
            # Visualization phase
            logger.info("Creating visualizations...")
            viz_artifacts = self.visualize()
            self.artifacts['visualizations'] = viz_artifacts
            
            # Cleanup phase
            logger.info("Cleaning up...")
            self.cleanup()
            
            self.end_time = datetime.now()
            duration = self.end_time - self.start_time
            
            logger.info(f"Experiment completed successfully in {duration}")
            
            # Save results
            self._save_results()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            self.end_time = datetime.now()
            self.results['error'] = str(e)
            self._save_results()
            raise
            
    @abstractmethod
    def setup(self) -> None:
        """Set up the experiment.
        
        This method should perform any necessary initialization,
        such as loading data, initializing models, etc.
        """
        pass
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Execute the main experiment logic.
        
        Returns:
            Dictionary of execution results
        """
        pass
    
    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """Analyze experiment results.
        
        Returns:
            Dictionary of analysis results
        """
        pass
    
    @abstractmethod
    def visualize(self) -> Dict[str, str]:
        """Create visualizations.
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up resources.
        
        Override this method if cleanup is needed.
        """
        pass
    
    def _save_results(self) -> None:
        """Save experiment results to file."""
        results_file = self.output_dir / f"{self.config.name}_results.json"
        
        results_data = {
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': str(self.end_time - self.start_time) if self.start_time and self.end_time else None,
            'results': self.results,
            'artifacts': self.artifacts
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
            
        logger.info(f"Results saved to {results_file}")
        
    def save_artifact(self, name: str, data: Any, format: str = 'json') -> str:
        """Save an artifact to file.
        
        Args:
            name: Artifact name
            data: Data to save
            format: Save format ('json', 'npy', 'pkl', etc.)
            
        Returns:
            Path to saved file
        """
        artifact_path = self.output_dir / f"{name}.{format}"
        
        if format == 'json':
            with open(artifact_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == 'npy':
            import numpy as np
            np.save(artifact_path, data)
        elif format == 'pkl':
            import pickle
            with open(artifact_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        self.artifacts[name] = str(artifact_path)
        
        return str(artifact_path)
        
    def load_artifact(self, name: str) -> Any:
        """Load a saved artifact.
        
        Args:
            name: Artifact name
            
        Returns:
            Loaded data
        """
        if name not in self.artifacts:
            raise ValueError(f"Artifact '{name}' not found")
            
        artifact_path = Path(self.artifacts[name])
        format = artifact_path.suffix[1:]  # Remove leading dot
        
        if format == 'json':
            with open(artifact_path, 'r') as f:
                return json.load(f)
        elif format == 'npy':
            import numpy as np
            return np.load(artifact_path)
        elif format == 'pkl':
            import pickle
            with open(artifact_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")