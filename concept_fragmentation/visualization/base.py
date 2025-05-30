"""Base classes for visualization components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from pathlib import Path
import logging
from .exceptions import VisualizationError, InvalidDataError

logger = logging.getLogger(__name__)


class BaseVisualizer(ABC):
    """Abstract base class for all visualization components.
    
    This class defines the interface for creating various types of
    visualizations for concept trajectory analysis.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize the visualizer.
        
        Args:
            config: Configuration object specific to the visualizer type
        """
        self.config = config
        self._figure = None
        
    @abstractmethod
    def create_figure(self, data: Dict[str, Any], **kwargs) -> Any:
        """Create visualization figure.
        
        Args:
            data: Input data for visualization
            **kwargs: Additional visualization parameters
            
        Returns:
            Figure object (implementation specific)
            
        Raises:
            VisualizationError: If visualization creation fails
            InvalidDataError: If input data is invalid
        """
        pass
    
    @abstractmethod
    def save_figure(self, 
                   fig: Any, 
                   output_path: Union[str, Path],
                   format: str = 'html',
                   **kwargs) -> None:
        """Save figure to file.
        
        Args:
            fig: Figure object to save
            output_path: Path to save the figure
            format: Output format ('html', 'png', 'pdf', etc.)
            **kwargs: Additional save parameters
            
        Raises:
            VisualizationError: If saving fails
        """
        pass
    
    def validate_data(self, data: Dict[str, Any]) -> None:
        """Validate input data for visualization.
        
        Args:
            data: Data to validate
            
        Raises:
            InvalidDataError: If data is invalid
        """
        if not isinstance(data, dict):
            raise InvalidDataError("Input data must be a dictionary")
            
        # Subclasses should implement specific validation
        self._validate_specific_data(data)
        
    def _validate_specific_data(self, data: Dict[str, Any]) -> None:
        """Validate data specific to this visualizer.
        
        Subclasses should override this method.
        
        Args:
            data: Data to validate
        """
        pass
    
    def create_and_save(self,
                       data: Dict[str, Any],
                       output_path: Union[str, Path],
                       format: str = 'html',
                       **kwargs) -> Any:
        """Create and save figure in one step.
        
        Args:
            data: Input data for visualization
            output_path: Path to save the figure
            format: Output format
            **kwargs: Additional parameters
            
        Returns:
            Created figure object
            
        Raises:
            VisualizationError: If creation or saving fails
        """
        try:
            # Validate data
            self.validate_data(data)
            
            # Create figure
            fig = self.create_figure(data, **kwargs)
            self._figure = fig
            
            # Save figure
            self.save_figure(fig, output_path, format=format, **kwargs)
            
            logger.info(f"Created and saved visualization to {output_path}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(f"Failed to create and save figure: {e}")
    
    @property
    def figure(self) -> Optional[Any]:
        """Return the last created figure."""
        return self._figure
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        if hasattr(self.config, '__dict__'):
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"Unknown config parameter: {key}")
                    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        if hasattr(self.config, '__dict__'):
            return vars(self.config).copy()
        elif isinstance(self.config, dict):
            return self.config.copy()
        else:
            return {}