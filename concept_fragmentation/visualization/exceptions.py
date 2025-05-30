"""Exceptions for visualization module."""


class VisualizationError(Exception):
    """Base exception for visualization errors."""
    pass


class InvalidDataError(VisualizationError):
    """Raised when input data for visualization is invalid."""
    pass


class ConfigurationError(VisualizationError):
    """Raised when visualization configuration is invalid."""
    pass


class RenderingError(VisualizationError):
    """Raised when rendering the visualization fails."""
    pass