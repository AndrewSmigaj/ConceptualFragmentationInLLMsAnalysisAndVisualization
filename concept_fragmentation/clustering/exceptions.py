"""Exceptions for clustering module."""


class ClustererError(Exception):
    """Base exception for clustering errors."""
    pass


class ClusteringNotFittedError(ClustererError):
    """Raised when trying to use an unfitted clusterer."""
    pass


class InvalidDataError(ClustererError):
    """Raised when input data is invalid."""
    pass


class InvalidParameterError(ClustererError):
    """Raised when clustering parameters are invalid."""
    pass