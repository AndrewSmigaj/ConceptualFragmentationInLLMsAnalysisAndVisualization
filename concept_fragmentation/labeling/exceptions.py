"""Exceptions for labeling module."""


class LabelerError(Exception):
    """Base exception for labeling errors."""
    pass


class LabelingError(LabelerError):
    """Raised when labeling process fails."""
    pass


class InvalidClusterDataError(LabelerError):
    """Raised when cluster data is invalid or incomplete."""
    pass


class ConsistencyError(LabelerError):
    """Raised when label consistency requirements are not met."""
    pass