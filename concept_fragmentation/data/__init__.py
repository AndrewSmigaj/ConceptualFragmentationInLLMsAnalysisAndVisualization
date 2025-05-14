"""
Data module for the Concept Fragmentation project.

This module provides tools for loading and preprocessing datasets.
"""

from concept_fragmentation.data.loaders import (
    TitanicDataset,
    AdultDataset,
    HeartDiseaseDataset,
    FashionMNISTDataset,
    get_dataset_loader
)

from concept_fragmentation.data.preprocessors import (
    DataPreprocessor,
    handle_class_imbalance,
    stratified_split
)

__all__ = [
    'TitanicDataset',
    'AdultDataset',
    'HeartDiseaseDataset',
    'FashionMNISTDataset',
    'get_dataset_loader',
    'DataPreprocessor',
    'handle_class_imbalance',
    'stratified_split'
]
