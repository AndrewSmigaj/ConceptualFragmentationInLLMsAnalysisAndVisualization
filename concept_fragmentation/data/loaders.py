"""
Dataset loaders for the Concept Fragmentation project.

This module contains functions to load and prepare various datasets:
- Titanic dataset
- Adult Income dataset
- Heart Disease dataset
- Fashion MNIST (binary subset: shirt vs. pullover)
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import seaborn as sns

from concept_fragmentation.config import DATASETS, RANDOM_SEED, TRAINING


class TitanicDataset:
    """Loader for the Titanic dataset."""
    
    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the Titanic dataset from seaborn.
        
        Returns:
            Tuple containing training and test dataframes
        """
        config = DATASETS["titanic"]
        
        # Load Titanic dataset from seaborn
        df = sns.load_dataset("titanic")
        
        # Check if target column exists, if not check lowercase version
        target = config["target"]
        if target not in df.columns and target.lower() in df.columns:
            target = target.lower()
        elif target not in df.columns:
            raise ValueError(f"Target column '{config['target']}' or '{config['target'].lower()}' not found in Titanic dataset")
        
        # Rename columns to match config if needed
        column_mapping = {}
        for col in config["categorical_features"] + config["numerical_features"]:
            if col not in df.columns and col.lower() in df.columns:
                column_mapping[col.lower()] = col
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            
        # Make sure target column matches config
        if target != config["target"]:
            df = df.rename(columns={target: config["target"]})
            target = config["target"]
        
        # Drop specified columns
        if "drop_columns" in config:
            df = df.drop(columns=[col for col in config["drop_columns"] if col in df.columns])
        
        # Split into train and test sets
        train_df, test_df = train_test_split(
            df, 
            test_size=config["test_size"], 
            random_state=RANDOM_SEED,
            stratify=df[target]
        )
        
        return train_df, test_df


class AdultDataset:
    """Loader for the Adult Income dataset."""
    
    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the Adult Income dataset from OpenML.
        
        Returns:
            Tuple containing training and test dataframes
        """
        config = DATASETS["adult"]
        
        # Use fetch_openml to get Adult dataset
        data = fetch_openml(name='adult', version=1, as_frame=True)
        df = data.data
        
        # Handle target column
        target = config["target"]
        df[target] = data.target
        
        # Fix column name mismatches by standardizing format
        # Map between common column name formats
        column_map = {
            # OpenML dataset uses these formats
            'capitalgain': 'capital-gain',
            'capitalloss': 'capital-loss',
            'hoursperweek': 'hours-per-week',
            
            # Add more mappings as needed
        }
        
        # Apply mapping
        df = df.rename(columns=column_map)
        
        # Check and rename feature columns if needed
        column_mapping = {}
        for col in config["categorical_features"] + config["numerical_features"]:
            if col not in df.columns:
                # Check for similar columns with different capitalization/formatting
                candidates = [c for c in df.columns if c.lower().replace('-', '_').replace(' ', '_') == 
                              col.lower().replace('-', '_').replace(' ', '_')]
                if candidates:
                    column_mapping[candidates[0]] = col
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            
        # Pre-process categorical columns before splitting
        # Convert categorical string values to numeric using LabelEncoder
        label_encoders = {}
        for col in config["categorical_features"]:
            if col in df.columns and df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        # Convert target column to numeric if it's categorical
        if df[target].dtype == 'object':
            le = LabelEncoder()
            df[target] = le.fit_transform(df[target].astype(str))
        
        # Split into train and test sets
        train_df, test_df = train_test_split(
            df, 
            test_size=config["test_size"], 
            random_state=RANDOM_SEED,
            stratify=df[target]
        )
        
        return train_df, test_df


class HeartDiseaseDataset:
    """Loader for the Heart Disease dataset."""
    
    @staticmethod
    def load_data(test_size: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the Heart Disease dataset from OpenML.
        
        Args:
            test_size: Fraction of data to use for testing (default: from config)
            
        Returns:
            Tuple containing training and test dataframes
        """
        config = DATASETS["heart"]
        
        # Use fetch_openml to get Heart Disease dataset
        data = fetch_openml(name='heart-statlog', version=1, as_frame=True)
        df = data.data
        
        # Handle target column
        target = config["target"]
        df[target] = data.target
        
        # Column name mapping between the OpenML dataset and our config
        column_map = {
            'resting_blood_pressure': 'trestbps',
            'serum_cholestoral': 'chol',
            'maximum_heart_rate_achieved': 'thalach',
            'chest': 'cp',
            'fasting_blood_sugar': 'fbs',
            'resting_electrocardiographic_results': 'restecg',
            'exercise_induced_angina': 'exang',
            'number_of_major_vessels': 'ca'
        }
        
        # Apply mapping
        df = df.rename(columns=column_map)
        
        # Check and rename feature columns if needed
        column_mapping = {}
        for col in config["categorical_features"] + config["numerical_features"]:
            if col not in df.columns:
                # Check for similar columns with different capitalization/formatting
                candidates = [c for c in df.columns if c.lower().replace('-', '_').replace(' ', '_') == 
                             col.lower().replace('-', '_').replace(' ', '_')]
                if candidates:
                    column_mapping[candidates[0]] = col
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            
        # Pre-process categorical columns before splitting
        # Convert categorical string values to numeric using LabelEncoder
        label_encoders = {}
        for col in config["categorical_features"]:
            if col in df.columns and df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        # Convert target column to numeric if it's categorical
        if df[target].dtype == 'object':
            le = LabelEncoder()
            df[target] = le.fit_transform(df[target].astype(str))
            
        # If test_size is not provided, use the value from config
        if test_size is None:
            test_size = config["test_size"]
        
        # Split into train and test sets
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=RANDOM_SEED,
            stratify=df[target]
        )
        
        return train_df, test_df


class FashionMNISTDataset:
    """Loader for the Fashion MNIST dataset."""
    
    @staticmethod
    def load_data(root: str = "./data") -> Tuple[DataLoader, DataLoader]:
        """
        Load the Fashion MNIST dataset.
        
        Args:
            root: Directory where the data will be stored
            
        Returns:
            Tuple containing training and test DataLoaders
        """
        config = DATASETS["fashion_mnist"]
        
        # Define transformations - make sure to flatten the images
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Load the training data
        train_dataset = datasets.FashionMNIST(
            root=root,
            train=True,
            download=True,
            transform=transform
        )
        
        # Load the test data
        test_dataset = datasets.FashionMNIST(
            root=root,
            train=False,
            download=True,
            transform=transform
        )
        
        # Map class names to indices if needed
        class_to_idx = {
            "T-shirt/top": 0,
            "Trouser": 1, 
            "Pullover": 2, 
            "Dress": 3, 
            "Coat": 4,
            "Sandal": 5, 
            "Shirt": 6, 
            "Sneaker": 7, 
            "Bag": 8, 
            "Ankle boot": 9
        }
        
        # If we need to use a subset of classes
        if "classes" in config and config["classes"]:
            selected_classes = []
            
            # Convert from class names to indices if needed
            for cls in config["classes"]:
                if isinstance(cls, str) and cls in class_to_idx:
                    selected_classes.append(class_to_idx[cls])
                elif isinstance(cls, int) and 0 <= cls < 10:
                    selected_classes.append(cls)
            
            if not selected_classes:
                # Default to first two classes if none were successfully mapped
                selected_classes = [0, 1]
                
            # Filter to include only the selected classes
            train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in selected_classes]
            test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in selected_classes]
            
            if not train_indices or not test_indices:
                raise ValueError(f"No samples found for classes {config['classes']}.")
            
            train_subset = Subset(train_dataset, train_indices)
            test_subset = Subset(test_dataset, test_indices)
            
            batch_size = TRAINING["batch_size"]["fashion_mnist"]
            
            # Create custom Dataset classes that flatten the images
            class FlattenedDataset(Dataset):
                def __init__(self, dataset):
                    self.dataset = dataset
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    img, label = self.dataset[idx]
                    # Flatten the image (28x28 to 784)
                    return img.view(-1), label
            
            # Wrap the subset datasets with our flattening wrappers
            flattened_train = FlattenedDataset(train_subset)
            flattened_test = FlattenedDataset(test_subset)
            
            # Create DataLoaders with flattened datasets
            train_loader = DataLoader(
                flattened_train,
                batch_size=batch_size,
                shuffle=True
            )
            
            test_loader = DataLoader(
                flattened_test,
                batch_size=batch_size,
                shuffle=False
            )
        else:
            # Create DataLoaders for all classes (still flattening the images)
            batch_size = TRAINING["batch_size"]["fashion_mnist"]
            
            # Create custom Dataset classes that flatten the images
            class FlattenedDataset(Dataset):
                def __init__(self, dataset):
                    self.dataset = dataset
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    img, label = self.dataset[idx]
                    # Flatten the image (28x28 to 784)
                    return img.view(-1), label
            
            # Wrap the datasets with our flattening wrappers
            flattened_train = FlattenedDataset(train_dataset)
            flattened_test = FlattenedDataset(test_dataset)
            
            train_loader = DataLoader(
                flattened_train,
                batch_size=batch_size,
                shuffle=True
            )
            
            test_loader = DataLoader(
                flattened_test,
                batch_size=batch_size,
                shuffle=False
            )
        
        return train_loader, test_loader


def get_dataset_loader(dataset_name: str):
    """
    Get the appropriate dataset loader based on the dataset name.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        The corresponding dataset loader class
        
    Raises:
        ValueError: If the dataset name is not recognized
    """
    loaders = {
        "titanic": TitanicDataset,
        "adult": AdultDataset,
        "heart": HeartDiseaseDataset,
        "fashion_mnist": FashionMNISTDataset
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Dataset '{dataset_name}' not recognized. Available datasets: {list(loaders.keys())}")
    
    return loaders[dataset_name]
