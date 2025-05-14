"""
Test script for dataset loaders.

This script tests loading all datasets to ensure they work correctly.
"""

import os
import logging
import torch
import numpy as np
import pandas as pd
from concept_fragmentation.data.loaders import (
    TitanicDataset,
    AdultDataset,
    HeartDiseaseDataset,
    FashionMNISTDataset
)
from concept_fragmentation.data.preprocessors import DataPreprocessor
from concept_fragmentation.config import DATASETS, RESULTS_DIR

# Configure logging
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "test_datasets.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_titanic():
    """Test loading the Titanic dataset."""
    logger.info("Testing Titanic dataset...")
    
    try:
        # Load data
        train_df, test_df = TitanicDataset.load_data()
        
        logger.info(f"Titanic dataset loaded: {len(train_df)} train samples, {len(test_df)} test samples")
        logger.info(f"Columns: {train_df.columns.tolist()}")
        
        # Test preprocessor
        config = DATASETS["titanic"]
        preprocessor = DataPreprocessor(
            categorical_cols=config["categorical_features"],
            numerical_cols=config["numerical_features"],
            target_col=config["target"]
        )
        
        # Preprocess the data
        X_train, y_train = preprocessor.fit_transform(train_df)
        X_test, y_test = preprocessor.transform(test_df)
        
        logger.info(f"Preprocessed data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f"Preprocessed data shapes: X_test={X_test.shape}, y_test={y_test.shape}")
        logger.info("Titanic dataset test completed successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error testing Titanic dataset: {str(e)}")
        return False

def test_adult():
    """Test loading the Adult dataset."""
    logger.info("Testing Adult dataset...")
    
    try:
        # Load data
        train_df, test_df = AdultDataset.load_data()
        
        logger.info(f"Adult dataset loaded: {len(train_df)} train samples, {len(test_df)} test samples")
        logger.info(f"Columns: {train_df.columns.tolist()}")
        
        # Test preprocessor
        config = DATASETS["adult"]
        preprocessor = DataPreprocessor(
            categorical_cols=config["categorical_features"],
            numerical_cols=config["numerical_features"],
            target_col=config["target"]
        )
        
        # Preprocess the data
        X_train, y_train = preprocessor.fit_transform(train_df)
        X_test, y_test = preprocessor.transform(test_df)
        
        logger.info(f"Preprocessed data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f"Preprocessed data shapes: X_test={X_test.shape}, y_test={y_test.shape}")
        logger.info("Adult dataset test completed successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error testing Adult dataset: {str(e)}")
        return False

def test_heart():
    """Test loading the Heart Disease dataset."""
    logger.info("Testing Heart Disease dataset...")
    
    try:
        # Load data
        train_df, test_df = HeartDiseaseDataset.load_data()
        
        logger.info(f"Heart Disease dataset loaded: {len(train_df)} train samples, {len(test_df)} test samples")
        logger.info(f"Columns: {train_df.columns.tolist()}")
        
        # Test preprocessor
        config = DATASETS["heart"]
        preprocessor = DataPreprocessor(
            categorical_cols=config["categorical_features"],
            numerical_cols=config["numerical_features"],
            target_col=config["target"]
        )
        
        # Preprocess the data
        X_train, y_train = preprocessor.fit_transform(train_df)
        X_test, y_test = preprocessor.transform(test_df)
        
        logger.info(f"Preprocessed data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f"Preprocessed data shapes: X_test={X_test.shape}, y_test={y_test.shape}")
        logger.info("Heart Disease dataset test completed successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error testing Heart Disease dataset: {str(e)}")
        return False

def test_fashion_mnist():
    """Test loading the Fashion MNIST dataset."""
    logger.info("Testing Fashion MNIST dataset...")
    
    try:
        # Load data
        train_loader, test_loader = FashionMNISTDataset.load_data()
        
        # Get one batch from each loader
        X_train, y_train = next(iter(train_loader))
        X_test, y_test = next(iter(test_loader))
        
        logger.info(f"Fashion MNIST loaders created successfully")
        logger.info(f"Train batch shape: {X_train.shape}, {y_train.shape}")
        logger.info(f"Test batch shape: {X_test.shape}, {y_test.shape}")
        logger.info("Fashion MNIST dataset test completed successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error testing Fashion MNIST dataset: {str(e)}")
        return False

def test_all_datasets():
    """Test all datasets."""
    datasets = {
        "titanic": test_titanic,
        "adult": test_adult,
        "heart": test_heart,
        "fashion_mnist": test_fashion_mnist
    }
    
    results = {}
    
    for name, test_func in datasets.items():
        logger.info(f"Testing {name} dataset...")
        success = test_func()
        results[name] = "SUCCESS" if success else "FAILED"
    
    # Print summary
    logger.info("=== Dataset Test Results ===")
    for name, result in results.items():
        logger.info(f"{name}: {result}")

if __name__ == "__main__":
    test_all_datasets() 