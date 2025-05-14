"""
Test script to verify dataset loader fixes.
This script tests loading and preprocessing all datasets to ensure they work correctly.
"""

import sys
import os
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import the dataset loaders
from concept_fragmentation.data.loaders import get_dataset_loader
from concept_fragmentation.data.preprocessors import DataPreprocessor
from concept_fragmentation.config import DATASETS

def test_dataset(dataset_name):
    """Test loading and preprocessing a dataset."""
    print(f"\nTesting {dataset_name} dataset:")
    
    # Get the dataset loader
    loader = get_dataset_loader(dataset_name)
    
    try:
        # Load the data
        print(f"  Loading {dataset_name}...")
        if dataset_name == "fashion_mnist":
            # Fashion MNIST returns DataLoaders
            train_loader, test_loader = loader.load_data()
            print(f"  ✓ Successfully loaded dataset")
            
            # Test one batch
            print(f"  Testing a batch of data...")
            sample_batch, sample_labels = next(iter(train_loader))
            print(f"  ✓ Sample batch shape: {sample_batch.shape}")
            print(f"  ✓ Sample labels shape: {sample_labels.shape}")
            print(f"  ✓ Data type: {sample_batch.dtype}")
            
        else:
            # Tabular datasets return DataFrames
            train_df, test_df = loader.load_data()
            print(f"  ✓ Successfully loaded dataset: {len(train_df)} train samples, {len(test_df)} test samples")
            
            # Get dataset configuration
            config = DATASETS[dataset_name]
            
            # Check data types
            print(f"  Checking data types...")
            print(f"  Target column: {config['target']}")
            print(f"  Target dtype: {train_df[config['target']].dtype}")
            
            # Test preprocessing
            print(f"  Testing preprocessing pipeline...")
            preprocessor = DataPreprocessor(
                categorical_cols=config["categorical_features"],
                numerical_cols=config["numerical_features"],
                target_col=config["target"]
            )
            
            # Preprocess the data
            X_train, y_train = preprocessor.fit_transform(train_df)
            X_test, y_test = preprocessor.transform(test_df)
            
            # Convert to PyTorch tensors as a final test
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            
            print(f"  ✓ Preprocessed data shapes: X={X_train_tensor.shape}, y={y_train_tensor.shape}")
            print(f"  ✓ Data types: X={X_train_tensor.dtype}, y={y_train_tensor.dtype}")
        
        print(f"  ✓ ALL TESTS PASSED for {dataset_name}")
        return True
        
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    datasets = ["titanic", "adult", "heart", "fashion_mnist"]
    results = {}
    
    for dataset in datasets:
        results[dataset] = test_dataset(dataset)
    
    # Summary
    print("\n" + "="*50)
    print("DATASET TEST SUMMARY")
    print("="*50)
    all_passed = True
    for dataset, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{dataset}: {status}")
        if not passed:
            all_passed = False
    
    # Exit with the appropriate code
    sys.exit(0 if all_passed else 1) 