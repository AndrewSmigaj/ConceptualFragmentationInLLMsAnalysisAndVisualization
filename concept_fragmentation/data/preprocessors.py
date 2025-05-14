"""
Data preprocessing utilities for the Concept Fragmentation project.

This module provides functions for:
- One-hot encoding for categorical features
- Normalization for numerical features
- Missing value imputation
- Stratified sampling for class imbalance
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from concept_fragmentation.config import RANDOM_SEED


class DataPreprocessor:
    """Class for preprocessing tabular datasets."""
    
    def __init__(self, 
                categorical_cols: List[str], 
                numerical_cols: List[str],
                target_col: str):
        """
        Initialize the DataPreprocessor with column specifications.
        
        Args:
            categorical_cols: List of categorical column names
            numerical_cols: List of numerical column names
            target_col: Name of the target column
        """
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.target_col = target_col
        
        # Initialize transformers
        self.num_imputer = SimpleImputer(strategy='mean')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # For category type columns
        self.label_encoders = {}
        
        # Storage for fitted transformers
        self.fitted = False
        self.encoded_feature_names = []
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the preprocessing transformers on the given dataframe.
        
        Args:
            df: Input dataframe to fit transformers on
        """
        # Make a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Handle categorical columns that are stored as category dtype
        for col in self.categorical_cols:
            if col in df.columns:
                # Convert category dtype to string
                if pd.api.types.is_categorical_dtype(df[col]):
                    df[col] = df[col].astype(str)
                # Convert object dtype to string 
                elif df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
                    
                # Label-encode string columns for consistent numeric representation
                if df[col].dtype == 'object' or df[col].dtype == 'string':
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
        
        # Handle target column if it's categorical
        if self.target_col in df.columns:
            if pd.api.types.is_categorical_dtype(df[self.target_col]):
                df[self.target_col] = df[self.target_col].astype(str)
                self.label_encoders[self.target_col] = LabelEncoder()
                df[self.target_col] = self.label_encoders[self.target_col].fit_transform(df[self.target_col])
        
        # Fit imputers
        if self.numerical_cols:
            numeric_data = df[self.numerical_cols].copy()
            # Convert any remaining non-numeric columns to numeric
            for col in self.numerical_cols:
                if not pd.api.types.is_numeric_dtype(numeric_data[col]):
                    numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
            self.num_imputer.fit(numeric_data)
        
        if self.categorical_cols:
            categorical_data = df[self.categorical_cols].copy()
            # Ensure all categorical data is numeric at this point
            for col in self.categorical_cols:
                if not pd.api.types.is_numeric_dtype(categorical_data[col]):
                    categorical_data[col] = pd.to_numeric(categorical_data[col], errors='coerce')
            
            self.cat_imputer.fit(categorical_data)
            
            # Impute categorical data for fitting the encoder
            imputed_cat_data = self.cat_imputer.transform(categorical_data)
            imputed_cat_df = pd.DataFrame(
                imputed_cat_data, 
                columns=self.categorical_cols
            )
            
            # Fit the encoder on imputed categorical data
            self.encoder.fit(imputed_cat_df)
            
            # Get the encoded feature names
            self.encoded_feature_names = []
            for i, col in enumerate(self.categorical_cols):
                categories = self.encoder.categories_[i]
                for cat in categories:
                    self.encoded_feature_names.append(f"{col}_{cat}")
        
        # Fit scaler on imputed numerical data
        if self.numerical_cols:
            imputed_num_data = self.num_imputer.transform(numeric_data)
            self.scaler.fit(imputed_num_data)
        
        self.fitted = True
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform the dataframe into preprocessed features and target.
        
        Args:
            df: Input dataframe to transform
            
        Returns:
            Tuple containing (features array, target array)
            
        Raises:
            ValueError: If the preprocessor hasn't been fitted yet
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Make a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Apply the same transformations as in fit
        for col in self.categorical_cols:
            if col in df.columns:
                # Convert category dtype to string
                if pd.api.types.is_categorical_dtype(df[col]):
                    df[col] = df[col].astype(str)
                # Convert object dtype to string 
                elif df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
                
                # Apply label encoding if fitted
                if col in self.label_encoders:
                    # Handle any new categories not seen during training
                    df[col] = df[col].apply(lambda x: x if x in self.label_encoders[col].classes_ else self.label_encoders[col].classes_[0])
                    df[col] = self.label_encoders[col].transform(df[col])
        
        # Handle target column if it's categorical
        if self.target_col in df.columns and self.target_col in self.label_encoders:
            if pd.api.types.is_categorical_dtype(df[self.target_col]):
                df[self.target_col] = df[self.target_col].astype(str)
            # Handle any new categories not seen during training
            df[self.target_col] = df[self.target_col].apply(
                lambda x: x if x in self.label_encoders[self.target_col].classes_ else self.label_encoders[self.target_col].classes_[0]
            )
            df[self.target_col] = self.label_encoders[self.target_col].transform(df[self.target_col])
        
        # Extract target
        y = df[self.target_col].values
        
        # Process numerical features
        X_numerical = np.zeros((df.shape[0], 0))
        if self.numerical_cols:
            # Convert any remaining non-numeric columns to numeric
            numeric_data = df[self.numerical_cols].copy()
            for col in self.numerical_cols:
                if not pd.api.types.is_numeric_dtype(numeric_data[col]):
                    numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
            
            # Impute missing values
            imputed_num_data = self.num_imputer.transform(numeric_data)
            
            # Scale numerical features
            X_numerical = self.scaler.transform(imputed_num_data)
        
        # Process categorical features
        X_categorical = np.zeros((df.shape[0], 0))
        if self.categorical_cols:
            # Ensure all categorical data is numeric at this point
            categorical_data = df[self.categorical_cols].copy()
            for col in self.categorical_cols:
                if not pd.api.types.is_numeric_dtype(categorical_data[col]):
                    categorical_data[col] = pd.to_numeric(categorical_data[col], errors='coerce')
            
            # Impute missing values
            imputed_cat_data = self.cat_imputer.transform(categorical_data)
            imputed_cat_df = pd.DataFrame(
                imputed_cat_data, 
                columns=self.categorical_cols
            )
            
            # One-hot encode categorical features
            X_categorical = self.encoder.transform(imputed_cat_df)
        
        # Combine numerical and categorical features
        X = np.hstack([X_numerical, X_categorical])
        
        return X, y
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor and transform the dataframe.
        
        Args:
            df: Input dataframe to fit and transform
            
        Returns:
            Tuple containing (features array, target array)
        """
        self.fit(df)
        return self.transform(df)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of the features after preprocessing.
        
        Returns:
            List of feature names after preprocessing
            
        Raises:
            ValueError: If the preprocessor hasn't been fitted yet
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before getting feature names")
        
        feature_names = list(self.numerical_cols) + self.encoded_feature_names
        return feature_names


def handle_class_imbalance(X: np.ndarray, 
                          y: np.ndarray,
                          method: str = 'oversample',
                          sampling_strategy: Union[str, Dict, float] = 'auto') -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle class imbalance using oversampling or undersampling.
    
    Args:
        X: Features array
        y: Target array
        method: Method to use ('oversample', 'undersample', or 'both')
        sampling_strategy: Strategy to determine resampling (see imblearn docs)
        
    Returns:
        Tuple containing (resampled features, resampled targets)
        
    Raises:
        ValueError: If method is not recognized
    """
    if method == 'oversample':
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_SEED)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
    elif method == 'undersample':
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_SEED)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
    elif method == 'both':
        # First oversample the minority class, then undersample the majority class
        over_sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_SEED)
        X_over, y_over = over_sampler.fit_resample(X, y)
        
        under_sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=RANDOM_SEED)
        X_resampled, y_resampled = under_sampler.fit_resample(X_over, y_over)
        
    else:
        raise ValueError(f"Method {method} not recognized. Use 'oversample', 'undersample', or 'both'")
    
    return X_resampled, y_resampled


def stratified_split(X: np.ndarray, 
                    y: np.ndarray, 
                    test_size: float = 0.2,
                    val_size: Optional[float] = None) -> Tuple[np.ndarray, ...]:
    """
    Perform stratified train/test or train/val/test split.
    
    Args:
        X: Features array
        y: Target array
        test_size: Proportion of data for testing
        val_size: Proportion of data for validation (if None, performs train/test split)
        
    Returns:
        Tuple containing split datasets (X_train, X_test, y_train, y_test) or
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if val_size is None:
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
        )
        return X_train, X_test, y_train, y_test
    else:
        # Train/val/test split
        # First, split into train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
        )
        
        # Then split train+val into train and val
        # Calculate the validation size relative to the train+val set
        relative_val_size = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, 
            test_size=relative_val_size, 
            random_state=RANDOM_SEED, 
            stratify=y_trainval
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
