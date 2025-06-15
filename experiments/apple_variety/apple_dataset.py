"""
Apple variety dataset loader for CTA experiments.

This module provides a dataset loader for apple variety data following
the pattern established in concept_fragmentation.data.loaders.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class AppleVarietyDataset:
    """Loader for the Apple Variety dataset."""
    
    def __init__(self, data_path: str, n_top_varieties: int = 10):
        """
        Initialize the Apple Variety dataset loader.
        
        Args:
            data_path: Path to the apple data CSV file
            n_top_varieties: Number of top varieties to include in analysis
        """
        self.data_path = Path(data_path)
        self.n_top_varieties = n_top_varieties
        self.variety_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'variety'
        
    def load_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the Apple Variety dataset from CSV.
        
        Args:
            test_size: Fraction of data to use for testing (default: 0.2)
            
        Returns:
            Tuple containing training and test dataframes
        """
        # Load the processed apple data
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        
        # Filter to top N varieties by count
        variety_counts = df['variety'].value_counts()
        top_varieties = variety_counts.head(self.n_top_varieties).index.tolist()
        df_filtered = df[df['variety'].isin(top_varieties)].copy()
        
        print(f"Selected {self.n_top_varieties} varieties with {len(df_filtered)} total samples")
        print(f"Varieties: {', '.join(top_varieties)}")
        
        # Define feature columns
        self.feature_columns = [
            'brix_numeric',      # Sugar content
            'firmness_numeric',  # Texture measurement
            'red_pct_numeric',   # Color percentage
            'size_numeric',      # Size score (1-5)
            'season_numeric',    # Harvest timing (1-3)
            'starch_numeric'     # Maturity indicator
        ]
        
        # Check which features are available
        available_features = [col for col in self.feature_columns if col in df_filtered.columns]
        missing_features = [col for col in self.feature_columns if col not in df_filtered.columns]
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            self.feature_columns = available_features
            
        # Add derived features if base features exist
        if 'brix_numeric' in df_filtered.columns:
            # Sweetness to acidity ratio (using assumed acidity of 3.2)
            df_filtered['sweetness_ratio'] = df_filtered['brix_numeric'] / 3.2
            self.feature_columns.append('sweetness_ratio')
            
        if all(col in df_filtered.columns for col in ['brix_numeric', 'firmness_numeric', 'size_numeric']):
            # Quality index combining key features
            df_filtered['quality_index'] = (
                df_filtered['brix_numeric'] * 0.4 + 
                df_filtered['firmness_numeric'] * 10 * 0.3 +  # Scale firmness to similar range
                df_filtered['size_numeric'] * 4 * 0.3  # Scale size to similar range
            )
            self.feature_columns.append('quality_index')
        
        # Handle missing values
        # For numeric features, use variety-specific median
        for col in self.feature_columns:
            if col in df_filtered.columns:
                df_filtered[col] = df_filtered.groupby('variety')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # If still missing (e.g., single sample varieties), use global median
                df_filtered[col] = df_filtered[col].fillna(df_filtered[col].median())
        
        # Encode variety labels
        df_filtered['variety_encoded'] = self.variety_encoder.fit_transform(df_filtered['variety'])
        
        # Create variety name mapping for later use
        self.variety_mapping = dict(enumerate(self.variety_encoder.classes_))
        self.variety_reverse_mapping = {v: k for k, v in self.variety_mapping.items()}
        
        # Split into train and test sets
        train_df, test_df = train_test_split(
            df_filtered,
            test_size=test_size,
            random_state=42,
            stratify=df_filtered['variety_encoded']
        )
        
        print(f"Train set: {len(train_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        
        return train_df, test_df
    
    def get_features_and_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract feature matrix and labels from dataframe.
        
        Args:
            df: DataFrame with apple data
            
        Returns:
            Tuple of (features, labels, variety_names)
        """
        # Get features
        X = df[self.feature_columns].values
        
        # Get labels (encoded variety)
        y = df['variety_encoded'].values
        
        # Get variety names for each sample
        variety_names = df['variety'].tolist()
        
        return X, y, variety_names
    
    def get_variety_info(self) -> Dict[int, str]:
        """
        Get mapping of variety codes to names.
        
        Returns:
            Dictionary mapping variety code to variety name
        """
        return self.variety_mapping
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature column names.
        
        Returns:
            List of feature names
        """
        return self.feature_columns