"""
Base model trainer class that reuses existing data infrastructure.
"""
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import existing data infrastructure
from concept_fragmentation.data import (
    TitanicDataset, HeartDiseaseDataset, DataPreprocessor
)
from concept_fragmentation.utils.helpers import set_random_seed


class ModelTrainer:
    """
    Base trainer class that leverages existing data infrastructure.
    """
    
    def __init__(self, dataset_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trainer with dataset and optional configuration.
        
        Args:
            dataset_name: Name of dataset ('titanic' or 'heart_disease')
            config: Optional configuration dictionary
        """
        self.dataset_name = dataset_name
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load data using existing infrastructure
        self._load_data()
        
    def _load_data(self):
        """Load data using existing dataset classes."""
        # Get appropriate dataset loader
        if self.dataset_name == 'titanic':
            dataset_loader = TitanicDataset()
        elif self.dataset_name == 'heart_disease':
            dataset_loader = HeartDiseaseDataset()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Load raw data
        (X_train, y_train), (X_test, y_test) = dataset_loader.load_data()
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        self.X_train, self.X_test = preprocessor.fit_transform(X_train, X_test)
        
        # Convert to tensors
        self.X_train = torch.FloatTensor(self.X_train)
        self.y_train = torch.LongTensor(y_train)
        self.X_test = torch.FloatTensor(self.X_test)
        self.y_test = torch.LongTensor(y_test)
        
        # Store dimensions
        self.input_size = self.X_train.shape[1]
        self.output_size = len(torch.unique(self.y_train))
        
        # Store feature names if available
        self.feature_names = getattr(preprocessor, 'feature_names_', None)
        
        print(f"Loaded {self.dataset_name} dataset:")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Test samples: {len(self.X_test)}")
        print(f"  Input features: {self.input_size}")
        print(f"  Output classes: {self.output_size}")
        
    def create_data_loaders(self, batch_size: int) -> Tuple[torch.utils.data.DataLoader, 
                                                           torch.utils.data.DataLoader]:
        """Create PyTorch DataLoaders for training."""
        train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        test_dataset = torch.utils.data.TensorDataset(self.X_test, self.y_test)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, test_loader
    
    def train_model(self, model: nn.Module, params: Dict[str, Any], 
                   verbose: bool = True) -> Dict[str, Any]:
        """
        Train a model with given parameters.
        
        Args:
            model: PyTorch model to train
            params: Training parameters (lr, batch_size, epochs, etc.)
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history and metrics
        """
        # Set seed for reproducibility
        if 'seed' in params:
            set_random_seed(params['seed'])
        
        # Move model to device
        model = model.to(self.device)
        
        # Create optimizer
        optimizer_name = params.get('optimizer', 'adam')
        lr = params.get('lr', 0.001)
        weight_decay = params.get('weight_decay', 0.0)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, 
                                momentum=params.get('momentum', 0.9))
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Create data loaders
        batch_size = params.get('batch_size', 32)
        train_loader, test_loader = self.create_data_loaders(batch_size)
        
        # Training loop
        epochs = params.get('epochs', 50)
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        best_val_acc = 0.0
        best_model_state = None
        patience = params.get('early_stopping_patience', 10)
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(batch_y.cpu().numpy())
            
            # Calculate training metrics
            train_loss /= len(train_loader.dataset)
            train_acc = accuracy_score(train_targets, train_preds)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item() * batch_X.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            # Calculate validation metrics
            val_loss /= len(test_loader.dataset)
            val_acc = accuracy_score(val_targets, val_preds)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping check
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            # Full dataset predictions
            X_test_device = self.X_test.to(self.device)
            outputs = model(X_test_device)
            _, final_preds = torch.max(outputs, 1)
            final_preds = final_preds.cpu().numpy()
            
            final_acc = accuracy_score(self.y_test.numpy(), final_preds)
            
        # Return results
        results = {
            'history': history,
            'best_val_acc': best_val_acc,
            'final_test_acc': final_acc,
            'epochs_trained': len(history['train_loss']),
            'model_state': model.state_dict(),
            'classification_report': classification_report(
                self.y_test.numpy(), final_preds, 
                output_dict=True
            )
        }
        
        return results
    
    def save_model_for_concept_mri(self, model: nn.Module, results: Dict[str, Any],
                                  params: Dict[str, Any], variant_name: str):
        """
        Save model in format expected by Concept MRI.
        
        Args:
            model: Trained model
            results: Training results
            params: Model/training parameters
            variant_name: Name of model variant (e.g., 'optimal', 'bottleneck')
        """
        output_dir = Path(__file__).parent.parent.parent / 'concept_mri' / 'demos' / self.dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_path = output_dir / f'model_{variant_name}.pt'
        torch.save({
            'model_state_dict': results['model_state'],
            'input_size': self.input_size,
            'output_size': self.output_size,
            'architecture': params.get('hidden_sizes', []),
            'activation': params.get('activation', 'relu'),
            'dropout_rate': params.get('dropout', 0.0)
        }, model_path)
        
        # Save metadata
        metadata = {
            'dataset': self.dataset_name,
            'variant': variant_name,
            'input_features': self.input_size,
            'output_classes': self.output_size,
            'feature_names': self.feature_names,
            'architecture': {
                'hidden_sizes': params.get('hidden_sizes', []),
                'activation': params.get('activation', 'relu'),
                'dropout_rate': params.get('dropout', 0.0)
            },
            'training': {
                'epochs': results['epochs_trained'],
                'batch_size': params.get('batch_size', 32),
                'learning_rate': params.get('lr', 0.001),
                'optimizer': params.get('optimizer', 'adam'),
                'early_stopping': params.get('early_stopping_patience', 10) > 0
            },
            'performance': {
                'best_val_acc': float(results['best_val_acc']),
                'final_test_acc': float(results['final_test_acc']),
                'classification_report': results['classification_report']
            }
        }
        
        metadata_path = output_dir / f'metadata_{variant_name}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved model to {model_path}")
        print(f"Saved metadata to {metadata_path}")