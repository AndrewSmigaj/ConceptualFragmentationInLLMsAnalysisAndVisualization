"""
Create a simple demo model for testing Concept MRI without full training infrastructure.
"""
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import json
from pathlib import Path

class SimpleDemoModel(nn.Module):
    """Simple feedforward network for demo purposes."""
    def __init__(self, input_size=10, hidden_sizes=[32, 16, 8], output_size=2):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
    def forward(self, x):
        return self.model(x)

def create_demo_model_and_data():
    """Create a simple demo model with synthetic data."""
    
    # Create model
    model = SimpleDemoModel(
        input_size=10,
        hidden_sizes=[32, 16, 8],
        output_size=2
    )
    
    # Create synthetic data (100 samples, 10 features)
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Create two clusters in the data
    cluster1 = np.random.randn(n_samples // 2, n_features) - 1
    cluster2 = np.random.randn(n_samples // 2, n_features) + 1
    X = np.vstack([cluster1, cluster2])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Get activations for each layer
    activations = {}
    x = X_tensor
    
    layer_idx = 0
    for i, module in enumerate(model.model):
        x = module(x)
        if isinstance(module, nn.Linear):
            activations[f'layer_{layer_idx}'] = x.detach().numpy()
            layer_idx += 1
    
    # Create model save format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_data = {
        'model_state_dict': model.state_dict(),
        'input_size': model.input_size,
        'output_size': model.output_size,
        'architecture': model.hidden_sizes,
        'activation': 'relu',
        'dropout_rate': 0.1,
        'model_type': 'feedforward',
        'timestamp': timestamp
    }
    
    # Create dataset info
    dataset_info = {
        'name': 'synthetic_demo',
        'num_samples': n_samples,
        'num_features': n_features,
        'feature_names': [f'feature_{i}' for i in range(n_features)],
        'target_names': ['class_0', 'class_1']
    }
    
    # Save files
    demo_dir = Path(__file__).parent / 'synthetic_demo'
    demo_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = demo_dir / f'model_{timestamp}.pt'
    torch.save(model_data, model_path)
    
    # Save dataset
    dataset_path = demo_dir / 'dataset.npz'
    np.savez(dataset_path, X=X, y=y)
    
    # Save dataset info
    info_path = demo_dir / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Save activations (for quick testing)
    activations_path = demo_dir / 'sample_activations.npz'
    np.savez(activations_path, **activations)
    
    print(f"Demo model and data created in {demo_dir}")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Activations: {activations_path}")
    
    return model, X, y, activations

if __name__ == "__main__":
    create_demo_model_and_data()