"""
Model interface for Concept MRI.
Wraps existing activation collection infrastructure.
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
import json

# Add parent directory to path to import concept_fragmentation
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concept_fragmentation.activation.collector import ActivationCollector

class ModelInterface:
    """
    Interface for loading and analyzing neural network models.
    Wraps existing ActivationCollector for UI integration.
    """
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.activation_collector = None
        self.layer_info = []
        
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load a neural network model and extract architecture information.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Dictionary containing model information
        """
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Determine model type by extension
        extension = path.suffix.lower()
        
        if extension in ['.pt', '.pth']:
            return self._load_pytorch_model(model_path)
        elif extension == '.onnx':
            return self._load_onnx_model(model_path)
        elif extension in ['.h5', '.keras']:
            return self._load_tensorflow_model(model_path)
        elif extension == '.pkl':
            return self._load_pickle_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {extension}")
    
    def _load_pytorch_model(self, model_path: str) -> Dict[str, Any]:
        """Load a PyTorch model."""
        try:
            # Load the model or checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model_type = 'pytorch'
            
            # Check if it's a demo model format (with architecture info)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # This is our demo model format
                return self._load_demo_model(model_path, checkpoint)
            
            # Otherwise, treat as regular PyTorch model
            self.model = checkpoint
            
            # If it's a state dict, we need the model architecture
            if isinstance(self.model, dict):
                raise ValueError(
                    "Model file contains only state dict. "
                    "Please provide the complete model with architecture."
                )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Extract layer information
            self.layer_info = self._extract_pytorch_layers(self.model)
            
            # Create activation collector
            self.activation_collector = ActivationCollector(
                model=self.model,
                layers=None  # Will be set based on UI selection
            )
            
            # Calculate model statistics
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                'type': 'PyTorch',
                'num_layers': len(self.layer_info),
                'num_params': total_params,
                'trainable_params': trainable_params,
                'layers': self.layer_info,
                'input_shape': self._infer_input_shape(),
                'output_shape': self._infer_output_shape()
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {str(e)}")
    
    def _extract_pytorch_layers(self, model: torch.nn.Module) -> List[Dict[str, Any]]:
        """Extract layer information from PyTorch model."""
        layers = []
        layer_idx = 0
        
        def extract_layers_recursive(module, prefix=''):
            nonlocal layer_idx
            
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check if it's a leaf module (actual layer)
                if len(list(child.children())) == 0:
                    # Only track layers that produce activations
                    if isinstance(child, (
                        torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d,
                        torch.nn.LSTM, torch.nn.GRU, torch.nn.TransformerEncoderLayer
                    )):
                        layer_info = {
                            'index': layer_idx,
                            'name': full_name,
                            'type': child.__class__.__name__,
                            'params': sum(p.numel() for p in child.parameters()),
                            'shape': self._get_layer_shape(child)
                        }
                        layers.append(layer_info)
                        layer_idx += 1
                else:
                    # Recurse into container modules
                    extract_layers_recursive(child, full_name)
        
        extract_layers_recursive(model)
        return layers
    
    def _get_layer_shape(self, layer: torch.nn.Module) -> str:
        """Get human-readable shape description for a layer."""
        if isinstance(layer, torch.nn.Linear):
            return f"({layer.in_features}, {layer.out_features})"
        elif isinstance(layer, torch.nn.Conv2d):
            return f"({layer.in_channels}, {layer.out_channels}, {layer.kernel_size})"
        elif isinstance(layer, torch.nn.Conv1d):
            return f"({layer.in_channels}, {layer.out_channels}, {layer.kernel_size})"
        elif isinstance(layer, (torch.nn.LSTM, torch.nn.GRU)):
            return f"({layer.input_size}, {layer.hidden_size})"
        else:
            return "Unknown"
    
    def _infer_input_shape(self) -> Optional[Tuple]:
        """Infer input shape from first layer."""
        if not self.layer_info:
            return None
            
        first_layer = self.layer_info[0]
        if 'shape' in first_layer:
            # Extract first dimension from shape string
            shape_str = first_layer['shape']
            if shape_str != "Unknown":
                # Parse shape like "(13, 64)" -> 13
                try:
                    input_dim = int(shape_str.split('(')[1].split(',')[0])
                    return (input_dim,)
                except:
                    pass
        return None
    
    def _infer_output_shape(self) -> Optional[Tuple]:
        """Infer output shape from last layer."""
        if not self.layer_info:
            return None
            
        last_layer = self.layer_info[-1]
        if 'shape' in last_layer:
            # Extract last dimension from shape string
            shape_str = last_layer['shape']
            if shape_str != "Unknown":
                # Parse shape like "(8, 2)" -> 2
                try:
                    output_dim = int(shape_str.split(',')[-1].split(')')[0])
                    return (output_dim,)
                except:
                    pass
        return None
    
    def _load_onnx_model(self, model_path: str) -> Dict[str, Any]:
        """Load an ONNX model."""
        # Placeholder for ONNX support
        raise NotImplementedError("ONNX model support coming soon")
    
    def _load_tensorflow_model(self, model_path: str) -> Dict[str, Any]:
        """Load a TensorFlow/Keras model."""
        # Placeholder for TensorFlow support
        raise NotImplementedError("TensorFlow model support coming soon")
    
    def _load_pickle_model(self, model_path: str) -> Dict[str, Any]:
        """Load a pickled model."""
        import pickle
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Try to determine model type
            if hasattr(self.model, 'forward'):
                # Likely PyTorch
                return self._load_pytorch_model(model_path)
            else:
                # Generic sklearn-style model
                return {
                    'type': 'Scikit-learn',
                    'num_layers': 1,  # Treat as single layer
                    'layers': [{
                        'index': 0,
                        'name': 'model',
                        'type': self.model.__class__.__name__,
                        'shape': 'N/A'
                    }]
                }
        except Exception as e:
            raise RuntimeError(f"Failed to load pickled model: {str(e)}")
    
    def extract_activations(self, data: np.ndarray, layer_indices: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """
        Extract activations for specified layers using the data.
        
        Args:
            data: Input data as numpy array
            layer_indices: List of layer indices to extract (None = all layers)
            
        Returns:
            Dictionary mapping layer index to activations
        """
        if self.activation_collector is None:
            raise RuntimeError("No model loaded")
        
        # Convert numpy to appropriate format
        if self.model_type == 'pytorch':
            data_tensor = torch.FloatTensor(data)
            
            # Use existing ActivationCollector
            # This would be configured based on selected layers
            if layer_indices:
                # Set up hooks for specific layers
                layer_names = [self.layer_info[i]['name'] for i in layer_indices]
                self.activation_collector.set_layers(layer_names)
            
            # Collect activations
            activations = self.activation_collector.collect(data_tensor)
            
            # Convert to numpy and organize by layer index
            result = {}
            for i, (name, acts) in enumerate(activations.items()):
                if isinstance(acts, torch.Tensor):
                    result[i] = acts.cpu().numpy()
                else:
                    result[i] = acts
                    
            return result
        else:
            raise NotImplementedError(f"Activation extraction not implemented for {self.model_type}")
    
    def get_layer_names(self) -> List[str]:
        """Get list of layer names."""
        return [layer['name'] for layer in self.layer_info]
    
    def get_layer_info(self) -> List[Dict[str, Any]]:
        """Get detailed layer information."""
        return self.layer_info
    
    def _load_demo_model(self, model_path: str, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Load a demo model with saved architecture information."""
        # Import model architectures from training scripts
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts' / 'prepare_demo_models'))
        from model_architectures import OptimizableFFNet
        
        # Extract architecture information
        input_size = checkpoint['input_size']
        output_size = checkpoint['output_size']
        hidden_sizes = checkpoint['architecture']
        activation = checkpoint.get('activation', 'relu')
        dropout_rate = checkpoint.get('dropout_rate', 0.0)
        
        # Create model with saved architecture
        self.model = OptimizableFFNet(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Extract layer information
        self.layer_info = self._extract_pytorch_layers(self.model)
        
        # Create activation collector
        self.activation_collector = ActivationCollector(
            model=self.model,
            layers=None
        )
        
        # Calculate model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Load metadata if available
        metadata = self._load_demo_metadata(model_path)
        
        return {
            'type': 'PyTorch (Demo Model)',
            'num_layers': len(self.layer_info),
            'num_params': total_params,
            'trainable_params': trainable_params,
            'layers': self.layer_info,
            'input_shape': (input_size,),
            'output_shape': (output_size,),
            'architecture': hidden_sizes,
            'activation': activation,
            'metadata': metadata
        }
    
    def _load_demo_metadata(self, model_path: str) -> Optional[Dict[str, Any]]:
        """Load metadata for demo model if available."""
        # Look for metadata file with same base name
        model_path = Path(model_path)
        metadata_path = model_path.parent / f"metadata_{model_path.stem.replace('model_', '')}.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    @staticmethod
    def list_demo_models() -> Dict[str, List[Dict[str, Any]]]:
        """List all available demo models."""
        demos_dir = Path(__file__).parent.parent.parent / 'concept_mri' / 'demos'
        demo_models = {}
        
        if demos_dir.exists():
            for dataset_dir in demos_dir.iterdir():
                if dataset_dir.is_dir() and dataset_dir.name not in ['.', '..']:
                    models = []
                    
                    # Look for model files
                    for model_file in dataset_dir.glob('model_*.pt'):
                        variant = model_file.stem.replace('model_', '')
                        
                        # Check for metadata
                        metadata_file = dataset_dir / f'metadata_{variant}.json'
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                description = metadata.get('architecture', {}).get('description', '')
                                performance = metadata.get('performance', {})
                        else:
                            description = f"{variant} model"
                            performance = {}
                        
                        models.append({
                            'variant': variant,
                            'path': str(model_file),
                            'description': description,
                            'performance': performance
                        })
                    
                    if models:
                        demo_models[dataset_dir.name] = models
        
        return demo_models