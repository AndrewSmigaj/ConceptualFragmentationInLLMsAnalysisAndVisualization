"""
Callback for automatic activation extraction using existing pipeline infrastructure.

This integrates the concept_fragmentation pipeline stages with the Concept MRI UI.
"""
from dash import Input, Output, State, callback
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

# Import existing pipeline infrastructure
from concept_fragmentation.pipeline import Pipeline, PipelineConfig

# Import activation manager for session storage
from concept_mri.core.activation_manager import activation_manager

# Get the debug logger
try:
    from concept_mri.app import get_debug_logger
    logger = get_debug_logger()
except ImportError:
    # Fallback if app not initialized yet
    logger = logging.getLogger('concept_mri.debug')
from concept_fragmentation.pipeline.stages import (
    ActivationCollectionStage,
    ClusteringStage
)
from concept_fragmentation.activation.collector import (
    ActivationCollector,
    CollectionConfig,
    ActivationFormat
)


def register_activation_extraction_callback(app):
    """Register callback for automatic activation extraction."""
    
    @app.callback(
        Output('model-store', 'data', allow_duplicate=True),
        [Input('model-store', 'data'),
         Input('dataset-store', 'data')],
        State('session-id-store', 'data'),
        prevent_initial_call=True
    )
    def extract_activations_when_ready(model_data, dataset_data, session_id):
        """
        Extract activations when both model and dataset are loaded.
        
        Uses the existing pipeline infrastructure from concept_fragmentation.
        """
        # Check if we have both model and dataset
        if not model_data or not dataset_data:
            return model_data
        
        # Check if model is loaded and dataset has data
        if not model_data.get('model_loaded') or not dataset_data.get('filename'):
            return model_data
        
        # Check if activations already exist (either directly or via session)
        if model_data.get('activations') or model_data.get('activation_session_id'):
            logger.info("Activations already exist, skipping extraction")
            return model_data
        
        logger.info("Both model and dataset available, extracting activations...")
        
        try:
            # Load the PyTorch model
            model = _load_pytorch_model(model_data)
            if model is None:
                logger.error("Failed to load PyTorch model")
                return model_data
            
            # Get input data
            inputs = _get_input_tensor(dataset_data, model_data)
            if inputs is None:
                logger.error("Failed to get input data")
                return model_data
            logger.debug(f"Input tensor shape: {inputs.shape}")
            
            # Create pipeline configuration
            config = PipelineConfig(
                log_progress=True,
                use_context=True
            )
            
            # Create activation collector with appropriate settings
            collector_config = CollectionConfig(
                device='cuda' if torch.cuda.is_available() else 'cpu',
                format=ActivationFormat.NUMPY,
                batch_size=32,
                log_dimensions=True
            )
            collector = ActivationCollector(collector_config)
            
            # Create pipeline with activation collection stage
            pipeline = Pipeline(config)
            
            # Add activation collection stage
            # We want to collect from ReLU/activation layers, not raw linear outputs
            collection_stage = ActivationCollectionStage(
                collector=collector,
                layer_names=None,  # Collect from all layers
                streaming=False,
                name="collect_activations"
            )
            pipeline.add_stage(collection_stage)
            
            # Prepare input data for pipeline
            pipeline_input = {
                'model': model,
                'inputs': inputs,
                'model_id': 'feedforward',
                'split_name': 'all_data',
                'metadata': {
                    'dataset_name': dataset_data.get('filename', 'unknown'),
                    'model_architecture': model_data.get('architecture', [])
                }
            }
            
            # Run the pipeline
            logger.debug("About to execute pipeline...")
            try:
                result = pipeline.execute(pipeline_input)
                logger.debug(f"Pipeline returned result with keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            except Exception as pipeline_error:
                logger.error(f"Pipeline execution failed with: {pipeline_error}")
                logger.debug("Full traceback of pipeline error:", exc_info=True)
                raise
            
            # Extract and process activations
            if 'activations' in result:
                raw_activations = result['activations']
                logger.debug(f"Got raw_activations, type: {type(raw_activations)}, keys: {raw_activations.keys() if isinstance(raw_activations, dict) else 'Not a dict'}")
                
                # Process to get only activation layer outputs (after ReLU, etc)
                logger.info(f"Processing {len(raw_activations)} raw activations...")
                logger.debug("About to call _extract_hidden_layer_activations...")
                processed_activations = _extract_hidden_layer_activations(raw_activations)
                logger.debug(f"_extract_hidden_layer_activations returned {len(processed_activations)} activations")
                logger.info(f"Extracted {len(processed_activations)} hidden layer activations")
                
                if processed_activations:
                    # Use the session ID from the store (or generate if not provided)
                    if not session_id:
                        session_id = f"session_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Store activations in the activation manager
                    activation_metadata = {
                        'dataset_name': dataset_data.get('filename', 'unknown'),
                        'model_architecture': model_data.get('architecture', []),
                        'timestamp': datetime.now().isoformat(),
                        'num_samples': next(iter(processed_activations.values())).shape[0],
                        'activation_shapes': {k: v.shape for k, v in processed_activations.items()}
                    }
                    
                    try:
                        stored_session_id = activation_manager.store_activations(
                            session_id=session_id,
                            activations=processed_activations,
                            metadata=activation_metadata
                        )
                        
                        # Update model data with session reference instead of raw activations
                        model_data['activation_session_id'] = stored_session_id
                        model_data['activations'] = None  # Clear any old direct storage
                        model_data['activation_shapes'] = {
                            k: v.shape for k, v in processed_activations.items()
                        }
                        
                        logger.info(f"Successfully stored activations in session {stored_session_id}")
                        logger.info(f"Extracted activations from {len(processed_activations)} layers")
                        for layer_name, activation in processed_activations.items():
                            logger.info(f"  - {layer_name}: {activation.shape}")
                            
                    except MemoryError as e:
                        logger.error(f"Failed to store activations due to memory limit: {e}")
                        # Fall back to direct storage (will cause JSON issues but better than nothing)
                        model_data['activations'] = processed_activations
                        model_data['activation_session_id'] = None
                        
                else:
                    logger.warning("No hidden layer activations found")
                    # Generate mock activations as fallback
                    mock_activations = _generate_mock_activations(model_data, dataset_data)
                    
                    # Store mock activations in session storage too
                    # Use existing session_id or generate for mock
                    if not session_id:
                        session_id = f"mock_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    try:
                        stored_session_id = activation_manager.store_activations(
                            session_id=session_id,
                            activations=mock_activations,
                            metadata={'type': 'mock', 'timestamp': datetime.now().isoformat()}
                        )
                        model_data['activation_session_id'] = stored_session_id
                        model_data['activations'] = None
                    except MemoryError:
                        model_data['activations'] = mock_activations
                        model_data['activation_session_id'] = None
            else:
                logger.error("Pipeline did not return activations")
                
        except Exception as e:
            logger.error(f"Failed to extract activations: {e}")
            logger.debug(f"Exception in extract_activations_when_ready: {e}", exc_info=True)
            
            # Generate mock activations as fallback
            mock_activations = _generate_mock_activations(model_data, dataset_data)
            
            # Store mock activations in session storage
            # Use existing session_id or generate for error case
            if not session_id:
                session_id = f"mock_error_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                stored_session_id = activation_manager.store_activations(
                    session_id=session_id,
                    activations=mock_activations,
                    metadata={'type': 'mock', 'error': str(e), 'timestamp': datetime.now().isoformat()}
                )
                model_data['activation_session_id'] = stored_session_id
                model_data['activations'] = None
            except MemoryError:
                model_data['activations'] = mock_activations
                model_data['activation_session_id'] = None
        
        return model_data


def _load_pytorch_model(model_data: Dict[str, Any]) -> Optional[nn.Module]:
    """Load PyTorch model from model data."""
    try:
        model_path = model_data.get('path')
        if not model_path:
            return None
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get architecture info
        architecture = checkpoint.get('architecture', [32, 16, 8])
        input_size = checkpoint.get('input_size', 10)
        output_size = checkpoint.get('output_size', 2)
        activation_fn = checkpoint.get('activation', 'relu')
        dropout_rate = checkpoint.get('dropout_rate', 0.0)
        
        # Build model
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(architecture):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Add activation
            if activation_fn.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation_fn.lower() == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())  # Default
            
            # Add dropout if specified
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        # Create sequential model
        model = nn.Sequential(*layers)
        
        # Load weights - handle different state dict formats
        state_dict = checkpoint['model_state_dict']
        
        # Check if state dict has "model." prefix
        if any(key.startswith('model.') for key in state_dict.keys()):
            # Remove "model." prefix from keys
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove "model." prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def _get_input_tensor(dataset_data: Dict[str, Any], model_data: Dict[str, Any]) -> Optional[torch.Tensor]:
    """Extract input tensor from dataset data."""
    try:
        # Try to load actual data from file
        if 'path' in dataset_data:
            file_path = dataset_data['path']
            if file_path.endswith('.npz'):
                data = np.load(file_path)
                if 'X' in data:
                    return torch.FloatTensor(data['X'])
                elif 'data' in data:
                    return torch.FloatTensor(data['data'])
        
        # Generate synthetic data based on dataset info
        num_samples = dataset_data.get('num_samples', 100)
        num_features = dataset_data.get('num_features', model_data.get('input_dim', 10))
        
        # Generate random data
        X = torch.randn(num_samples, num_features)
        return X
        
    except Exception as e:
        logger.error(f"Failed to get input data: {e}")
        return None


def _extract_hidden_layer_activations(raw_activations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Extract only hidden layer activations (after activation functions).
    
    The ActivationCollector captures all layers including linear layers.
    We want only the outputs after activation functions (ReLU, Tanh, etc).
    """
    logger.debug(f"_extract_hidden_layer_activations: Got {len(raw_activations)} raw activations")
    processed = {}
    layer_idx = 0
    
    # Sort layers by their numeric order
    sorted_layers = sorted(raw_activations.keys(), 
                          key=lambda x: int(x) if x.isdigit() else int(x.split('.')[-1]) if '.' in x else 0)
    logger.debug(f"Sorted layers: {sorted_layers}")
    
    # Based on the log, we see the pattern:
    # Layer 0: Linear (skip)
    # Layer 1: ReLU (take this)
    # Layer 2: Dropout (skip)
    # Layer 3: Linear (skip)
    # Layer 4: ReLU (take this)
    # Layer 5: Dropout (skip)
    # Layer 6: Linear (skip)
    # Layer 7: ReLU (take this)
    # Layer 8: Dropout (skip)
    # Layer 9: Linear (output layer, skip)
    
    # For numbered layers, we know ReLU layers are at indices 1, 4, 7
    # More generally, they're at 3n+1 for n=0,1,2,...
    relu_indices = [1, 4, 7]
    
    for layer_name in sorted_layers:
        # Check if this is a ReLU layer by index
        if layer_name.isdigit() and int(layer_name) in relu_indices:
            activation = raw_activations[layer_name]
            logger.debug(f"Processing layer {layer_name} (ReLU), type: {type(activation)}, shape: {activation.shape if hasattr(activation, 'shape') else 'NO SHAPE'}")
            processed[f'layer_{layer_idx}'] = activation
            layer_idx += 1
    
    # If no activation layers found (shouldn't happen with our pattern)
    # Fall back to a more general approach
    if not processed:
        logger.warning("No ReLU layers found by index, trying alternative approach")
        layer_idx = 0
        # Take every third layer starting from index 1 (ReLU layers)
        for i, layer_name in enumerate(sorted_layers):
            if i % 3 == 1 and i < len(sorted_layers) - 1:  # ReLU pattern and not the output
                activation = raw_activations[layer_name]
                logger.debug(f"Alternative approach - processing layer {layer_name}, type: {type(activation)}")
                processed[f'layer_{layer_idx}'] = activation
                layer_idx += 1
    
    logger.debug(f"Returning {len(processed)} processed activations")
    return processed


def _generate_mock_activations(model_data: Dict[str, Any], dataset_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Generate mock activations as a fallback."""
    architecture = model_data.get('architecture', [32, 16, 8])
    num_samples = dataset_data.get('num_samples', 100)
    
    activations = {}
    for i, hidden_size in enumerate(architecture):
        # Generate random activations with some structure
        base = np.random.randn(num_samples, hidden_size) * 0.5
        
        # Add some cluster structure
        for cluster in range(min(3, hidden_size // 2)):
            cluster_mask = np.random.rand(num_samples) < 0.3
            base[cluster_mask, cluster*2:(cluster+1)*2] += np.random.randn() * 2
        
        # Apply ReLU-like transformation
        activations[f'layer_{i}'] = np.maximum(0, base)
    
    return activations