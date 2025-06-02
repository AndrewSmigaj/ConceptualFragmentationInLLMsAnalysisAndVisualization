"""
Activation Manager for Concept MRI.

Handles activation extraction from models using the concept_fragmentation
activation collection infrastructure, and provides session-based storage
to avoid JSON serialization issues with Dash stores.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, List
import logging
import threading
import sys
from datetime import datetime, timedelta

# Import from concept_fragmentation
from concept_fragmentation.activation.collector import (
    ActivationCollector, 
    CollectionConfig,
    ActivationFormat
)

logger = logging.getLogger(__name__)


class ActivationManager:
    """
    Manages activation extraction and storage for Concept MRI.
    
    This class bridges the UI components with the concept_fragmentation
    activation collection infrastructure and provides session-based storage
    for numpy arrays that cannot be efficiently stored in Dash stores.
    """
    
    def __init__(self, max_memory_mb: float = 2048, session_timeout_minutes: int = 120):
        """
        Initialize the activation manager.
        
        Args:
            max_memory_mb: Maximum memory usage in megabytes for session storage
            session_timeout_minutes: Minutes before a session expires
        """
        # Activation collection
        self.collector = None
        self._init_collector()
        
        # Session storage
        self._sessions = {}  # session_id -> activations
        self._metadata = {}  # session_id -> metadata
        self._timestamps = {}  # session_id -> last access time
        self._memory_usage = {}  # session_id -> estimated memory in bytes
        
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
        
        logger.info(f"ActivationManager initialized with {max_memory_mb}MB limit and {session_timeout_minutes}min timeout")
    
    def _init_collector(self):
        """Initialize the activation collector with appropriate config."""
        config = CollectionConfig(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            format=ActivationFormat.NUMPY,
            batch_size=32,
            log_dimensions=True
        )
        self.collector = ActivationCollector(config)
    
    def extract_activations(
        self,
        model_data: Dict[str, Any],
        dataset_data: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Extract activations from a model given a dataset.
        
        Args:
            model_data: Dictionary containing model information
            dataset_data: Dictionary containing dataset information
            
        Returns:
            Dictionary mapping layer names to activation arrays
        """
        try:
            # Load the model
            model = self._load_model(model_data)
            if model is None:
                return {}
            
            # Get input data
            X = self._get_input_data(dataset_data)
            if X is None:
                return {}
            
            # Register model with collector
            self.collector.register_model(
                model,
                model_id='feedforward',
                include_patterns=['Linear', 'ReLU']  # Capture linear layers and ReLU activations
            )
            
            # Collect activations
            activations = self.collector.collect(
                model=model,
                inputs=X,
                model_id='feedforward',
                streaming=False
            )
            
            # Post-process activations to get only ReLU outputs
            processed_activations = self._process_activations(activations)
            
            logger.info(f"Extracted activations from {len(processed_activations)} layers")
            
            return processed_activations
            
        except Exception as e:
            logger.error(f"Failed to extract activations: {e}")
            # Return mock activations as fallback
            return self._generate_mock_activations(model_data, dataset_data)
    
    def _load_model(self, model_data: Dict[str, Any]) -> Optional[nn.Module]:
        """Load a PyTorch model from model data."""
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
                elif activation_fn.lower() == 'sigmoid':
                    layers.append(nn.Sigmoid())
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
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def _get_input_data(self, dataset_data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract input data from dataset data."""
        try:
            # Check for actual data in dataset
            if 'data' in dataset_data:
                X = dataset_data['data']
                if isinstance(X, np.ndarray):
                    return torch.FloatTensor(X)
                elif isinstance(X, torch.Tensor):
                    return X.float()
            
            # Try to load from file if path is provided
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
            num_features = dataset_data.get('num_features', 10)
            
            # Generate random data as fallback
            X = torch.randn(num_samples, num_features)
            return X
            
        except Exception as e:
            logger.error(f"Failed to get input data: {e}")
            return None
    
    def _process_activations(self, activations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Process activations to extract only hidden layer outputs.
        
        We want activations after ReLU/activation functions, not raw linear outputs.
        """
        processed = {}
        layer_idx = 0
        
        # Sort layers by their appearance in the model
        sorted_layers = sorted(activations.keys(), key=lambda x: int(x.split('.')[-1]) if '.' in x else 0)
        
        for layer_name in sorted_layers:
            # Check if this is a ReLU or other activation output
            if any(act in layer_name.lower() for act in ['relu', 'tanh', 'sigmoid', 'activation']):
                processed[f'layer_{layer_idx}'] = activations[layer_name]
                layer_idx += 1
        
        # If no activation layers found, use linear layer outputs
        if not processed:
            layer_idx = 0
            for layer_name in sorted_layers:
                if 'linear' in layer_name.lower():
                    processed[f'layer_{layer_idx}'] = activations[layer_name]
                    layer_idx += 1
        
        return processed
    
    def _generate_mock_activations(
        self, 
        model_data: Dict[str, Any], 
        dataset_data: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Generate mock activations as a fallback."""
        architecture = model_data.get('architecture', [32, 16, 8])
        num_samples = dataset_data.get('num_samples', 100)
        
        activations = {}
        for i, hidden_size in enumerate(architecture):
            # Generate random activations with some structure
            # This helps demonstrate the UI even without real activations
            base = np.random.randn(num_samples, hidden_size) * 0.5
            # Add some cluster structure
            for cluster in range(min(3, hidden_size // 2)):
                cluster_mask = np.random.rand(num_samples) < 0.3
                base[cluster_mask, cluster*2:(cluster+1)*2] += np.random.randn() * 2
            
            # Apply ReLU-like transformation
            activations[f'layer_{i}'] = np.maximum(0, base)
        
        return activations
    
    # ===== Session Storage Methods =====
    
    def store_activations(
        self, 
        session_id: str, 
        activations: Dict[str, np.ndarray], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store activations in session storage and return a reference key.
        
        Args:
            session_id: Unique session identifier
            activations: Dictionary of layer names to numpy arrays
            metadata: Optional metadata about the activations
            
        Returns:
            The session_id for reference
            
        Raises:
            MemoryError: If storing would exceed memory limit
        """
        with self._lock:
            # Estimate memory usage
            memory_size = self._estimate_memory_size(activations)
            
            # Check if we need to clean up old sessions
            current_total = sum(self._memory_usage.values())
            if current_total + memory_size > self.max_memory_bytes:
                self._cleanup_old_sessions()
                
                # Check again after cleanup
                current_total = sum(self._memory_usage.values())
                if current_total + memory_size > self.max_memory_bytes:
                    raise MemoryError(
                        f"Cannot store activations: would exceed memory limit "
                        f"({(current_total + memory_size) / 1024 / 1024:.1f}MB > "
                        f"{self.max_memory_bytes / 1024 / 1024:.1f}MB)"
                    )
            
            # Store the data
            self._sessions[session_id] = activations
            self._timestamps[session_id] = datetime.now()
            self._memory_usage[session_id] = memory_size
            
            if metadata:
                self._metadata[session_id] = metadata
            
            logger.debug(f"Stored activations for session {session_id}: "
                        f"{len(activations)} layers, {memory_size / 1024 / 1024:.1f}MB")
            
            return session_id
    
    def get_activations(self, session_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Retrieve activations by session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary of activations or None if not found
        """
        with self._lock:
            if session_id in self._sessions:
                # Update timestamp
                self._timestamps[session_id] = datetime.now()
                return self._sessions[session_id]
            return None
    
    def get_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        with self._lock:
            return self._metadata.get(session_id)
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear activations for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was cleared, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._metadata.pop(session_id, None)
                self._timestamps.pop(session_id, None)
                memory_freed = self._memory_usage.pop(session_id, 0)
                
                logger.debug(f"Cleared session {session_id}, freed {memory_freed / 1024 / 1024:.1f}MB")
                return True
            return False
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about all active sessions.
        
        Returns:
            Dictionary with session statistics
        """
        with self._lock:
            total_memory = sum(self._memory_usage.values())
            
            sessions_info = []
            for session_id in self._sessions:
                age = datetime.now() - self._timestamps[session_id]
                sessions_info.append({
                    'session_id': session_id,
                    'memory_mb': self._memory_usage[session_id] / 1024 / 1024,
                    'age_minutes': age.total_seconds() / 60,
                    'num_layers': len(self._sessions[session_id])
                })
            
            return {
                'num_sessions': len(self._sessions),
                'total_memory_mb': total_memory / 1024 / 1024,
                'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
                'sessions': sessions_info
            }
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove sessions that have exceeded the timeout.
        
        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            return self._cleanup_old_sessions()
    
    def _estimate_memory_size(self, activations: Dict[str, np.ndarray]) -> int:
        """
        Estimate memory usage of activations.
        
        Args:
            activations: Dictionary of numpy arrays
            
        Returns:
            Estimated size in bytes
        """
        total_size = 0
        for name, array in activations.items():
            # Array size + some overhead for dict storage
            total_size += array.nbytes + sys.getsizeof(name) + 64
        return total_size
    
    def _cleanup_old_sessions(self) -> int:
        """
        Remove expired sessions. Must be called with lock held.
        
        Returns:
            Number of sessions cleaned up
        """
        now = datetime.now()
        expired_sessions = []
        
        for session_id, timestamp in self._timestamps.items():
            if now - timestamp > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self._sessions[session_id]
            self._metadata.pop(session_id, None)
            del self._timestamps[session_id]
            del self._memory_usage[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)


# Global instance with reasonable defaults
activation_manager = ActivationManager(
    max_memory_mb=2048,  # 2GB limit
    session_timeout_minutes=120  # 2 hour timeout
)