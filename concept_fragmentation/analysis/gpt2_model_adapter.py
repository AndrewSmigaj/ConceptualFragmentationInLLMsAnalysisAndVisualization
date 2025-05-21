"""
GPT-2 model adapter for activation extraction and analysis.

This module provides specialized functionality for extracting and processing
activations from GPT-2 models specifically for Archetypal Path Analysis.
It builds on the existing GPT2Adapter with functionality tailored to APA.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import numpy as np
import logging
from pathlib import Path
import json
import os
from dataclasses import dataclass, field
from enum import Enum

# Import transformers classes conditionally to handle environments without them
try:
    from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
# Import from existing modules
from ..models.transformer_adapter import GPT2Adapter, TransformerModelArchitecture
from ..hooks.activation_hooks import ActivationHook, get_activation_hooks
from ..config import CACHE_DIR


# Setup logger
logger = logging.getLogger(__name__)


class GPT2ModelType(Enum):
    """Enumeration of supported GPT-2 model types."""
    SMALL = "gpt2"           # 124M parameters
    MEDIUM = "gpt2-medium"   # 355M parameters
    LARGE = "gpt2-large"     # 774M parameters
    XL = "gpt2-xl"           # 1.5B parameters
    
    @classmethod
    def from_string(cls, model_name: str) -> 'GPT2ModelType':
        """Convert string to GPT2ModelType enum."""
        try:
            return cls[model_name.upper().replace('-', '_')]
        except KeyError:
            # Try direct value match
            for member in cls:
                if member.value == model_name:
                    return member
            
            # Fallback to best match
            if "small" in model_name.lower():
                return cls.SMALL
            elif "medium" in model_name.lower():
                return cls.MEDIUM
            elif "large" in model_name.lower():
                return cls.LARGE
            elif "xl" in model_name.lower():
                return cls.XL
            else:
                return cls.SMALL  # Default to small if unknown
                

@dataclass
class GPT2ActivationConfig:
    """Configuration for GPT-2 activation extraction."""
    
    model_type: GPT2ModelType = GPT2ModelType.SMALL
    output_hidden_states: bool = True
    output_attentions: bool = True
    use_cache: bool = True
    cache_dir: Optional[str] = None
    device: str = "cpu"
    include_lm_head: bool = True
    context_window: int = 1024
    layer_groups: Dict[str, List[str]] = field(default_factory=dict)
    # Activation selection options
    include_layer_norm: bool = False
    include_residual: bool = False
    concat_attention_heads: bool = True
    capture_embeddings: bool = True


class GPT2ActivationExtractor:
    """
    Specialized extractor for GPT-2 activations designed for APA analysis.
    
    This class extends the functionality of GPT2Adapter with specific methods
    for extracting and processing activations in formats suitable for 
    Archetypal Path Analysis.
    
    Attributes:
        model: The GPT-2 model
        tokenizer: GPT-2 tokenizer for processing text inputs
        adapter: GPT2Adapter instance for model interaction
        activation_hooks: Dictionary of ActivationHook objects
        config: Configuration for activation extraction
    """
    
    def __init__(
        self,
        model_type: Union[str, GPT2ModelType] = GPT2ModelType.SMALL,
        config: Optional[GPT2ActivationConfig] = None,
        tokenizer: Optional[Any] = None,
        model: Optional[nn.Module] = None
    ):
        """
        Initialize the GPT-2 activation extractor.
        
        Args:
            model_type: Type of GPT-2 model (small, medium, large, xl)
            config: Extraction configuration options
            tokenizer: Optional pre-initialized tokenizer
            model: Optional pre-initialized model (if provided, model_type is ignored)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The transformers library is required for GPT2ActivationExtractor. "
                "Install it with: pip install transformers"
            )
        
        # Set up configuration
        if config is None:
            model_type_enum = (
                model_type if isinstance(model_type, GPT2ModelType)
                else GPT2ModelType.from_string(model_type)
            )
            self.config = GPT2ActivationConfig(model_type=model_type_enum)
        else:
            self.config = config
            
        # Set up device
        device = self.config.device
        if device != 'cpu' and not (device.startswith('cuda') and torch.cuda.is_available()):
            logger.warning(f"Device {device} not available, falling back to CPU")
            device = 'cpu'
            self.config.device = device
        
        # Set up model and tokenizer
        if model is None:
            logger.info(f"Initializing GPT-2 model type: {self.config.model_type.value}")
            model = GPT2LMHeadModel.from_pretrained(
                self.config.model_type.value,
                output_hidden_states=self.config.output_hidden_states,
                output_attentions=self.config.output_attentions,
                cache_dir=self.config.cache_dir
            )
            model.to(device)
            model.eval()  # Set to evaluation mode
            
        self.model = model
        
        # Set up tokenizer
        if tokenizer is None:
            logger.info(f"Initializing GPT-2 tokenizer for: {self.config.model_type.value}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                self.config.model_type.value,
                cache_dir=self.config.cache_dir
            )
        else:
            self.tokenizer = tokenizer
        
        # Initialize adapter with tokenizer
        self.adapter = GPT2Adapter(
            model=model,
            tokenizer=self.tokenizer,
            output_hidden_states=self.config.output_hidden_states,
            output_attentions=self.config.output_attentions
        )
        
        # Set up layer groups for analysis
        if not self.config.layer_groups:
            # Default layer grouping
            self._initialize_default_layer_groups()
        
        # Initialize activation hooks
        self.activation_hooks = {}
        self._setup_hooks()
    
    def _initialize_default_layer_groups(self):
        """Initialize default layer groups for analysis."""
        # Count the number of transformer layers
        num_layers = len(self.model.transformer.h)
        
        # Create groups for embeddings, transformer layers, and the final layer
        self.config.layer_groups = {
            "embeddings": ["token_embedding", "position_embedding", "embedding"],
            "transformer_layers": [f"transformer_layer_{i}" for i in range(num_layers)],
            "attention_layers": [f"transformer_layer_{i}_attention" for i in range(num_layers)],
            "mlp_layers": [f"transformer_layer_{i}_mlp" for i in range(num_layers)],
            "final_layers": ["final_layer_norm", "lm_head"] if self.config.include_lm_head else ["final_layer_norm"]
        }
    
    def _setup_hooks(self):
        """Set up hooks for activation extraction."""
        # Clear any existing hooks
        self._remove_hooks()
        
        # Get architecture information
        architecture = self.adapter.architecture
        
        # Register hooks for each layer group
        for group_name, layer_names in self.config.layer_groups.items():
            logger.debug(f"Setting up hooks for group: {group_name}")
            
            # Filter only existing layers
            existing_layers = [name for name in layer_names if name in architecture.get_layers()]
            
            if not existing_layers:
                logger.warning(f"No layers found for group {group_name}")
                continue
            
            # Create activation hook for this group
            self.activation_hooks[group_name] = get_activation_hooks(
                model=self.model,
                layer_names=existing_layers,
                device=self.config.device
            )
    
    def _remove_hooks(self):
        """Remove all activation hooks."""
        for hook in self.activation_hooks.values():
            hook.remove_hooks()
        self.activation_hooks = {}
    
    def prepare_inputs(self, text_input: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Prepare model inputs from text.
        
        Args:
            text_input: Text or list of texts to process
            
        Returns:
            Dictionary of model inputs ready for the forward pass
        """
        # Handle single string or list of strings
        if isinstance(text_input, str):
            text_input = [text_input]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.context_window
        )
        
        # Move to the correct device
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        return inputs
    
    def get_layer_activations(
        self,
        text_input: Union[str, List[str]],
        layer_groups: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract activations for specified layer groups.
        
        Args:
            text_input: Text or list of texts to process
            layer_groups: List of layer group names to extract (None for all)
            
        Returns:
            Dictionary mapping group names to layer activations
        """
        # Prepare inputs
        inputs = self.prepare_inputs(text_input)
        
        # Determine which groups to extract
        if layer_groups is None:
            layer_groups = list(self.config.layer_groups.keys())
        
        # Run forward pass with no gradient tracking
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract activations from hooks
        activations = {}
        for group_name in layer_groups:
            if group_name not in self.activation_hooks:
                logger.warning(f"Layer group {group_name} not found in hooks")
                continue
            
            # Get numpy activations for this group
            group_activations = self.activation_hooks[group_name].numpy_activations()
            activations[group_name] = group_activations
        
        return activations
    
    def get_token_representations(
        self,
        text_input: Union[str, List[str]],
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get token-level representations from specified layers.
        
        Args:
            text_input: Text or list of texts to process
            layer_names: List of specific layer names (None for all transformer layers)
            
        Returns:
            Dictionary mapping layer names to token representations
            Shape: [batch_size, seq_len, hidden_size]
        """
        # Prepare inputs
        inputs = self.prepare_inputs(text_input)
        
        # Use adapter to get token representations
        token_tensors = self.adapter.get_token_representations(inputs, layer_names)
        
        # Convert to numpy
        token_arrays = {name: tensor.cpu().numpy() for name, tensor in token_tensors.items()}
        
        return token_arrays
    
    def get_attention_patterns(
        self,
        text_input: Union[str, List[str]],
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get attention patterns from specified layers.
        
        Args:
            text_input: Text or list of texts to process
            layer_names: List of specific layer names (None for all attention layers)
            
        Returns:
            Dictionary mapping layer names to attention patterns
            Shape: [batch_size, n_heads, seq_len, seq_len]
        """
        # Prepare inputs
        inputs = self.prepare_inputs(text_input)
        
        # Use adapter to get attention patterns
        attention_tensors = self.adapter.get_attention_patterns(inputs, layer_names)
        
        # Convert to numpy
        attention_arrays = {name: tensor.cpu().numpy() for name, tensor in attention_tensors.items()}
        
        return attention_arrays
    
    def get_apa_activations(
        self,
        text_input: Union[str, List[str]],
        layers: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get activations formatted specifically for APA analysis.
        
        Args:
            text_input: Text or list of texts to process
            layers: List of layer indices to include (None for all layers)
            
        Returns:
            Dictionary mapping layer names to processed activations
            ready for dimensionality reduction and clustering
        """
        # Prepare inputs
        inputs = self.prepare_inputs(text_input)
        
        # Extract token ids and attention mask for later token tracking
        token_ids = inputs["input_ids"].cpu().numpy()
        attention_mask = inputs["attention_mask"].cpu().numpy()
        
        # Determine which transformer layers to extract
        num_layers = len(self.model.transformer.h)
        if layers is None:
            layers = list(range(num_layers))
        
        # Format layer names for extraction
        layer_names = [f"transformer_layer_{i}_output" for i in layers]
        
        # Add embedding layer if requested
        if self.config.capture_embeddings:
            layer_names.insert(0, "embedding")
        
        # Get token representations
        token_representations = self.adapter.get_token_representations(inputs, layer_names)
        
        # Process representations for APA
        apa_activations = {}
        
        for name, tensor in token_representations.items():
            # Extract activations as numpy array
            activations = tensor.cpu().numpy()
            
            # Apply attention mask to zero out padding tokens
            batch_size, seq_len = token_ids.shape
            for batch_idx in range(batch_size):
                mask = attention_mask[batch_idx]
                for seq_idx in range(seq_len):
                    if mask[seq_idx] == 0:  # Padding token
                        activations[batch_idx, seq_idx, :] = 0
            
            # Create name for APA format
            if name == "embedding":
                apa_name = "layer_0"  # Embedding is the first layer
            else:
                # Extract layer index from name and add 1 (to account for embedding)
                layer_idx = int(name.split("_")[2]) + 1 if self.config.capture_embeddings else int(name.split("_")[2])
                apa_name = f"layer_{layer_idx}"
            
            apa_activations[apa_name] = activations
        
        return apa_activations
    
    def extract_activations_for_windows(
        self,
        text_input: Union[str, List[str]],
        window_size: int = 3,
        stride: int = 1,
        include_metadata: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract activations for sliding windows of layers for APA analysis.
        
        Args:
            text_input: Text or list of texts to process
            window_size: Number of consecutive layers in each window
            stride: Step size for sliding the window
            include_metadata: Whether to include tokenization metadata
            
        Returns:
            Dictionary mapping window names to activations and metadata
        """
        # Get total number of layers
        num_layers = len(self.model.transformer.h)
        
        # Calculate number of windows
        num_windows = ((num_layers + 1 - window_size) // stride) + 1
        
        # Prepare inputs
        inputs = self.prepare_inputs(text_input)
        
        # Get token metadata if requested
        metadata = {}
        if include_metadata:
            token_ids = inputs["input_ids"].cpu().numpy()
            attention_mask = inputs["attention_mask"].cpu().numpy()
            
            # Decode tokens for readability
            tokens = []
            for batch_idx in range(token_ids.shape[0]):
                batch_tokens = []
                for token_id in token_ids[batch_idx]:
                    batch_tokens.append(self.tokenizer.decode([token_id]))
                tokens.append(batch_tokens)
            
            metadata = {
                "token_ids": token_ids,
                "attention_mask": attention_mask,
                "tokens": tokens
            }
        
        # Extract activations for all layers at once for efficiency
        all_layers = list(range(num_layers + 1))  # +1 for embedding layer
        all_activations = self.get_apa_activations(text_input, all_layers)
        
        # Create windows
        windows = {}
        for window_idx in range(num_windows):
            start_layer = window_idx * stride
            end_layer = start_layer + window_size
            
            # Get layer indices for this window
            window_layers = list(range(start_layer, end_layer))
            
            # Create window name
            window_name = f"window_{start_layer}_{end_layer-1}"
            
            # Extract activations for this window
            window_activations = {
                f"layer_{i}": all_activations[f"layer_{i}"]
                for i in window_layers
                if f"layer_{i}" in all_activations
            }
            
            # Create window data
            windows[window_name] = {
                "activations": window_activations,
                "window_layers": window_layers
            }
            
            # Add metadata if requested
            if include_metadata:
                windows[window_name]["metadata"] = metadata
        
        return windows
    
    def save_activations(
        self,
        activations: Dict[str, Any],
        output_dir: str,
        filename_prefix: str = "gpt2_activations"
    ) -> str:
        """
        Save extracted activations to disk.
        
        Args:
            activations: Dictionary of activations to save
            output_dir: Directory to save files to
            filename_prefix: Prefix for generated filenames
            
        Returns:
            Path to metadata file
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata
        metadata = {
            "model_type": self.config.model_type.value,
            "layer_files": {},
            "config": {
                "context_window": self.config.context_window,
                "include_lm_head": self.config.include_lm_head,
                "include_layer_norm": self.config.include_layer_norm,
                "include_residual": self.config.include_residual,
                "concat_attention_heads": self.config.concat_attention_heads,
                "capture_embeddings": self.config.capture_embeddings
            }
        }
        
        # Save each activation array as a separate file
        for window_name, window_data in activations.items():
            # Create directory for this window
            window_dir = output_path / window_name
            window_dir.mkdir(exist_ok=True)
            
            # Save activations
            activation_data = window_data["activations"]
            for layer_name, layer_activations in activation_data.items():
                # Create filename
                layer_filename = f"{filename_prefix}_{window_name}_{layer_name}.npy"
                file_path = window_dir / layer_filename
                
                # Save numpy array
                np.save(file_path, layer_activations)
                
                # Add to metadata
                if window_name not in metadata["layer_files"]:
                    metadata["layer_files"][window_name] = {}
                metadata["layer_files"][window_name][layer_name] = str(file_path)
            
            # Save window metadata
            window_metadata = {k: v for k, v in window_data.items() if k != "activations"}
            window_metadata_file = window_dir / f"{window_name}_metadata.json"
            
            # Convert numpy arrays in metadata to lists for JSON serialization
            if "metadata" in window_metadata and "token_ids" in window_metadata["metadata"]:
                window_metadata["metadata"]["token_ids"] = window_metadata["metadata"]["token_ids"].tolist()
                window_metadata["metadata"]["attention_mask"] = window_metadata["metadata"]["attention_mask"].tolist()
            
            with open(window_metadata_file, 'w') as f:
                json.dump(window_metadata, f, indent=2)
            
            # Add to main metadata
            metadata["window_metadata_file"] = str(window_metadata_file)
        
        # Save main metadata file
        metadata_file = output_path / f"{filename_prefix}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(metadata_file)


def load_gpt2_small():
    """Helper function to load GPT-2 small model with extractor."""
    return GPT2ActivationExtractor(model_type=GPT2ModelType.SMALL)


def load_gpt2_medium():
    """Helper function to load GPT-2 medium model with extractor."""
    return GPT2ActivationExtractor(model_type=GPT2ModelType.MEDIUM)


def load_gpt2_large():
    """Helper function to load GPT-2 large model with extractor."""
    return GPT2ActivationExtractor(model_type=GPT2ModelType.LARGE)


def load_gpt2_xl():
    """Helper function to load GPT-2 XL model with extractor."""
    return GPT2ActivationExtractor(model_type=GPT2ModelType.XL)