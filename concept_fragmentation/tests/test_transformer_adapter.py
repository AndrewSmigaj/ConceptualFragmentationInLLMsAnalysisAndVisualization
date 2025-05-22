"""
Tests for the transformer model adapter implementation.

This module provides tests for the transformer model adapter
classes to ensure they work correctly with transformer models.
"""

import unittest
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import pytest

from concept_fragmentation.models.transformer_adapter import (
    TransformerModelArchitecture,
    TransformerModelAdapter,
    GPT2Adapter,
    get_transformer_adapter
)

# Skip tests if transformers is not installed
transformers_available = True
try:
    from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
except ImportError:
    transformers_available = False


# Simple transformer model for testing when transformers not available
class SimpleTransformerModel(nn.Module):
    """
    A simple transformer model for testing the adapter.
    """
    
    def __init__(self, vocab_size=1000, hidden_size=64, num_layers=2, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        embeddings = self.embedding(input_ids)
        
        # Create an attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Convert boolean mask to attention mask
        if attention_mask.dtype == torch.bool:
            # Create a float mask that's 0 for masked positions and 1 for unmasked
            float_mask = attention_mask.float().masked_fill(~attention_mask, float('-inf'))
            # Ensure proper broadcasting
            float_mask = float_mask.unsqueeze(1).unsqueeze(2)
        else:
            float_mask = attention_mask
        
        hidden_states = []
        x = embeddings
        
        # Collect hidden states
        if output_hidden_states:
            hidden_states.append(x)
        
        # Process each layer
        for layer in self.transformer.layers:
            x = layer(x, src_mask=float_mask)
            if output_hidden_states:
                hidden_states.append(x)
        
        logits = self.output_projection(x)
        
        if output_hidden_states:
            return SimpleTransformerOutput(logits=logits, hidden_states=hidden_states)
        
        return logits


class SimpleTransformerOutput:
    """Simple container for transformer outputs."""
    
    def __init__(self, logits, hidden_states=None, attentions=None):
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions = attentions


class TestTransformerModelArchitecture(unittest.TestCase):
    """Test cases for TransformerModelArchitecture."""
    
    def test_generic_transformer_architecture(self):
        """Test generic transformer architecture identification."""
        # Create a simple transformer model
        model = SimpleTransformerModel()
        
        # Create the architecture
        architecture = TransformerModelArchitecture(model, model_type="simple_transformer")
        
        # Check architecture name
        self.assertEqual(architecture.name, "transformer")
        
        # Check that layers were identified
        self.assertGreater(len(architecture._layers), 0)
        
        # Check that there are activation points
        self.assertGreater(len(architecture.get_activation_points()), 0)
        
        # Check that we detected basic layer types
        self.assertIn('embedding', architecture.layer_types)
        self.assertIn('transformer_encoder_layer', architecture.layer_types)
        self.assertIn('linear', architecture.layer_types)


@pytest.mark.skipif(not transformers_available, reason="transformers not installed")
class TestGPT2Adapter(unittest.TestCase):
    """Test cases for GPT2Adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small GPT-2 model for testing
        config = GPT2Config(
            vocab_size=1000,
            n_positions=128,
            n_embd=64,
            n_layer=2,
            n_head=4
        )
        
        self.model = GPT2Model(config)
        
        # Create tokenizer
        self.tokenizer = None
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        except Exception:
            # Skip tokenizer if can't load
            pass
    
    def test_gpt2_adapter_creation(self):
        """Test creating a GPT-2 adapter."""
        # Create adapter
        adapter = GPT2Adapter(self.model, tokenizer=self.tokenizer)
        
        # Check adapter properties
        self.assertEqual(adapter.architecture.name, "transformer")
        self.assertEqual(adapter.architecture.model_type, "gpt2")
        
        # Check that layers were identified
        layer_info = adapter.architecture._layers
        self.assertIn('token_embedding', layer_info)
        self.assertIn('transformer_layer_0', layer_info)
        self.assertIn('transformer_layer_1', layer_info)
        
        # Check activation points
        act_points = adapter.architecture.get_activation_points()
        self.assertIn('embedding', act_points)
        self.assertIn('transformer_layer_0_output', act_points)
        self.assertIn('transformer_layer_1_output', act_points)
        self.assertIn('transformer_layer_0_attention_attn_probs', act_points)
    
    def test_gpt2_forward_pass(self):
        """Test forward pass through GPT-2 adapter."""
        # Create adapter
        adapter = GPT2Adapter(self.model, tokenizer=self.tokenizer)
        
        # Create sample input
        input_ids = torch.randint(0, 1000, (2, 16))  # Batch of 2, sequence length 16
        attention_mask = torch.ones_like(input_ids)
        
        # Run forward pass
        outputs = adapter.forward({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })
        
        # Check outputs
        self.assertIsNotNone(outputs)
        self.assertEqual(outputs.last_hidden_state.shape, (2, 16, 64))
    
    def test_gpt2_layer_outputs(self):
        """Test getting layer outputs from GPT-2 adapter."""
        # Create adapter
        adapter = GPT2Adapter(self.model, tokenizer=self.tokenizer)
        
        # Create sample input
        input_ids = torch.randint(0, 1000, (2, 16))  # Batch of 2, sequence length 16
        attention_mask = torch.ones_like(input_ids)
        
        # Get layer outputs
        layer_outputs = adapter.get_layer_outputs({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })
        
        # Check that we have outputs from different layers
        self.assertGreater(len(layer_outputs), 0)
        
        # Check the shape of activations
        for name, tensor in layer_outputs.items():
            self.assertEqual(tensor.shape[0], 2)  # Batch size
            self.assertEqual(tensor.shape[1], 16)  # Sequence length
    
    def test_gpt2_attention_patterns(self):
        """Test getting attention patterns from GPT-2 adapter."""
        # Create adapter
        adapter = GPT2Adapter(self.model, tokenizer=self.tokenizer)
        
        # Create sample input
        input_ids = torch.randint(0, 1000, (2, 16))  # Batch of 2, sequence length 16
        attention_mask = torch.ones_like(input_ids)
        
        # Get attention patterns
        attention_patterns = adapter.get_attention_patterns({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })
        
        # Check that we have attention patterns
        self.assertGreater(len(attention_patterns), 0)
        
        # Check the shape of attention patterns (batch, heads, seq_len, seq_len)
        for name, tensor in attention_patterns.items():
            self.assertEqual(tensor.shape[0], 2)  # Batch size
            self.assertEqual(tensor.shape[2], 16)  # Sequence length (source)
            self.assertEqual(tensor.shape[3], 16)  # Sequence length (target)
    
    def test_text_input_processing(self):
        """Test text input processing with tokenizer."""
        # Skip if no tokenizer
        if self.tokenizer is None:
            self.skipTest("Tokenizer not available")
        
        # Create adapter
        adapter = GPT2Adapter(self.model, tokenizer=self.tokenizer)
        
        # Sample text input
        text = "Hello, world!"
        
        # Process through adapter
        outputs = adapter.forward(text)
        
        # Check that we got outputs
        self.assertIsNotNone(outputs)
        
        # Get layer outputs from text
        layer_outputs = adapter.get_layer_outputs(text)
        
        # Check that we have layer outputs
        self.assertGreater(len(layer_outputs), 0)
    
    def test_factory_function(self):
        """Test the adapter factory function."""
        # Create adapter using factory
        adapter = get_transformer_adapter(self.model, model_type="gpt2", tokenizer=self.tokenizer)
        
        # Check that we got a GPT2Adapter
        self.assertIsInstance(adapter, GPT2Adapter)
        
        # Check its functionality
        input_ids = torch.randint(0, 1000, (1, 8))
        outputs = adapter.forward({'input_ids': input_ids})
        
        # Check outputs
        self.assertIsNotNone(outputs)


# Only run these if transformers is available
if transformers_available:
    def test_gpt2_model_loading():
        """Test loading a pre-trained GPT-2 model."""
        try:
            # Try loading the smallest GPT-2 model
            model = GPT2Model.from_pretrained("openai-community/gpt2")
            tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
            
            # Create adapter
            adapter = GPT2Adapter(model, tokenizer=tokenizer)
            
            # Basic checks
            assert adapter.architecture.name == "transformer"
            
            # Check layers
            layer_info = adapter.architecture._layers
            assert 'token_embedding' in layer_info
            
            # Check for transformer blocks
            transformer_layers = [name for name in layer_info if name.startswith('transformer_layer_')]
            assert len(transformer_layers) == 12  # GPT-2 small has 12 layers
            
        except Exception as e:
            pytest.skip(f"Could not load pretrained GPT-2 model: {e}")


if __name__ == '__main__':
    unittest.main()