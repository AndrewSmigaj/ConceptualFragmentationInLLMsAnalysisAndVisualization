"""
Tests for the GPT-2 model adapter functionality.
"""

import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add path to import from project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import module to test
from concept_fragmentation.analysis.gpt2_model_adapter import (
    GPT2ModelType,
    GPT2ActivationConfig,
    GPT2ActivationExtractor
)


# Check if transformers is available
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@unittest.skipIf(not TRANSFORMERS_AVAILABLE, "transformers library not available")
class TestGPT2ModelType(unittest.TestCase):
    """Test the GPT2ModelType enum."""
    
    def test_from_string_exact_match(self):
        """Test converting exact model type strings."""
        self.assertEqual(GPT2ModelType.from_string("gpt2"), GPT2ModelType.SMALL)
        self.assertEqual(GPT2ModelType.from_string("gpt2-medium"), GPT2ModelType.MEDIUM)
        self.assertEqual(GPT2ModelType.from_string("gpt2-large"), GPT2ModelType.LARGE)
        self.assertEqual(GPT2ModelType.from_string("gpt2-xl"), GPT2ModelType.XL)
    
    def test_from_string_enum_name(self):
        """Test converting enum names."""
        self.assertEqual(GPT2ModelType.from_string("SMALL"), GPT2ModelType.SMALL)
        self.assertEqual(GPT2ModelType.from_string("MEDIUM"), GPT2ModelType.MEDIUM)
        self.assertEqual(GPT2ModelType.from_string("LARGE"), GPT2ModelType.LARGE)
        self.assertEqual(GPT2ModelType.from_string("XL"), GPT2ModelType.XL)
    
    def test_from_string_fallback(self):
        """Test fallback behavior for non-exact matches."""
        self.assertEqual(GPT2ModelType.from_string("small"), GPT2ModelType.SMALL)
        self.assertEqual(GPT2ModelType.from_string("gpt2_small"), GPT2ModelType.SMALL)
        self.assertEqual(GPT2ModelType.from_string("medium_sized"), GPT2ModelType.MEDIUM)
        self.assertEqual(GPT2ModelType.from_string("large_model"), GPT2ModelType.LARGE)
        self.assertEqual(GPT2ModelType.from_string("extra_large"), GPT2ModelType.XL)
        
        # Default to small for unknown
        self.assertEqual(GPT2ModelType.from_string("unknown"), GPT2ModelType.SMALL)


@unittest.skipIf(not TRANSFORMERS_AVAILABLE, "transformers library not available")
class TestGPT2ActivationConfig(unittest.TestCase):
    """Test the GPT2ActivationConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = GPT2ActivationConfig()
        
        # Check defaults
        self.assertEqual(config.model_type, GPT2ModelType.SMALL)
        self.assertTrue(config.output_hidden_states)
        self.assertTrue(config.output_attentions)
        self.assertTrue(config.use_cache)
        self.assertIsNone(config.cache_dir)
        self.assertEqual(config.device, "cpu")
        self.assertTrue(config.include_lm_head)
        self.assertEqual(config.context_window, 1024)
        self.assertEqual(config.layer_groups, {})


@unittest.skipIf(not TRANSFORMERS_AVAILABLE, "transformers library not available")
class TestGPT2ActivationExtractor(unittest.TestCase):
    """Test the GPT2ActivationExtractor class with mocked models."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock objects
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        
        # Mock the transformer hooks/model to avoid loading actual models
        self.patch_transformers()
        
        # Create config with small context for faster tests
        self.config = GPT2ActivationConfig(
            context_window=128,
            device="cpu"
        )
        
        # Create extractor with mocked components
        self.extractor = GPT2ActivationExtractor(
            model_type="gpt2",
            config=self.config,
            tokenizer=self.mock_tokenizer,
            model=self.mock_model
        )
    
    def patch_transformers(self):
        """Patch transformers components to use mocks."""
        # Mock the model's transformer attribute
        self.mock_transformer = MagicMock()
        self.mock_model.transformer = self.mock_transformer
        
        # Mock transformer blocks
        self.mock_blocks = [MagicMock() for _ in range(12)]  # GPT-2 small has 12 layers
        self.mock_transformer.h = self.mock_blocks
        
        # Mock embeddings
        self.mock_transformer.wte = MagicMock()  # Token embeddings
        self.mock_transformer.wpe = MagicMock()  # Position embeddings
        
        # Mock final layer norm
        self.mock_transformer.ln_f = MagicMock()
        
        # Mock language model head
        self.mock_model.lm_head = MagicMock()
        
        # Mock forward returns
        self.mock_model.return_value = MagicMock(
            hidden_states=(
                MagicMock(),  # embeddings
                *[MagicMock() for _ in range(12)]  # layer outputs
            ),
            attentions=(
                *[MagicMock() for _ in range(12)]  # attention patterns
            )
        )
        
        # Mock tokenizer behavior
        self.mock_tokenizer.return_value = {
            "input_ids": torch.ones((1, 10), dtype=torch.long),  # Batch size 1, seq len 10
            "attention_mask": torch.ones((1, 10), dtype=torch.long)
        }
        self.mock_tokenizer.decode.return_value = "token"
    
    def test_initialization(self):
        """Test that the extractor initializes correctly."""
        # Check model and tokenizer assignment
        self.assertEqual(self.extractor.model, self.mock_model)
        self.assertEqual(self.extractor.tokenizer, self.mock_tokenizer)
        
        # Check adapter initialization
        self.assertIsNotNone(self.extractor.adapter)
        
        # Check layer groups initialization
        self.assertIn("transformer_layers", self.extractor.config.layer_groups)
        self.assertIn("attention_layers", self.extractor.config.layer_groups)
        self.assertIn("mlp_layers", self.extractor.config.layer_groups)
    
    def test_prepare_inputs(self):
        """Test input preparation."""
        # Test with single string
        result = self.extractor.prepare_inputs("test input")
        
        # Check that tokenizer was called
        self.mock_tokenizer.assert_called_once()
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        
        # Reset mock
        self.mock_tokenizer.reset_mock()
        
        # Test with list of strings
        result = self.extractor.prepare_inputs(["test input 1", "test input 2"])
        
        # Check that tokenizer was called with list
        self.mock_tokenizer.assert_called_once()
        self.mock_tokenizer.assert_called_with(
            ["test input 1", "test input 2"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.context_window
        )
    
    def test_get_layer_activations(self):
        """Test layer activation extraction."""
        # Set up hook mocks
        mock_hook = MagicMock()
        mock_hook.numpy_activations.return_value = {"layer_name": np.zeros((1, 10, 768))}
        self.extractor.activation_hooks = {"embeddings": mock_hook}
        
        # Test activation extraction
        result = self.extractor.get_layer_activations("test input", ["embeddings"])
        
        # Check results
        self.assertIn("embeddings", result)
        self.assertIn("layer_name", result["embeddings"])
        
        # Check model was called
        self.mock_model.assert_called_once()
    
    def test_get_token_representations(self):
        """Test token representation extraction."""
        # Set up adapter mock
        self.extractor.adapter.get_token_representations = MagicMock()
        self.extractor.adapter.get_token_representations.return_value = {
            "transformer_layer_0_output": torch.zeros((1, 10, 768))
        }
        
        # Test token representation extraction
        result = self.extractor.get_token_representations("test input")
        
        # Check results
        self.assertIn("transformer_layer_0_output", result)
        self.assertEqual(result["transformer_layer_0_output"].shape, (1, 10, 768))
        
        # Check adapter method was called
        self.extractor.adapter.get_token_representations.assert_called_once()
    
    def test_get_attention_patterns(self):
        """Test attention pattern extraction."""
        # Set up adapter mock
        self.extractor.adapter.get_attention_patterns = MagicMock()
        self.extractor.adapter.get_attention_patterns.return_value = {
            "transformer_layer_0_attention_probs": torch.zeros((1, 12, 10, 10))  # batch, heads, seq, seq
        }
        
        # Test attention pattern extraction
        result = self.extractor.get_attention_patterns("test input")
        
        # Check results
        self.assertIn("transformer_layer_0_attention_probs", result)
        self.assertEqual(result["transformer_layer_0_attention_probs"].shape, (1, 12, 10, 10))
        
        # Check adapter method was called
        self.extractor.adapter.get_attention_patterns.assert_called_once()
    
    def test_get_apa_activations(self):
        """Test APA-specific activation formatting."""
        # Set up adapter mock
        self.extractor.prepare_inputs = MagicMock()
        self.extractor.prepare_inputs.return_value = {
            "input_ids": torch.ones((1, 5), dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long)
        }
        
        self.extractor.adapter.get_token_representations = MagicMock()
        self.extractor.adapter.get_token_representations.return_value = {
            "embedding": torch.zeros((1, 5, 768)),
            "transformer_layer_0_output": torch.zeros((1, 5, 768)),
            "transformer_layer_1_output": torch.zeros((1, 5, 768))
        }
        
        # Test APA activation formatting
        result = self.extractor.get_apa_activations("test input", [0, 1])
        
        # Check results
        self.assertIn("layer_0", result)  # Embedding
        self.assertIn("layer_1", result)  # First transformer layer
        self.assertIn("layer_2", result)  # Second transformer layer
        
        # Check shapes
        self.assertEqual(result["layer_0"].shape, (1, 5, 768))
        
        # Check adapter method was called
        self.extractor.adapter.get_token_representations.assert_called_once()
    
    def test_extract_activations_for_windows(self):
        """Test window-based activation extraction."""
        # Mock apa activations method
        self.extractor.get_apa_activations = MagicMock()
        self.extractor.get_apa_activations.return_value = {
            "layer_0": np.zeros((1, 5, 768)),
            "layer_1": np.zeros((1, 5, 768)),
            "layer_2": np.zeros((1, 5, 768)),
            "layer_3": np.zeros((1, 5, 768)),
            "layer_4": np.zeros((1, 5, 768))
        }
        
        # Mock prepare inputs
        self.extractor.prepare_inputs = MagicMock()
        self.extractor.prepare_inputs.return_value = {
            "input_ids": torch.ones((1, 5), dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long)
        }
        
        # Test window extraction with window size 3, stride 1
        result = self.extractor.extract_activations_for_windows(
            "test input",
            window_size=3,
            stride=1
        )
        
        # Should create windows: [0,1,2], [1,2,3], [2,3,4]
        self.assertIn("window_0_2", result)
        self.assertIn("window_1_3", result)
        self.assertIn("window_2_4", result)
        
        # Check content structure
        self.assertIn("activations", result["window_0_2"])
        self.assertIn("window_layers", result["window_0_2"])
        self.assertIn("metadata", result["window_0_2"])
        
        # Check layer extraction in first window
        self.assertIn("layer_0", result["window_0_2"]["activations"])
        self.assertIn("layer_1", result["window_0_2"]["activations"])
        self.assertIn("layer_2", result["window_0_2"]["activations"])
        
        # Check layers list
        self.assertEqual(result["window_0_2"]["window_layers"], [0, 1, 2])


# Run tests if script is executed directly
if __name__ == "__main__":
    unittest.main()