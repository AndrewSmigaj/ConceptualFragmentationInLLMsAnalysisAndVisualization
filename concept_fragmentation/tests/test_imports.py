"""
Test that all modules can be imported successfully.
This is a basic smoke test to verify the project structure.
"""

import unittest

class TestImports(unittest.TestCase):
    """Test that all modules can be imported successfully."""
    
    def test_model_imports(self):
        """Test that model modules can be imported."""
        try:
            from concept_fragmentation.models.feedforward import FeedforwardNetwork
            from concept_fragmentation.models.regularizers import CohesionRegularizer
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import model modules: {str(e)}")
    
    def test_hook_imports(self):
        """Test that hook modules can be imported."""
        try:
            from concept_fragmentation.hooks.activation_hooks import (
                ActivationHook, 
                get_activation_hooks, 
                capture_activations,
                get_neuron_importance
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import hook modules: {str(e)}")
    
    def test_metric_imports(self):
        """Test that metric modules can be imported."""
        try:
            from concept_fragmentation.metrics.cluster_entropy import compute_cluster_entropy
            from concept_fragmentation.metrics.subspace_angle import compute_subspace_angle
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import metric modules: {str(e)}")
    
    def test_config_import(self):
        """Test that config can be imported."""
        try:
            from concept_fragmentation import config
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import config: {str(e)}")

if __name__ == '__main__':
    unittest.main() 