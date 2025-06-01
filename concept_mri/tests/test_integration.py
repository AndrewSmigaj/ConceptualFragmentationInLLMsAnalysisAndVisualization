"""
Integration tests for Concept MRI.
Tests component interactions and data flow.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concept_mri.tests.test_data_generators import TestDataGenerator


class TestComponentInteractions(unittest.TestCase):
    """Test interactions between components."""
    
    def setUp(self):
        """Set up test state."""
        self.test_state = TestDataGenerator.generate_complete_test_state()
    
    def test_window_config_to_sankey(self):
        """Test window configuration propagates to Sankey."""
        from concept_mri.components.visualizations.sankey_wrapper import SankeyWrapper
        
        sankey = SankeyWrapper()
        
        # Test with window configuration
        window_config = self.test_state['window-config-store']
        clustering_data = self.test_state['clustering-store']
        path_data = self.test_state['path-analysis-store']
        labels = self.test_state['cluster-labels-store']
        
        # Generate Sankey with window filter
        fig_data = sankey.generate_sankey(
            clustering_data=clustering_data,
            path_data=path_data,
            cluster_labels=labels,
            window_config=window_config,
            config={'window': 'Early', 'hierarchy': 'meso'}
        )
        
        # Verify figure was generated
        self.assertIsNotNone(fig_data)
        self.assertIn('data', fig_data)
    
    def test_hierarchy_to_clustering(self):
        """Test hierarchy level affects clustering parameters."""
        from concept_mri.components.controls.clustering_panel import get_k_for_hierarchy
        
        # Test different hierarchy levels
        n_samples = 100
        
        # Get k values for each level
        k_values = {
            'macro': get_k_for_hierarchy(1, n_samples),
            'meso': get_k_for_hierarchy(2, n_samples),
            'micro': get_k_for_hierarchy(3, n_samples)
        }
        
        # Verify progression
        self.assertLess(k_values['macro'], k_values['meso'])
        self.assertLess(k_values['meso'], k_values['micro'])
    
    def test_clustering_to_visualizations(self):
        """Test clustering data flows to visualizations."""
        from concept_mri.components.visualizations.stepped_trajectory import SteppedTrajectoryVisualization
        from concept_mri.components.visualizations.cluster_cards import ClusterCards
        
        clustering_data = self.test_state['clustering-store']
        
        # Test stepped trajectory
        stepped = SteppedTrajectoryVisualization()
        fig = stepped.generate_stepped_plot(clustering_data)
        self.assertIsNotNone(fig)
        
        # Test cluster cards
        cards = ClusterCards()
        card_list = cards.create_cluster_cards(
            clustering_data,
            layer='layer_0',
            card_type='standard',
            show_options=['stats']
        )
        self.assertGreater(len(card_list), 0)
    
    def test_ets_data_flow(self):
        """Test ETS-specific data flow through components."""
        # Generate ETS clustering data
        model_data = TestDataGenerator.generate_model_data()
        ets_clustering = TestDataGenerator.generate_clustering_data(
            model_data,
            algorithm='ets'
        )
        
        # Verify ETS-specific fields
        layer_data = ets_clustering['clusters_per_layer']['layer_0']
        self.assertIn('thresholds', layer_data)
        self.assertIn('statistics', layer_data)
        
        # Test ETS card creation
        from concept_mri.components.visualizations.cluster_cards import ClusterCards
        
        cards = ClusterCards()
        cluster_data = {
            'size': 50,
            'thresholds': layer_data['thresholds'][:20],
            'active_dimensions': list(range(10))
        }
        
        ets_card = cards.create_ets_card(
            cluster_id=0,
            cluster_data=cluster_data,
            layer_name='layer_0',
            show_options=['stats']
        )
        
        self.assertIsNotNone(ets_card)


class TestDataPersistence(unittest.TestCase):
    """Test data persistence across components."""
    
    def test_store_data_format(self):
        """Test data format in stores."""
        test_state = TestDataGenerator.generate_complete_test_state()
        
        # Verify all required stores
        required_stores = [
            'model-store',
            'clustering-store',
            'window-config-store',
            'path-analysis-store',
            'cluster-labels-store',
            'hierarchy-results-store'
        ]
        
        for store in required_stores:
            self.assertIn(store, test_state)
            self.assertIsNotNone(test_state[store])
    
    def test_model_data_structure(self):
        """Test model data structure."""
        model_data = TestDataGenerator.generate_model_data()
        
        # Check required fields
        self.assertIn('model_id', model_data)
        self.assertIn('model_loaded', model_data)
        self.assertIn('num_layers', model_data)
        self.assertIn('activations', model_data)
        self.assertIn('dataset', model_data)
        
        # Check activations structure
        activations = model_data['activations']
        self.assertEqual(len(activations), model_data['num_layers'])
        
        # Check each layer has correct shape
        for layer_name, layer_data in activations.items():
            self.assertEqual(len(layer_data.shape), 2)  # 2D array
    
    def test_clustering_data_structure(self):
        """Test clustering data structure."""
        model_data = TestDataGenerator.generate_model_data()
        clustering_data = TestDataGenerator.generate_clustering_data(model_data)
        
        # Check required fields
        self.assertIn('algorithm', clustering_data)
        self.assertIn('hierarchy', clustering_data)
        self.assertIn('completed', clustering_data)
        self.assertIn('clusters_per_layer', clustering_data)
        
        # Check each layer has clustering results
        for layer_name, layer_data in clustering_data['clusters_per_layer'].items():
            self.assertIn('labels', layer_data)
            self.assertIn('n_clusters', layer_data)
            self.assertEqual(
                len(layer_data['labels']), 
                model_data['dataset']['num_samples']
            )


class TestCallbackChain(unittest.TestCase):
    """Test callback chains and state updates."""
    
    @patch('dash.callback_context')
    def test_window_update_chain(self, mock_ctx):
        """Test window configuration update chain."""
        # Simulate window configuration update
        mock_ctx.triggered = [{'prop_id': 'apply-preset-btn.n_clicks'}]
        
        from concept_mri.components.controls.layer_window_manager import compute_preset_windows
        
        # Test preset application
        windows = compute_preset_windows('gpt2', 12)
        
        # Verify windows structure
        self.assertEqual(len(windows), 3)
        for window_name, window_data in windows.items():
            self.assertIn('start', window_data)
            self.assertIn('end', window_data)
            self.assertIn('color', window_data)
    
    def test_hierarchy_update_chain(self):
        """Test hierarchy level update chain."""
        from concept_mri.components.controls.clustering_panel import get_k_for_hierarchy
        
        # Simulate hierarchy level changes
        hierarchy_levels = [1, 2, 3]  # macro, meso, micro
        n_samples = 100
        
        k_values = []
        for level in hierarchy_levels:
            k = get_k_for_hierarchy(level, n_samples)
            k_values.append(k)
        
        # Verify k values change appropriately
        self.assertEqual(len(set(k_values)), 3)  # All different
        self.assertEqual(sorted(k_values), k_values)  # Increasing order


class TestErrorHandling(unittest.TestCase):
    """Test error handling across components."""
    
    def test_empty_data_handling(self):
        """Test components handle empty data gracefully."""
        from concept_mri.components.visualizations.sankey_wrapper import SankeyWrapper
        from concept_mri.components.visualizations.stepped_trajectory import SteppedTrajectoryVisualization
        from concept_mri.components.visualizations.cluster_cards import ClusterCards
        
        # Test Sankey with empty data
        sankey = SankeyWrapper()
        fig = sankey.generate_sankey({}, {}, {})
        self.assertIsNotNone(fig)  # Should return empty figure
        
        # Test stepped trajectory with empty data
        stepped = SteppedTrajectoryVisualization()
        fig = stepped.generate_stepped_plot({})
        self.assertIsNotNone(fig)
        
        # Test cluster cards with empty data
        cards = ClusterCards()
        card_list = cards.create_cluster_cards({}, 'layer_0', 'standard', [])
        self.assertEqual(len(card_list), 1)  # Should return empty card
    
    def test_invalid_parameters(self):
        """Test components handle invalid parameters."""
        from concept_mri.components.controls.clustering_panel import get_k_for_hierarchy
        
        # Test with invalid hierarchy level
        k = get_k_for_hierarchy(99, 100)  # Invalid level
        self.assertIsInstance(k, int)  # Should still return valid k
        
        # Test with very small sample size
        k = get_k_for_hierarchy(1, 5)  # Only 5 samples
        self.assertGreaterEqual(k, 2)  # Should be at least 2
        self.assertLessEqual(k, 2)  # But not more than n/2


def run_integration_tests():
    """Run all integration tests."""
    unittest.main(argv=[''], exit=False)


if __name__ == '__main__':
    run_integration_tests()