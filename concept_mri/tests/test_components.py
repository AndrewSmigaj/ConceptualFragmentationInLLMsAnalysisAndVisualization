"""
Component tests for Concept MRI.
Tests individual components in isolation.
"""
import unittest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concept_mri.tests.test_data_generators import TestDataGenerator
from concept_fragmentation.metrics.explainable_threshold_similarity import (
    compute_ets_clustering,
    compute_dimension_thresholds,
    explain_ets_similarity
)


class TestETSClustering(unittest.TestCase):
    """Test ETS clustering functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = TestDataGenerator.generate_model_data(
            num_layers=3,
            layer_size=100,
            num_samples=50
        )
        self.activations = list(self.test_data['activations'].values())[0]
    
    def test_dimension_thresholds(self):
        """Test threshold calculation."""
        thresholds = compute_dimension_thresholds(
            self.activations,
            threshold_percentile=0.1
        )
        
        # Check output shape
        self.assertEqual(len(thresholds), self.activations.shape[1])
        
        # Check all thresholds are positive
        self.assertTrue(np.all(thresholds > 0))
        
        # Check minimum threshold
        self.assertTrue(np.all(thresholds >= 1e-5))
    
    def test_ets_clustering(self):
        """Test ETS clustering."""
        labels, thresholds = compute_ets_clustering(
            self.activations,
            threshold_percentile=0.1
        )
        
        # Check output shapes
        self.assertEqual(len(labels), self.activations.shape[0])
        self.assertEqual(len(thresholds), self.activations.shape[1])
        
        # Check valid cluster labels
        self.assertTrue(np.all(labels >= 0))
        self.assertGreater(len(np.unique(labels)), 1)  # More than 1 cluster
    
    def test_ets_explanation(self):
        """Test ETS similarity explanation."""
        point1 = self.activations[0]
        point2 = self.activations[1]
        thresholds = compute_dimension_thresholds(self.activations)
        
        explanation = explain_ets_similarity(
            point1, point2, thresholds
        )
        
        # Check explanation structure
        self.assertIn('is_similar', explanation)
        self.assertIn('dimensions', explanation)
        self.assertIn('distinguishing_dimensions', explanation)
        self.assertEqual(len(explanation['dimensions']), len(thresholds))


class TestHierarchyControl(unittest.TestCase):
    """Test hierarchy control functionality."""
    
    def test_k_calculation(self):
        """Test k calculation for different hierarchy levels."""
        from concept_mri.components.controls.clustering_panel import get_k_for_hierarchy
        
        n_samples = 100
        
        # Test macro level
        k_macro = get_k_for_hierarchy(1, n_samples, 'kmeans')
        self.assertLessEqual(k_macro, 10)
        self.assertGreaterEqual(k_macro, 2)
        
        # Test meso level
        k_meso = get_k_for_hierarchy(2, n_samples, 'kmeans')
        self.assertGreater(k_meso, k_macro)
        self.assertLessEqual(k_meso, 20)
        
        # Test micro level
        k_micro = get_k_for_hierarchy(3, n_samples, 'kmeans')
        self.assertGreater(k_micro, k_meso)
        self.assertLessEqual(k_micro, 50)
        
        # Test ETS percentiles
        p_macro = get_k_for_hierarchy(1, n_samples, 'ets')
        p_meso = get_k_for_hierarchy(2, n_samples, 'ets')
        p_micro = get_k_for_hierarchy(3, n_samples, 'ets')
        
        self.assertEqual(p_macro, 30)
        self.assertEqual(p_meso, 10)
        self.assertEqual(p_micro, 5)


class TestWindowManager(unittest.TestCase):
    """Test Layer Window Manager functionality."""
    
    def test_preset_windows(self):
        """Test window preset calculations."""
        from concept_mri.components.controls.layer_window_manager import compute_preset_windows
        
        # Test GPT-2 preset
        windows = compute_preset_windows('gpt2', 12)
        self.assertEqual(len(windows), 3)
        self.assertIn('Early', windows)
        self.assertIn('Middle', windows)
        self.assertIn('Late', windows)
        
        # Test thirds preset
        windows = compute_preset_windows('thirds', 12)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows['Early']['start'], 0)
        self.assertEqual(windows['Late']['end'], 11)
        
        # Test halves preset
        windows = compute_preset_windows('halves', 10)
        self.assertEqual(len(windows), 2)
        self.assertIn('First Half', windows)
        self.assertIn('Second Half', windows)
    
    def test_window_detection_metrics(self):
        """Test window detection metric computation."""
        from concept_mri.components.controls.window_detection_utils import WindowDetectionMetrics
        
        test_data = TestDataGenerator.generate_model_data(num_layers=12)
        activations = test_data['activations']
        
        metrics = WindowDetectionMetrics.compute_boundary_metrics(
            activations_dict=activations,
            compute_all=False
        )
        
        # Check stability metrics
        self.assertIn('stability', metrics)
        if metrics['stability'] is not None:
            self.assertEqual(len(metrics['stability']), 11)  # n_layers - 1


class TestSankeyFiltering(unittest.TestCase):
    """Test Sankey diagram window filtering."""
    
    def test_path_filtering(self):
        """Test path filtering by window."""
        from concept_mri.components.visualizations.sankey_wrapper import SankeyWrapper
        
        sankey = SankeyWrapper()
        
        # Create test paths
        paths = [
            {'transitions': [{'layer': 0}, {'layer': 1}, {'layer': 2}]},
            {'transitions': [{'layer': 4}, {'layer': 5}, {'layer': 6}]},
            {'transitions': [{'layer': 8}, {'layer': 9}, {'layer': 10}]},
            {'transitions': [{'layer': 0}, {'layer': 5}, {'layer': 10}]},
        ]
        
        # Filter for early window (0-3)
        filtered = sankey._filter_paths_by_window(paths, 0, 3)
        self.assertEqual(len(filtered), 1)
        
        # Filter for middle window (4-7)
        filtered = sankey._filter_paths_by_window(paths, 4, 7)
        self.assertEqual(len(filtered), 1)


class TestClusterCards(unittest.TestCase):
    """Test cluster card generation."""
    
    def test_standard_card_creation(self):
        """Test standard cluster card creation."""
        from concept_mri.components.visualizations.cluster_cards import ClusterCards
        
        cards_component = ClusterCards()
        
        cluster_data = {
            'size': 50,
            'cohesion': 0.85,
            'separation': 0.72,
            'label': 'Test Cluster',
            'total_samples': 100
        }
        
        card = cards_component.create_standard_card(
            cluster_id=0,
            cluster_data=cluster_data,
            layer_name='layer_0',
            show_options=['stats', 'ci', 'transitions']
        )
        
        # Check card structure
        self.assertIsNotNone(card)
        self.assertEqual(len(card.children), 2)  # Header and body
    
    def test_ets_card_creation(self):
        """Test ETS cluster card creation."""
        from concept_mri.components.visualizations.cluster_cards import ClusterCards
        
        cards_component = ClusterCards()
        
        cluster_data = {
            'size': 50,
            'cohesion': 0.85,
            'separation': 0.72,
            'label': 'Test Cluster',
            'total_samples': 100,
            'thresholds': np.random.uniform(0.01, 0.1, 20).tolist(),
            'active_dimensions': list(range(10))
        }
        
        card = cards_component.create_ets_card(
            cluster_id=0,
            cluster_data=cluster_data,
            layer_name='layer_0',
            show_options=['stats']
        )
        
        # Check card has ETS-specific content
        self.assertIsNotNone(card)


class TestVisualizationModes(unittest.TestCase):
    """Test different visualization modes."""
    
    def test_stepped_trajectory_modes(self):
        """Test stepped trajectory visualization modes."""
        from concept_mri.components.visualizations.stepped_trajectory import SteppedTrajectoryVisualization
        
        viz = SteppedTrajectoryVisualization()
        
        # Generate test data
        test_state = TestDataGenerator.generate_complete_test_state()
        clustering_data = test_state['clustering-store']
        
        # Test individual mode
        fig_data = viz.generate_stepped_plot(
            clustering_data,
            config={'mode': 'individual', 'n_samples': 10}
        )
        self.assertIn('data', fig_data)
        
        # Test aggregated mode
        fig_data = viz.generate_stepped_plot(
            clustering_data,
            config={'mode': 'aggregated'}
        )
        self.assertIn('data', fig_data)
        
        # Test heatmap mode
        fig_data = viz.generate_stepped_plot(
            clustering_data,
            config={'mode': 'heatmap'}
        )
        self.assertIn('data', fig_data)


def run_component_tests():
    """Run all component tests."""
    unittest.main(argv=[''], exit=False)


if __name__ == '__main__':
    run_component_tests()