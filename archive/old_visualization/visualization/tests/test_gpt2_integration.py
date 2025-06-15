"""
Integration tests for GPT-2 visualization pipeline.

This module tests the complete data pipeline from analysis results
to visualization generation, including data loading, processing,
and transformation.
"""

import unittest
import tempfile
import shutil
import os
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add parent directories to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import test utilities
from visualization.tests.fixtures.gpt2_test_data import (
    GPT2TestDataGenerator, 
    create_test_analysis_results, 
    create_comprehensive_test_data
)
from visualization.tests.utils.visualization_validators import (
    PlotlyFigureValidator,
    DashComponentValidator,
    DataStructureValidator,
    validate_figure_basic_structure
)

# Import modules to test
try:
    from visualization.gpt2_token_sankey import (
        extract_token_paths,
        prepare_token_sankey_data,
        create_token_sankey_diagram
    )
    from visualization.gpt2_attention_sankey import (
        extract_attention_flow,
        create_attention_sankey_diagram
    )
    from visualization.gpt2_metrics_tab import (
        create_gpt2_metrics_tab,
        register_gpt2_metrics_callbacks
    )
    from concept_fragmentation.persistence import GPT2AnalysisPersistence
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    # Create mock modules for testing
    extract_token_paths = Mock()
    prepare_token_sankey_data = Mock()
    create_token_sankey_diagram = Mock()
    extract_attention_flow = Mock()
    create_attention_sankey_diagram = Mock()
    create_gpt2_metrics_tab = Mock()
    register_gpt2_metrics_callbacks = Mock()
    GPT2AnalysisPersistence = Mock


class TestGPT2DataPipeline(unittest.TestCase):
    """Test the complete GPT-2 data processing pipeline."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp(prefix="gpt2_test_")
        self.data_generator = GPT2TestDataGenerator(seed=42)
        self.validator = DataStructureValidator()
        
        # Create test data
        self.test_data = create_test_analysis_results(
            num_layers=3,
            seq_length=6,
            batch_size=1,
            num_clusters=3
        )
        
        # Set up mock file system
        self.setup_mock_filesystem()
    
    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def setup_mock_filesystem(self):
        """Create mock file system structure for testing."""
        # Create analysis results directory
        results_dir = os.path.join(self.temp_dir, "analysis_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save test data
        analysis_file = os.path.join(results_dir, "gpt2_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(self.test_data, f, indent=2, default=self._json_serializer)
        
        self.analysis_file_path = analysis_file
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def test_data_loading_pipeline(self):
        """Test complete data loading from analysis results."""
        # Test that test data is properly structured
        self.assertIsInstance(self.test_data, dict)
        
        # Check required top-level keys
        required_keys = [
            'model_type', 'layers', 'activations', 'token_metadata',
            'cluster_labels', 'token_paths', 'attention_data'
        ]
        
        for key in required_keys:
            self.assertIn(key, self.test_data, f"Missing key: {key}")
        
        # Validate data types
        self.assertIsInstance(self.test_data['layers'], list)
        self.assertIsInstance(self.test_data['activations'], dict)
        self.assertIsInstance(self.test_data['token_metadata'], dict)
        self.assertIsInstance(self.test_data['cluster_labels'], dict)
        self.assertIsInstance(self.test_data['token_paths'], dict)
        self.assertIsInstance(self.test_data['attention_data'], dict)
    
    def test_token_metadata_processing(self):
        """Test token metadata extraction and processing."""
        token_metadata = self.test_data['token_metadata']
        
        # Check structure
        required_keys = ['tokens', 'token_ids', 'positions', 'attention_mask']
        for key in required_keys:
            self.assertIn(key, token_metadata, f"Missing token metadata key: {key}")
        
        # Check data consistency
        tokens = token_metadata['tokens']
        token_ids = token_metadata['token_ids']
        positions = token_metadata['positions']
        
        self.assertIsInstance(tokens, list)
        self.assertIsInstance(token_ids, np.ndarray)
        self.assertIsInstance(positions, list)
        
        # Check dimensions match
        batch_size = len(tokens)
        for batch_idx in range(batch_size):
            self.assertEqual(
                len(tokens[batch_idx]), 
                len(token_ids[batch_idx]),
                f"Token count mismatch in batch {batch_idx}"
            )
            self.assertEqual(
                len(tokens[batch_idx]), 
                len(positions[batch_idx]),
                f"Position count mismatch in batch {batch_idx}"
            )
    
    def test_cluster_label_alignment(self):
        """Test alignment between tokens and cluster assignments."""
        token_metadata = self.test_data['token_metadata']
        cluster_labels = self.test_data['cluster_labels']
        
        # Calculate expected total tokens
        batch_size = len(token_metadata['tokens'])
        seq_length = len(token_metadata['tokens'][0])
        expected_total = batch_size * seq_length
        
        # Check each layer has correct number of cluster assignments
        for layer, labels in cluster_labels.items():
            self.assertEqual(
                len(labels), 
                expected_total,
                f"Layer {layer} has wrong number of cluster labels"
            )
            
            # Check cluster IDs are valid
            for cluster_id in labels:
                self.assertIsInstance(cluster_id, (int, np.integer))
                self.assertGreaterEqual(cluster_id, 0)
    
    def test_attention_data_integration(self):
        """Test attention data processing and validation."""
        attention_data = self.test_data['attention_data']
        layers = self.test_data['layers']
        
        # Validate attention data structure
        self.assertTrue(
            self.validator.validate_attention_data(attention_data, layers, check_weights=True),
            f"Attention data validation failed: {self.validator.get_validation_errors()}"
        )
        
        # Check attention statistics
        for layer, layer_data in attention_data.items():
            self.assertIn('entropy', layer_data)
            self.assertIn('head_agreement', layer_data)
            self.assertIn('num_heads', layer_data)
            
            # Check value ranges
            entropy = layer_data['entropy']
            head_agreement = layer_data['head_agreement']
            
            self.assertIsInstance(entropy, float)
            self.assertIsInstance(head_agreement, float)
            self.assertGreaterEqual(entropy, 0.0)
            self.assertGreaterEqual(head_agreement, 0.0)
            self.assertLessEqual(head_agreement, 1.0)
    
    def test_cross_layer_metrics_computation(self):
        """Test cross-layer metric calculations."""
        token_paths = self.test_data['token_paths']
        cluster_metrics = self.test_data['cluster_metrics']
        
        # Validate token paths structure
        self.assertTrue(
            self.validator.validate_token_paths(token_paths, check_path_structure=True),
            f"Token paths validation failed: {self.validator.get_validation_errors()}"
        )
        
        # Check cluster metrics
        layers = self.test_data['layers']
        for layer in layers:
            self.assertIn(layer, cluster_metrics, f"Missing cluster metrics for layer {layer}")
            
            layer_metrics = cluster_metrics[layer]
            self.assertIn('purity', layer_metrics)
            self.assertIn('silhouette', layer_metrics)
            self.assertIn('num_clusters', layer_metrics)
            
            # Check value ranges
            purity = layer_metrics['purity']
            silhouette = layer_metrics['silhouette']
            
            self.assertIsInstance(purity, float)
            self.assertIsInstance(silhouette, float)
            self.assertGreaterEqual(purity, 0.0)
            self.assertLessEqual(purity, 1.0)
            self.assertGreaterEqual(silhouette, -1.0)
            self.assertLessEqual(silhouette, 1.0)
    
    def test_error_handling_malformed_data(self):
        """Test error handling for malformed inputs."""
        # Test with missing keys
        incomplete_data = {'model_type': 'gpt2', 'layers': ['layer_0']}
        
        # The validator should handle missing data gracefully
        result = self.validator.validate_token_paths(incomplete_data.get('token_paths', {}), check_path_structure=True)
        self.assertFalse(result)  # Should return False for empty token paths
        
        # Test with wrong data types
        bad_token_paths = {'0': 'not_a_dict'}
        
        result = self.validator.validate_token_paths(bad_token_paths, check_path_structure=True)
        self.assertFalse(result)
        self.assertGreater(len(self.validator.get_validation_errors()), 0)
        
        # Test with invalid cluster IDs
        bad_attention_data = {
            'layer_0': {
                'entropy': 'not_a_number',
                'head_agreement': 0.5,
                'num_heads': 12
            }
        }
        
        result = self.validator.validate_attention_data(bad_attention_data)
        self.assertFalse(result)


class TestGPT2VisualizationGeneration(unittest.TestCase):
    """Test GPT-2 visualization generation components."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp(prefix="gpt2_viz_test_")
        self.data_generator = GPT2TestDataGenerator(seed=42)
        self.figure_validator = PlotlyFigureValidator()
        
        # Create comprehensive test data
        self.test_data = create_comprehensive_test_data()
    
    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('visualization.gpt2_token_sankey.create_token_sankey_diagram')
    def test_token_sankey_generation(self, mock_create_sankey):
        """Test token Sankey diagram generation."""
        # Mock return value
        mock_figure = Mock()
        mock_figure.data = [Mock()]
        mock_figure.data[0].type = "sankey"
        mock_figure.data[0].node = Mock()
        mock_figure.data[0].node.label = ["Node1", "Node2", "Node3"]
        mock_figure.data[0].link = Mock()
        mock_figure.data[0].link.source = [0, 1]
        mock_figure.data[0].link.target = [1, 2]
        mock_figure.data[0].link.value = [1, 2]
        mock_figure.layout = Mock()
        
        mock_create_sankey.return_value = mock_figure
        
        # Test function call
        token_paths = self.test_data['token_paths']
        selected_tokens = [0, 1, 2]
        
        result = create_token_sankey_diagram(
            token_paths=token_paths,
            selected_tokens=selected_tokens
        )
        
        # Verify function was called
        mock_create_sankey.assert_called_once()
        
        # Validate returned figure structure
        self.assertTrue(validate_figure_basic_structure(result))
        
        # Validate Sankey-specific structure
        is_valid = self.figure_validator.validate_sankey_diagram(
            result,
            expected_nodes=3,
            expected_links=2
        )
        if not is_valid:
            print("Sankey validation errors:", self.figure_validator.get_validation_errors())
        
        self.assertTrue(is_valid)
    
    @patch('visualization.gpt2_attention_sankey.create_attention_sankey_diagram')
    def test_attention_flow_generation(self, mock_create_attention):
        """Test attention flow visualization generation."""
        # Mock return value
        mock_figure = Mock()
        mock_figure.data = [Mock()]
        mock_figure.data[0].type = "sankey"
        mock_figure.data[0].node = Mock()
        mock_figure.data[0].node.label = ["Head1", "Head2", "Head3"]
        mock_figure.data[0].link = Mock()
        mock_figure.data[0].link.source = [0, 1]
        mock_figure.data[0].link.target = [1, 2]
        mock_figure.data[0].link.value = [0.5, 0.8]
        mock_figure.layout = Mock()
        
        mock_create_attention.return_value = mock_figure
        
        # Test function call
        attention_data = self.test_data['attention_data']
        token_metadata = self.test_data['token_metadata']
        
        result = create_attention_sankey_diagram(
            attention_data=attention_data,
            token_metadata=token_metadata
        )
        
        # Verify function was called
        mock_create_attention.assert_called_once()
        
        # Validate returned figure structure
        self.assertTrue(validate_figure_basic_structure(result))
    
    def test_visualization_data_consistency(self):
        """Test consistency between different visualization data sources."""
        token_paths = self.test_data['token_paths']
        attention_data = self.test_data['attention_data']
        token_metadata = self.test_data['token_metadata']
        
        # Check token count consistency
        num_tokens_metadata = sum(len(batch) for batch in token_metadata['tokens'])
        num_tokens_paths = len(token_paths)
        
        self.assertEqual(
            num_tokens_metadata, 
            num_tokens_paths,
            "Token count mismatch between metadata and paths"
        )
        
        # Check layer consistency
        layers_attention = set(attention_data.keys())
        layers_main = set(self.test_data['layers'])
        
        self.assertEqual(
            layers_attention,
            layers_main,
            "Layer mismatch between attention data and main data"
        )
    
    def test_visualization_parameter_validation(self):
        """Test parameter validation for visualization functions."""
        # Test with invalid parameters (should handle gracefully)
        
        # Empty token paths
        with patch('visualization.gpt2_token_sankey.create_token_sankey_diagram') as mock_create:
            mock_create.side_effect = ValueError("Invalid token paths")
            
            with self.assertRaises(ValueError):
                create_token_sankey_diagram(token_paths={})
        
        # Invalid layer selection
        with patch('visualization.gpt2_attention_sankey.create_attention_sankey_diagram') as mock_create:
            mock_create.side_effect = KeyError("Invalid layer")
            
            with self.assertRaises(KeyError):
                create_attention_sankey_diagram(
                    attention_data=self.test_data['attention_data'],
                    token_metadata=self.test_data['token_metadata'],
                    selected_layers=['nonexistent_layer']
                )


class TestGPT2PersistenceIntegration(unittest.TestCase):
    """Test integration between visualization and persistence systems."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp(prefix="gpt2_persist_test_")
        self.data_generator = GPT2TestDataGenerator(seed=42)
        
        # Create test data
        self.test_data = create_test_analysis_results()
        
        # Create persistence manager
        self.persistence = GPT2AnalysisPersistence(
            base_dir=self.temp_dir,
            enable_cache=True,
            cache_ttl=3600
        )
    
    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_analysis_save_load_cycle(self):
        """Test complete save and load cycle for analysis data."""
        # Save analysis
        analysis_id = self.persistence.save_analysis_results(
            analysis_data=self.test_data,
            model_name="gpt2-test",
            input_text="Test input"
        )
        
        self.assertIsNotNone(analysis_id)
        self.assertIsInstance(analysis_id, str)
        
        # Load analysis
        loaded_data = self.persistence.load_analysis_results(analysis_id)
        
        self.assertIsNotNone(loaded_data)
        self.assertIn('analysis_data', loaded_data)
        self.assertIn('metadata', loaded_data)
        
        # Verify data integrity
        original_layers = self.test_data['layers']
        loaded_layers = loaded_data['analysis_data']['layers']
        
        self.assertEqual(original_layers, loaded_layers)
    
    def test_visualization_state_persistence(self):
        """Test saving and loading visualization states."""
        # First save an analysis
        analysis_id = self.persistence.save_analysis_results(
            analysis_data=self.test_data,
            model_name="gpt2-test",
            input_text="Test input"
        )
        
        # Create visualization config
        viz_config = {
            "selected_tokens": [0, 1, 2],
            "selected_layers": ["layer_0", "layer_1"],
            "highlight_paths": True,
            "color_scheme": "viridis"
        }
        
        # Save visualization state
        state_id = self.persistence.save_visualization_state(
            analysis_id=analysis_id,
            visualization_config=viz_config,
            visualization_type="token_sankey",
            state_name="test_view"
        )
        
        self.assertIsNotNone(state_id)
        
        # Load visualization state
        loaded_state = self.persistence.load_visualization_state(state_id)
        
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state['config'], viz_config)
        self.assertEqual(loaded_state['visualization_type'], "token_sankey")
    
    def test_export_with_visualizations(self):
        """Test exporting analysis with visualization states."""
        # Save analysis
        analysis_id = self.persistence.save_analysis_results(
            analysis_data=self.test_data,
            model_name="gpt2-test",
            input_text="Test input"
        )
        
        # Save visualization state
        viz_config = {"test": "config"}
        self.persistence.save_visualization_state(
            analysis_id=analysis_id,
            visualization_config=viz_config,
            visualization_type="test_viz",
            state_name="test"
        )
        
        # Export with visualizations
        export_path = self.persistence.export_analysis(
            analysis_id=analysis_id,
            export_format="json",
            include_visualizations=True
        )
        
        self.assertTrue(os.path.exists(export_path))
        
        # Verify export contents
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        self.assertIn('analysis_results', exported_data)
        self.assertIn('visualization_states', exported_data)
        self.assertIn('export_metadata', exported_data)


class TestGPT2PerformanceMetrics(unittest.TestCase):
    """Test performance characteristics of GPT-2 visualizations."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.data_generator = GPT2TestDataGenerator(seed=42)
    
    def test_large_dataset_handling(self):
        """Test performance with large datasets."""
        # Create large test dataset
        large_data = create_test_analysis_results(
            num_layers=12,
            seq_length=512,
            batch_size=4,
            num_clusters=10
        )
        
        # Test data structure validation performance
        import time
        
        validator = DataStructureValidator()
        
        start_time = time.time()
        result = validator.validate_token_paths(
            large_data['token_paths'], 
            check_path_structure=True
        )
        validation_time = time.time() - start_time
        
        self.assertTrue(result, f"Large dataset validation failed: {validator.get_validation_errors()}")
        self.assertLess(validation_time, 5.0, "Validation took too long")  # Should be under 5 seconds
    
    def test_memory_usage_patterns(self):
        """Test memory usage patterns for different data sizes."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple datasets of increasing size
        datasets = []
        for size_factor in [1, 2, 4]:
            data = create_test_analysis_results(
                num_layers=2 * size_factor,
                seq_length=8 * size_factor,
                batch_size=1,
                num_clusters=3
            )
            datasets.append(data)
        
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for test data)
        self.assertLess(memory_increase, 100 * 1024 * 1024, "Excessive memory usage")
        
        # Clean up
        del datasets


def create_test_suite():
    """Create test suite with all integration tests."""
    suite = unittest.TestSuite()
    
    # Data pipeline tests
    suite.addTest(unittest.makeSuite(TestGPT2DataPipeline))
    
    # Visualization generation tests
    suite.addTest(unittest.makeSuite(TestGPT2VisualizationGeneration))
    
    # Persistence integration tests
    suite.addTest(unittest.makeSuite(TestGPT2PersistenceIntegration))
    
    # Performance tests
    suite.addTest(unittest.makeSuite(TestGPT2PerformanceMetrics))
    
    return suite


if __name__ == '__main__':
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)