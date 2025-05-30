"""Comprehensive tests for unified TrajectoryVisualizer."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from concept_fragmentation.visualization.trajectory import TrajectoryVisualizer
from concept_fragmentation.visualization.configs import TrajectoryConfig
from concept_fragmentation.visualization.exceptions import VisualizationError, InvalidDataError


class TestTrajectoryVisualizer:
    """Test suite for TrajectoryVisualizer class."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 50
        n_layers = 12
        n_features = 768  # Typical transformer hidden size
        
        return {
            'activations': np.random.randn(n_samples, n_layers, n_features),
            'labels': {
                'class_labels': np.random.randint(0, 3, n_samples),
                'cluster_labels': np.random.randint(0, 5, (n_samples, n_layers)),
                'path_labels': np.random.randint(0, 10, n_samples),
                'pos_labels': np.random.choice(['NN', 'VB', 'JJ', 'RB'], n_samples),
                'metric_values': np.random.rand(n_samples)
            },
            'metadata': {
                'dataset': 'test_dataset',
                'model': 'test_model'
            },
            'windows': {
                'early': [0, 1, 2, 3],
                'middle': [4, 5, 6, 7],
                'late': [8, 9, 10, 11]
            },
            'cluster_centers': np.random.randn(n_layers, 5, n_features)  # 5 clusters per layer
        }
    
    @pytest.fixture
    def small_data(self) -> Dict[str, Any]:
        """Create small data for testing without reduction."""
        return {
            'activations': np.random.randn(10, 4, 3),  # Already 3D
            'labels': {
                'class_labels': np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
            }
        }
    
    @pytest.fixture
    def trajectory_visualizer(self) -> TrajectoryVisualizer:
        """Create TrajectoryVisualizer instance with default config."""
        config = TrajectoryConfig(
            reduction_method='pca',  # Faster than UMAP for tests
            dimensions=3,
            backend='plotly'
        )
        return TrajectoryVisualizer(config)
    
    def test_initialization(self):
        """Test TrajectoryVisualizer initialization."""
        # Test with default config
        vis = TrajectoryVisualizer()
        assert vis.config.reduction_method == 'umap'
        assert vis.config.dimensions == 3
        assert vis.config.backend == 'plotly'
        
        # Test with custom config
        config = TrajectoryConfig(
            reduction_method='tsne',
            dimensions=2,
            backend='matplotlib'
        )
        vis = TrajectoryVisualizer(config)
        assert vis.config.reduction_method == 'tsne'
        assert vis.config.dimensions == 2
        assert vis.config.backend == 'matplotlib'
    
    def test_initialization_missing_backend(self):
        """Test initialization with missing backend."""
        config = TrajectoryConfig(backend='plotly')
        
        with patch('concept_fragmentation.visualization.trajectory.HAS_PLOTLY', False):
            with pytest.raises(ImportError, match="Plotly is required"):
                TrajectoryVisualizer(config)
    
    def test_validate_trajectory_data(self, trajectory_visualizer):
        """Test data validation."""
        # Valid data should pass
        valid_data = {
            'activations': np.random.randn(10, 5, 100)
        }
        trajectory_visualizer._validate_trajectory_data(valid_data)
        
        # Invalid data should raise errors
        with pytest.raises(InvalidDataError, match="Data must be a dictionary"):
            trajectory_visualizer._validate_trajectory_data("not a dict")
            
        with pytest.raises(InvalidDataError, match="Missing 'activations'"):
            trajectory_visualizer._validate_trajectory_data({})
            
        with pytest.raises(InvalidDataError, match="must be a numpy array"):
            trajectory_visualizer._validate_trajectory_data({'activations': [[1, 2, 3]]})
            
        with pytest.raises(InvalidDataError, match="must be 3D"):
            trajectory_visualizer._validate_trajectory_data({
                'activations': np.random.randn(10, 100)  # 2D
            })
    
    def test_prepare_activations(self, trajectory_visualizer, sample_data):
        """Test activation preparation and windowing."""
        # Test without window
        activations, layer_indices = trajectory_visualizer._prepare_activations(
            sample_data, window=None
        )
        assert activations.shape == sample_data['activations'].shape
        assert layer_indices == list(range(12))
        
        # Test with window
        activations, layer_indices = trajectory_visualizer._prepare_activations(
            sample_data, window='early'
        )
        assert activations.shape == (50, 4, 768)  # Only early layers
        assert layer_indices == [0, 1, 2, 3]
        
        # Test with max_samples
        trajectory_visualizer.config.max_samples = 20
        activations, layer_indices = trajectory_visualizer._prepare_activations(
            sample_data, window=None
        )
        assert activations.shape[0] == 20
    
    def test_reduce_dimensions_pca(self, trajectory_visualizer, sample_data):
        """Test PCA dimensionality reduction."""
        trajectory_visualizer.config.reduction_method = 'pca'
        trajectory_visualizer.config.dimensions = 3
        
        activations = sample_data['activations'][:10, :4, :]  # Smaller for speed
        reduced = trajectory_visualizer._reduce_dimensions(activations)
        
        assert reduced.shape == (10, 4, 3)
        assert not np.allclose(reduced, activations[:, :, :3])  # Should be transformed
        
        # Test caching
        reduced2 = trajectory_visualizer._reduce_dimensions(activations)
        np.testing.assert_array_equal(reduced, reduced2)
    
    def test_reduce_dimensions_none(self, trajectory_visualizer, small_data):
        """Test no dimensionality reduction."""
        trajectory_visualizer.config.reduction_method = 'none'
        
        activations = small_data['activations']
        reduced = trajectory_visualizer._reduce_dimensions(activations)
        
        np.testing.assert_array_equal(reduced, activations)
    
    @pytest.mark.skipif(True, reason="UMAP import may not be available")
    def test_reduce_dimensions_umap(self, trajectory_visualizer, sample_data):
        """Test UMAP dimensionality reduction."""
        trajectory_visualizer.config.reduction_method = 'umap'
        trajectory_visualizer.config.dimensions = 2
        
        activations = sample_data['activations'][:5, :3, :]  # Very small for speed
        reduced = trajectory_visualizer._reduce_dimensions(activations)
        
        assert reduced.shape == (5, 3, 2)
    
    def test_get_colors(self, trajectory_visualizer, sample_data):
        """Test color mapping generation."""
        # Test class-based coloring
        trajectory_visualizer.config.color_by = 'class'
        colors, _ = trajectory_visualizer._get_colors(sample_data, 50)
        assert len(colors) == 50
        assert all(isinstance(c, str) for c in colors)
        
        # Test with custom palette
        trajectory_visualizer.config.color_palette = ['red', 'blue', 'green']
        colors, _ = trajectory_visualizer._get_colors(sample_data, 50)
        assert all(c in ['red', 'blue', 'green'] for c in colors)
        
        # Test default coloring (no labels)
        trajectory_visualizer.config.color_by = 'nonexistent'
        colors, _ = trajectory_visualizer._get_colors({}, 10)
        assert len(colors) == 10
    
    def test_get_layer_positions(self, trajectory_visualizer):
        """Test layer position calculation."""
        # Test stepped separation
        trajectory_visualizer.config.layer_separation = 'stepped'
        trajectory_visualizer.config.layer_height = 0.2
        positions = trajectory_visualizer._get_layer_positions(5)
        np.testing.assert_array_almost_equal(positions, [0, 0.2, 0.4, 0.6, 0.8])
        
        # Test sequential separation
        trajectory_visualizer.config.layer_separation = 'sequential'
        positions = trajectory_visualizer._get_layer_positions(5)
        np.testing.assert_array_equal(positions, [0, 0.25, 0.5, 0.75, 1.0])
        
        # Test no separation
        trajectory_visualizer.config.layer_separation = 'none'
        positions = trajectory_visualizer._get_layer_positions(5)
        np.testing.assert_array_equal(positions, [0, 0, 0, 0, 0])
    
    @patch('concept_fragmentation.visualization.trajectory.go')
    def test_create_figure_plotly(self, mock_go, trajectory_visualizer, small_data):
        """Test Plotly figure creation."""
        mock_figure = Mock()
        mock_go.Figure.return_value = mock_figure
        mock_go.Scatter3d = Mock()
        
        fig = trajectory_visualizer.create_figure(small_data)
        
        assert fig == mock_figure
        mock_go.Figure.assert_called_once()
        # Should have created scatter traces
        assert mock_go.Scatter3d.call_count > 0
    
    @patch('concept_fragmentation.visualization.trajectory.plt')
    def test_create_figure_matplotlib(self, mock_plt, trajectory_visualizer, small_data):
        """Test Matplotlib figure creation."""
        trajectory_visualizer.config.backend = 'matplotlib'
        trajectory_visualizer.config.dimensions = 2
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        fig = trajectory_visualizer.create_figure(small_data)
        
        assert fig == mock_fig
        mock_plt.subplots.assert_called_once()
        # Should have plotted lines
        assert mock_ax.plot.call_count > 0
    
    def test_create_figure_with_window(self, trajectory_visualizer, sample_data):
        """Test figure creation with windowing."""
        trajectory_visualizer.config.reduction_method = 'pca'
        
        # Mock the plotting backend
        with patch.object(trajectory_visualizer, '_create_3d_plotly') as mock_create:
            mock_create.return_value = Mock()
            
            fig = trajectory_visualizer.create_figure(sample_data, window='middle')
            
            # Check that reduced data has correct shape (middle window = 4 layers)
            call_args = mock_create.call_args[0]
            reduced_data = call_args[0]
            assert reduced_data.shape[1] == 4  # Middle window has 4 layers
    
    def test_get_title(self, trajectory_visualizer, sample_data):
        """Test title generation."""
        trajectory_visualizer.config.color_by = 'cluster'
        title = trajectory_visualizer._get_title(sample_data)
        
        assert 'test_dataset' in title
        assert '3D Trajectory Visualization' in title
        assert 'colored by cluster' in title
    
    @patch('concept_fragmentation.visualization.trajectory.go')
    def test_save_figure_plotly(self, mock_go, trajectory_visualizer, tmp_path):
        """Test saving Plotly figures."""
        mock_fig = Mock()
        
        # Test HTML save
        output_path = tmp_path / 'test.html'
        trajectory_visualizer.save_figure(mock_fig, output_path, format='html')
        mock_fig.write_html.assert_called_once_with(str(output_path))
        
        # Test image save
        output_path = tmp_path / 'test.png'
        trajectory_visualizer.save_figure(mock_fig, output_path, format='png')
        mock_fig.write_image.assert_called_once_with(str(output_path), format='png')
    
    @patch('concept_fragmentation.visualization.trajectory.plt')
    def test_save_figure_matplotlib(self, mock_plt, trajectory_visualizer, tmp_path):
        """Test saving Matplotlib figures."""
        trajectory_visualizer.config.backend = 'matplotlib'
        mock_fig = Mock()
        
        # Test PNG save
        output_path = tmp_path / 'test.png'
        trajectory_visualizer.save_figure(mock_fig, output_path, format='png')
        mock_fig.savefig.assert_called_once()
    
    def test_error_handling(self, trajectory_visualizer):
        """Test error handling in create_figure."""
        # Test with invalid window
        data = {'activations': np.random.randn(5, 3, 10), 'windows': {'early': [0, 1]}}
        
        with pytest.raises(InvalidDataError, match="Window 'invalid' not found"):
            trajectory_visualizer.create_figure(data, window='invalid')
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid reduction method
        with pytest.raises(ValueError, match="Invalid reduction_method"):
            TrajectoryConfig(reduction_method='invalid')
            
        # Invalid dimensions
        with pytest.raises(ValueError, match="dimensions must be 2 or 3"):
            TrajectoryConfig(dimensions=4)
            
        # Invalid backend
        with pytest.raises(ValueError, match="Invalid backend"):
            TrajectoryConfig(backend='invalid')
            
        # Invalid color_by
        with pytest.raises(ValueError, match="Invalid color_by"):
            TrajectoryConfig(color_by='invalid')
            
        # Invalid layer_separation
        with pytest.raises(ValueError, match="Invalid layer_separation"):
            TrajectoryConfig(layer_separation='invalid')
    
    def test_cache_functionality(self, trajectory_visualizer, small_data):
        """Test dimensionality reduction caching."""
        trajectory_visualizer.config.cache_reduction = True
        trajectory_visualizer.config.reduction_method = 'pca'
        
        # First call should compute
        activations = small_data['activations']
        reduced1 = trajectory_visualizer._reduce_dimensions(activations)
        
        # Second call should use cache
        with patch.object(trajectory_visualizer, '_reducer', None):
            reduced2 = trajectory_visualizer._reduce_dimensions(activations)
            
        np.testing.assert_array_equal(reduced1, reduced2)
        
        # Disable cache
        trajectory_visualizer.config.cache_reduction = False
        trajectory_visualizer._reduction_cache.clear()
        
        # Should recompute
        reduced3 = trajectory_visualizer._reduce_dimensions(activations)
        assert reduced3.shape == reduced1.shape


@pytest.mark.integration
class TestTrajectoryIntegration:
    """Integration tests for TrajectoryVisualizer."""
    
    def test_full_pipeline_plotly(self):
        """Test full pipeline with Plotly backend."""
        # Create realistic data
        np.random.seed(42)
        n_samples = 100
        n_layers = 12
        n_features = 768
        
        data = {
            'activations': np.random.randn(n_samples, n_layers, n_features),
            'labels': {
                'class_labels': np.random.randint(0, 10, n_samples),
                'cluster_labels': np.random.randint(0, 5, (n_samples, n_layers))
            },
            'metadata': {'dataset': 'integration_test'},
            'windows': {
                'early': list(range(4)),
                'middle': list(range(4, 8)),
                'late': list(range(8, 12))
            }
        }
        
        # Create visualizer
        config = TrajectoryConfig(
            reduction_method='pca',  # Fast for tests
            dimensions=3,
            backend='plotly',
            color_by='class',
            show_arrows=True,
            layer_separation='stepped',
            max_samples=50
        )
        visualizer = TrajectoryVisualizer(config)
        
        # Create figures for all windows
        for window in ['early', 'middle', 'late']:
            fig = visualizer.create_figure(data, window=window)
            assert fig is not None
            
    @pytest.mark.skipif(True, reason="Matplotlib 3D can be slow")
    def test_full_pipeline_matplotlib(self):
        """Test full pipeline with Matplotlib backend."""
        # Similar to above but with matplotlib
        pass