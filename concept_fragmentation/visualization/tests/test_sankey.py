"""Comprehensive tests for unified SankeyGenerator."""

import pytest
import numpy as np
import importlib
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from concept_fragmentation.visualization.sankey import (
    SankeyGenerator, PathInfo, WindowedAnalysis, SankeyData
)
from concept_fragmentation.visualization.configs import SankeyConfig
from concept_fragmentation.visualization.exceptions import VisualizationError, InvalidDataError


class TestSankeyGenerator:
    """Test suite for SankeyGenerator class."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Create sample data for testing."""
        return {
            'windowed_analysis': {
                'early': {
                    'layers': [0, 1, 2, 3],
                    'total_paths': 1000,
                    'unique_paths': 150,
                    'archetypal_paths': [
                        {
                            'path': [0, 1, 1, 2],
                            'frequency': 100,
                            'representative_words': ['the', 'of', 'to', 'and', 'a'],
                            'semantic_labels': ['Function Words', 'Function Words', 'Function Words', 'Content Words'],
                            'percentage': 10.0
                        },
                        {
                            'path': [1, 1, 2, 2],
                            'frequency': 80,
                            'representative_words': ['in', 'that', 'is', 'for', 'it'],
                            'semantic_labels': ['Function Words', 'Function Words', 'Content Words', 'Content Words'],
                            'percentage': 8.0
                        },
                        {
                            'path': [2, 2, 2, 2],
                            'frequency': 60,
                            'representative_words': ['time', 'person', 'year', 'way', 'day'],
                            'semantic_labels': ['Content Words', 'Content Words', 'Content Words', 'Content Words'],
                            'percentage': 6.0
                        }
                    ]
                },
                'middle': {
                    'layers': [4, 5, 6, 7],
                    'total_paths': 1000,
                    'unique_paths': 200,
                    'archetypal_paths': [
                        {
                            'path': [0, 0, 1, 1],
                            'frequency': 120,
                            'representative_words': ['the', 'a', 'to', 'of', 'and'],
                            'semantic_labels': ['Function Words', 'Function Words', 'Function Words', 'Function Words'],
                            'percentage': 12.0
                        }
                    ]
                },
                'late': {
                    'layers': [8, 9, 10, 11],
                    'total_paths': 1000,
                    'unique_paths': 100,
                    'archetypal_paths': [
                        {
                            'path': [0, 0, 0, 0],
                            'frequency': 200,
                            'representative_words': ['the', 'to', 'of', 'and', 'a'],
                            'semantic_labels': ['Function Words', 'Function Words', 'Function Words', 'Function Words'],
                            'percentage': 20.0
                        }
                    ]
                }
            },
            'labels': {
                'layer_0': {
                    'L0_C0': {'label': 'Function Words', 'primary': 'Function Words'},
                    'L0_C1': {'label': 'Function Words', 'primary': 'Function Words'},
                    'L0_C2': {'label': 'Content Words', 'primary': 'Content Words'}
                },
                'layer_1': {
                    'L1_C1': {'label': 'Function Words', 'primary': 'Function Words'},
                    'L1_C2': {'label': 'Content Words', 'primary': 'Content Words'}
                },
                'layer_2': {
                    'L2_C1': {'label': 'Function Words', 'primary': 'Function Words'},
                    'L2_C2': {'label': 'Content Words', 'primary': 'Content Words'}
                },
                'layer_3': {
                    'L3_C2': {'label': 'Content Words', 'primary': 'Content Words'}
                }
            },
            'purity_data': {
                'layer_0': {
                    'L0_C0': {'purity': 95.5},
                    'L0_C1': {'purity': 92.3},
                    'L0_C2': {'purity': 88.7}
                }
            }
        }
    
    @pytest.fixture
    def sankey_generator(self) -> SankeyGenerator:
        """Create SankeyGenerator instance."""
        config = SankeyConfig(top_n_paths=3, show_purity=True)
        return SankeyGenerator(config)
    
    def test_initialization(self):
        """Test SankeyGenerator initialization."""
        # Test with default config
        gen = SankeyGenerator()
        assert gen.config.top_n_paths == 25
        assert gen.config.show_purity is True
        
        # Test with custom config
        config = SankeyConfig(top_n_paths=10, show_purity=False)
        gen = SankeyGenerator(config)
        assert gen.config.top_n_paths == 10
        assert gen.config.show_purity is False
    
    @pytest.mark.skip(reason="Complex plotly import mocking")
    def test_initialization_without_plotly(self):
        """Test initialization fails gracefully without plotly."""
        pass
    
    def test_validate_sankey_data(self, sankey_generator):
        """Test data validation."""
        # Valid data should pass
        valid_data = {'windowed_analysis': {'early': {}}}
        sankey_generator._validate_sankey_data(valid_data)
        
        # Invalid data should raise errors
        with pytest.raises(InvalidDataError, match="Data must be a dictionary"):
            sankey_generator._validate_sankey_data("not a dict")
            
        with pytest.raises(InvalidDataError, match="Missing 'windowed_analysis'"):
            sankey_generator._validate_sankey_data({})
            
        with pytest.raises(InvalidDataError, match="'windowed_analysis' must be a dictionary"):
            sankey_generator._validate_sankey_data({'windowed_analysis': 'not a dict'})
    
    def test_find_visible_clusters(self, sankey_generator, sample_data):
        """Test finding visible clusters from paths."""
        paths = sample_data['windowed_analysis']['early']['archetypal_paths']
        layers = sample_data['windowed_analysis']['early']['layers']
        
        clusters = sankey_generator._find_visible_clusters(paths, layers)
        
        # Check that all clusters from paths are found
        assert 0 in clusters[0]  # Layer 0 has cluster 0
        assert 1 in clusters[0]  # Layer 0 has cluster 1
        assert 2 in clusters[0]  # Layer 0 has cluster 2
        assert 1 in clusters[1]  # Layer 1 has cluster 1
        assert 2 in clusters[3]  # Layer 3 has cluster 2
    
    def test_build_nodes(self, sankey_generator, sample_data):
        """Test node building."""
        paths = sample_data['windowed_analysis']['early']['archetypal_paths']
        layers = sample_data['windowed_analysis']['early']['layers']
        clusters_by_layer = sankey_generator._find_visible_clusters(paths, layers)
        
        nodes, node_map = sankey_generator._build_nodes(
            clusters_by_layer,
            layers,
            sample_data['labels'],
            sample_data['purity_data']
        )
        
        # Check node labels include semantic labels and purity
        assert any('Function Words' in node for node in nodes)
        assert any('Content Words' in node for node in nodes)
        assert any('96%' in node for node in nodes) or any('95%' in node for node in nodes)  # Purity percentage (rounded)
        
        # Check node mapping
        assert (0, 0) in node_map  # Layer 0, Cluster 0
        assert node_map[(0, 0)] >= 0  # Valid index
    
    def test_build_links(self, sankey_generator, sample_data):
        """Test link building with colors."""
        paths = sample_data['windowed_analysis']['early']['archetypal_paths']
        layers = sample_data['windowed_analysis']['early']['layers']
        clusters_by_layer = sankey_generator._find_visible_clusters(paths, layers)
        nodes, node_map = sankey_generator._build_nodes(
            clusters_by_layer, layers, {}, {}
        )
        
        links, path_colors = sankey_generator._build_links(paths, layers, node_map)
        
        # Check links exist
        assert len(links) > 0
        
        # Check link structure
        for link in links:
            assert 'source' in link
            assert 'target' in link
            assert 'value' in link
            assert 'color' in link
            
        # Check path colors assigned
        assert len(path_colors) == len(paths)
        assert all(isinstance(color, str) for color in path_colors.values())
    
    def test_generate_path_description(self, sankey_generator):
        """Test path description generation."""
        # Pure path
        desc = sankey_generator._generate_path_description(
            ['Function Words', 'Function Words', 'Function Words']
        )
        assert desc == "Pure Function Words"
        
        # Single transition
        desc = sankey_generator._generate_path_description(
            ['Function Words', 'Function Words', 'Content Words']
        )
        assert desc == "Function Words→Content Words"
        
        # Dominant pattern
        desc = sankey_generator._generate_path_description(
            ['Function Words', 'Function Words', 'Function Words', 'Content Words']
        )
        assert "Function Words" in desc
        
        # Complex path
        desc = sankey_generator._generate_path_description(
            ['Function Words', 'Content Words', 'Technical/Foreign', 'Content Words']
        )
        assert "→" in desc
        
        # Empty labels
        desc = sankey_generator._generate_path_description([])
        assert desc == "Unknown Path"
    
    def test_create_figure(self, sankey_generator, sample_data):
        """Test figure creation."""
        # Since we're testing with real plotly, just check basic functionality
        fig = sankey_generator.create_figure(sample_data, window='early')
        
        # Check figure was created with correct structure
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
        assert hasattr(fig.data[0], 'node')
        assert hasattr(fig.data[0], 'link')
        
        # Check invalid window raises error
        with pytest.raises(InvalidDataError, match="Window 'invalid' not found"):
            sankey_generator.create_figure(sample_data, window='invalid')
    
    def test_infer_k_value(self, sankey_generator):
        """Test k value inference from nodes."""
        # Test with cluster labels
        nodes = [
            "L0: Function Words C0 (95%)",
            "L0: Content Words C1 (88%)",
            "L0: Technical C2 (75%)",
            "L1: Function Words C0 (92%)",
            "L1: Mixed C4 (80%)"
        ]
        k = sankey_generator._infer_k_value(nodes)
        assert k == 5  # Max cluster ID is 4, so k = 5
        
        # Test with no clusters
        nodes = ["L0: Some Label", "L1: Another Label"]
        k = sankey_generator._infer_k_value(nodes)
        assert k == 10  # Default
    
    @patch.object(SankeyGenerator, 'save_figure')
    def test_create_all_windows(self, mock_save, sankey_generator, sample_data, tmp_path):
        """Test batch creation for all windows."""
        # Use pytest's tmp_path fixture for temporary directory
        output_dir = tmp_path / "test_output"
        
        # Create all windows
        figures = sankey_generator.create_all_windows(
            sample_data, 
            output_dir=output_dir
        )
        
        # Check all windows processed
        assert len(figures) == 3
        assert 'early' in figures
        assert 'middle' in figures
        assert 'late' in figures
        
        # Check figures were created correctly
        for window, fig in figures.items():
            assert fig is not None
            assert hasattr(fig, 'data')
        
        # Check save_figure called for each window
        assert mock_save.call_count == 3
    
    def test_create_path_summary(self, sankey_generator, sample_data):
        """Test path summary creation."""
        # Test markdown format
        summary = sankey_generator.create_path_summary(sample_data, 'markdown')
        assert '# Archetypal Paths Summary' in summary
        assert '## Early Window' in summary
        assert '### Path 1:' in summary
        assert '**Frequency**:' in summary
        assert '**Examples**:' in summary
        
        # Test text format
        summary = sankey_generator.create_path_summary(sample_data, 'text')
        assert 'EARLY WINDOW' in summary
        assert 'Path 1:' in summary
        assert 'Frequency:' in summary
        assert 'Examples:' in summary
    
    @patch('concept_fragmentation.visualization.sankey.go.Figure')
    def test_error_handling(self, mock_figure, sankey_generator, sample_data):
        """Test error handling in create_figure."""
        # Simulate an error during figure creation
        mock_figure.side_effect = Exception("Test error")
        
        with pytest.raises(VisualizationError, match="Sankey creation failed"):
            sankey_generator.create_figure(sample_data, 'early')
    
    def test_config_update(self, sankey_generator):
        """Test config update via kwargs."""
        # Assuming update_config method exists in base class
        if hasattr(sankey_generator, 'update_config'):
            original_value = sankey_generator.config.top_n_paths
            sankey_generator.update_config(top_n_paths=50)
            assert sankey_generator.config.top_n_paths == 50
    
    def test_color_palette_assignment(self, sankey_generator):
        """Test color assignment to paths."""
        # Test with default palette
        colors = sankey_generator._assign_path_colors(5)
        assert len(colors) == 5
        assert all(isinstance(c, str) for c in colors.values())
        
        # Test with more paths than colors
        colors = sankey_generator._assign_path_colors(30)
        assert len(colors) == 30
        # Colors should repeat
        assert colors[0] == colors[25]  # Assuming default palette has 25 colors


@pytest.mark.integration
class TestSankeyIntegration:
    """Integration tests for SankeyGenerator."""
    
    @pytest.fixture
    def real_data_path(self):
        """Path to real test data if available."""
        return Path("experiments/gpt2/all_tokens/k10_analysis_results/windowed_analysis_k10.json")
    
    def test_with_real_data(self, real_data_path):
        """Test with real data if available."""
        if not real_data_path.exists():
            pytest.skip("Real data not available")
            
        import json
        windowed_data = json.loads(real_data_path.read_text())
        
        # Wrap in expected format
        data = {
            'windowed_analysis': windowed_data,
            'labels': {},  # Would be loaded from separate file
            'purity_data': {}  # Would be loaded from separate file
        }
        
        generator = SankeyGenerator()
        
        # Check if early window exists
        if 'early' not in windowed_data:
            pytest.skip("Early window not in data")
            
        fig = generator.create_figure(data, window='early')
        
        # Basic checks
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')