"""
Test the path fragmentation visualization components.
"""

import os
import json
import unittest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock

from visualization.path_fragmentation_tab import (
    create_path_fragmentation_histogram,
    create_path_table,
    create_path_detail_view,
    create_path_visualization
)

# Create a mock paths data structure for testing
def create_mock_paths_data():
    """Create a mock paths data structure for testing."""
    return {
        "layers": ["layer1", "layer2", "layer3"],
        "path_archetypes": [
            {
                "path": "0→1→2",
                "count": 50,
                "survived_rate": 0.8,
                "demo_stats": {
                    "sex": {"male": 0.3, "female": 0.7},
                    "pclass": {"1": 0.6, "2": 0.3, "3": 0.1},
                    "age": {"mean": 28.5, "median": 27.0, "min": 5.0, "max": 65.0}
                }
            },
            {
                "path": "1→2→0",
                "count": 30,
                "survived_rate": 0.2,
                "demo_stats": {
                    "sex": {"male": 0.8, "female": 0.2},
                    "pclass": {"1": 0.2, "2": 0.3, "3": 0.5},
                    "age": {"mean": 45.2, "median": 42.0, "min": 18.0, "max": 75.0}
                }
            }
        ],
        "similarity": {
            "fragmentation_scores": {
                "mean": 0.6,
                "median": 0.55,
                "high_threshold": 0.7,
                "low_threshold": 0.3,
                "scores": [0.8, 0.2],
                "high_fragmentation_paths": [0],
                "low_fragmentation_paths": [1]
            },
            "convergent_paths": {
                "0": [
                    {"early_layer": 0, "late_layer": 2, "early_cluster": 0, "late_cluster": 2, "similarity": 0.75}
                ],
                "1": []
            }
        }
    }

class TestPathFragmentationVisualization(unittest.TestCase):
    """Test the path fragmentation visualization components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.paths_data = create_mock_paths_data()
        self.mock_get_friendly_layer_name = lambda x: f"Friendly {x}"
        
    def test_create_path_fragmentation_histogram(self):
        """Test creating a path fragmentation histogram."""
        # Test 'all' filter
        fig = create_path_fragmentation_histogram(self.paths_data, "all")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)  # Should have one trace (histogram)
        
        # Test 'high' filter
        fig = create_path_fragmentation_histogram(self.paths_data, "high")
        self.assertIsInstance(fig, go.Figure)
        
        # Test 'low' filter
        fig = create_path_fragmentation_histogram(self.paths_data, "low") 
        self.assertIsInstance(fig, go.Figure)
        
        # Test with empty data
        fig = create_path_fragmentation_histogram({}, "all")
        self.assertIsInstance(fig, go.Figure)
        
    def test_create_path_table(self):
        """Test creating a path table."""
        from dash import html, dash_table
        
        # Test with 'all' filter
        table_div = create_path_table(
            self.paths_data, 
            "all", 
            "frag_desc", 
            self.mock_get_friendly_layer_name
        )
        self.assertIsInstance(table_div, html.Div)
        
        # Find the DataTable component in the returned Div
        data_table = None
        for child in table_div.children:
            if isinstance(child, dash_table.DataTable):
                data_table = child
                break
        
        self.assertIsNotNone(data_table)
        self.assertEqual(len(data_table.data), 2)  # Should have 2 rows
        
        # Test with 'high' filter
        table_div = create_path_table(
            self.paths_data, 
            "high", 
            "frag_desc", 
            self.mock_get_friendly_layer_name
        )
        self.assertIsInstance(table_div, html.Div)
        
        # Test with empty data
        table_div = create_path_table({}, "all", "frag_desc", self.mock_get_friendly_layer_name)
        self.assertIsInstance(table_div, html.Div)
        
    def test_create_path_detail_view(self):
        """Test creating a path detail view."""
        from dash import html
        
        # Create a mock selected row
        selected_row = {
            "Path ID": 0,
            "Path": "Friendly layer1 C0 → Friendly layer2 C1 → Friendly layer3 C2",
            "Raw Path": "0→1→2",
            "Fragmentation": 0.8,
            "Count": 50
        }
        
        # Test with valid data
        detail_view = create_path_detail_view(
            selected_row,
            self.paths_data,
            self.mock_get_friendly_layer_name
        )
        self.assertIsInstance(detail_view, html.Div)
        
        # Test with empty data
        detail_view = create_path_detail_view(selected_row, {}, self.mock_get_friendly_layer_name)
        self.assertIsInstance(detail_view, html.Div)
        
        # Test with no row selected
        detail_view = create_path_detail_view(None, self.paths_data, self.mock_get_friendly_layer_name)
        self.assertIsInstance(detail_view, html.Div)
        
    def test_create_path_visualization(self):
        """Test creating a path visualization."""
        # Create a mock selected row
        selected_row = {
            "Path ID": 0,
            "Path": "Friendly layer1 C0 → Friendly layer2 C1 → Friendly layer3 C2",
            "Raw Path": "0→1→2",
            "Fragmentation": 0.8,
            "Count": 50
        }
        
        # Test with valid data
        fig = create_path_visualization(
            selected_row,
            self.paths_data,
            self.mock_get_friendly_layer_name
        )
        self.assertIsInstance(fig, go.Figure)
        
        # Test with empty data
        fig = create_path_visualization(selected_row, {}, self.mock_get_friendly_layer_name)
        self.assertIsInstance(fig, go.Figure)
        
        # Test with no row selected
        fig = create_path_visualization(None, self.paths_data, self.mock_get_friendly_layer_name)
        self.assertIsInstance(fig, go.Figure)

if __name__ == "__main__":
    unittest.main()