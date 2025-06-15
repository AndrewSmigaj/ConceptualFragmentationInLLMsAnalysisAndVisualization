#!/usr/bin/env python3
"""Example usage of the unified SankeyGenerator.

This example shows how to use the new unified SankeyGenerator to create
Sankey diagrams for concept trajectory visualization.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from concept_fragmentation.visualization.sankey import SankeyGenerator
from concept_fragmentation.visualization.configs import SankeyConfig


def basic_example():
    """Basic example with default configuration."""
    print("=== Basic Example ===")
    
    # Create generator with default config
    generator = SankeyGenerator()
    
    # Load sample data (you would load your actual data here)
    sample_data = {
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
                        'semantic_labels': ['Function Words', 'Function Words', 
                                          'Function Words', 'Content Words']
                    },
                    {
                        'path': [2, 2, 2, 2],
                        'frequency': 80,
                        'representative_words': ['time', 'person', 'year', 'way', 'day'],
                        'semantic_labels': ['Content Words', 'Content Words',
                                          'Content Words', 'Content Words']
                    }
                ]
            }
        },
        'labels': {},  # Optional semantic labels
        'purity_data': {}  # Optional purity scores
    }
    
    # Create figure for early window
    fig = generator.create_figure(sample_data, window='early')
    
    # Save as HTML
    generator.save_figure(fig, 'sankey_basic.html')
    print("Saved: sankey_basic.html")


def custom_config_example():
    """Example with custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom config
    config = SankeyConfig(
        top_n_paths=10,  # Show top 10 paths instead of default 25
        show_purity=False,  # Don't show purity percentages
        colored_paths=True,  # Use colored paths
        legend_position='right',  # Legend on the right
        last_layer_labels_position='right',  # Last layer labels on right
        width=1200,  # Custom width
        height=600,  # Custom height
        generate_summary=True  # Generate path summary file
    )
    
    # Create generator with custom config
    generator = SankeyGenerator(config)
    
    # ... (load your data here) ...
    
    print("Generator created with custom configuration")


def batch_generation_example():
    """Example of generating diagrams for all windows at once."""
    print("\n=== Batch Generation Example ===")
    
    # Create generator
    generator = SankeyGenerator(SankeyConfig(top_n_paths=15))
    
    # Load data with all windows
    data = {
        'windowed_analysis': {
            'early': {
                'layers': [0, 1, 2, 3],
                'total_paths': 1000,
                'unique_paths': 150,
                'archetypal_paths': [
                    {
                        'path': [0, 0, 1, 1],
                        'frequency': 120,
                        'representative_words': ['the', 'a', 'an', 'to', 'of']
                    }
                ]
            },
            'middle': {
                'layers': [4, 5, 6, 7],
                'total_paths': 1000,
                'unique_paths': 100,
                'archetypal_paths': [
                    {
                        'path': [0, 0, 0, 0],
                        'frequency': 200,
                        'representative_words': ['the', 'to', 'of', 'and', 'a']
                    }
                ]
            },
            'late': {
                'layers': [8, 9, 10, 11],
                'total_paths': 1000,
                'unique_paths': 50,
                'archetypal_paths': [
                    {
                        'path': [0, 0, 0, 0],
                        'frequency': 300,
                        'representative_words': ['the', 'to', 'of', 'and', 'a']
                    }
                ]
            }
        }
    }
    
    # Generate all windows at once
    output_dir = Path('sankey_output')
    figures = generator.create_all_windows(data, output_dir)
    
    print(f"Generated {len(figures)} diagrams in {output_dir}/")
    for window, fig in figures.items():
        print(f"  - {window}: {fig}")


def load_from_files_example():
    """Example of loading data from actual analysis files."""
    print("\n=== Load from Files Example ===")
    
    # Paths to your data files
    windowed_analysis_path = Path("path/to/windowed_analysis_k10.json")
    labels_path = Path("path/to/cluster_labels_k10.json")
    purity_path = Path("path/to/semantic_purity_k10.json")
    
    # Check if files exist
    if not windowed_analysis_path.exists():
        print("Note: This example requires actual data files")
        print(f"Expected file: {windowed_analysis_path}")
        return
    
    # Load data
    windowed_analysis = json.loads(windowed_analysis_path.read_text())
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    purity_data = json.loads(purity_path.read_text()) if purity_path.exists() else {}
    
    # Combine into expected format
    data = {
        'windowed_analysis': windowed_analysis,
        'labels': labels,
        'purity_data': purity_data
    }
    
    # Create generator
    generator = SankeyGenerator()
    
    # Generate for a specific window
    fig = generator.create_figure(data, window='early')
    generator.save_figure(fig, 'sankey_from_files.html')
    
    print("Generated sankey from data files")


def main():
    """Run all examples."""
    print("Unified SankeyGenerator Examples\n")
    
    # Run examples
    basic_example()
    custom_config_example()
    batch_generation_example()
    load_from_files_example()
    
    print("\nExamples complete!")
    print("\nKey points:")
    print("1. Use SankeyConfig to customize appearance and behavior")
    print("2. Data must include 'windowed_analysis' key")
    print("3. 'labels' and 'purity_data' are optional")
    print("4. Use create_all_windows() for batch generation")
    print("5. Path summaries can be generated automatically")


if __name__ == '__main__':
    main()