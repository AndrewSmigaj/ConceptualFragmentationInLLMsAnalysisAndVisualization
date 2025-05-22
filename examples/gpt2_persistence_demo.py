"""
Demo script for GPT-2 analysis persistence functionality.

This script demonstrates how to use the GPT-2 analysis persistence system
to save, load, and manage analysis results.
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path to import our modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from concept_fragmentation.persistence import GPT2AnalysisPersistence, save_gpt2_analysis, load_gpt2_analysis


def create_sample_analysis_data():
    """Create sample GPT-2 analysis data for demonstration."""
    return {
        "model_type": "gpt2-small",
        "layers": ["layer_0", "layer_1", "layer_2", "layer_3"],
        "token_metadata": {
            "tokens": ["Hello", ",", " world", "!", " How", " are", " you", "?"],
            "positions": [0, 1, 2, 3, 4, 5, 6, 7],
            "token_ids": [15496, 11, 995, 0, 1374, 389, 345, 30]
        },
        "token_paths": {
            "0": {
                "token_text": "Hello",
                "position": 0,
                "cluster_path": [0, 1, 2, 1],
                "path_length": 3.2,
                "cluster_changes": 3,
                "mobility_score": 0.8
            },
            "1": {
                "token_text": ",",
                "position": 1,
                "cluster_path": [1, 1, 1, 1],
                "path_length": 0.0,
                "cluster_changes": 0,
                "mobility_score": 0.0
            },
            "2": {
                "token_text": " world",
                "position": 2,
                "cluster_path": [2, 2, 3, 2],
                "path_length": 2.1,
                "cluster_changes": 2,
                "mobility_score": 0.5
            }
        },
        "cluster_labels": {
            "layer_0": [0, 1, 2, 1, 0, 1, 2, 1],
            "layer_1": [1, 1, 2, 1, 1, 1, 2, 1],
            "layer_2": [2, 1, 3, 1, 2, 1, 3, 1],
            "layer_3": [1, 1, 2, 1, 1, 1, 2, 1]
        },
        "attention_data": {
            "layer_0": {
                "entropy": 2.3,
                "head_agreement": 0.7,
                "num_heads": 12
            },
            "layer_1": {
                "entropy": 2.1,
                "head_agreement": 0.8,
                "num_heads": 12
            }
        },
        "cluster_metrics": {
            "layer_0": {
                "purity": 0.85,
                "silhouette": 0.6,
                "num_clusters": 3
            },
            "layer_1": {
                "purity": 0.90,
                "silhouette": 0.7,
                "num_clusters": 3
            }
        },
        "similarity": {
            "fragmentation_scores": {
                "scores": [0.8, 0.0, 0.5, 0.2, 0.7, 0.1, 0.6, 0.3],
                "tokens": ["Hello", ",", " world", "!", " How", " are", " you", "?"]
            }
        }
    }


def demo_basic_persistence():
    """Demonstrate basic save and load functionality."""
    print("=== Basic Persistence Demo ===")
    
    # Create persistence manager
    persistence_manager = GPT2AnalysisPersistence(
        base_dir="demo_gpt2_persistence",
        enable_cache=True,
        cache_ttl=3600,
        max_versions=5
    )
    
    # Create sample data
    analysis_data = create_sample_analysis_data()
    model_name = "gpt2-small"
    input_text = "Hello, world! How are you?"
    
    # Save analysis
    print("Saving analysis...")
    analysis_id = persistence_manager.save_analysis_results(
        analysis_data=analysis_data,
        model_name=model_name,
        input_text=input_text,
        metadata={"demo": True, "description": "Basic persistence demo"}
    )
    print(f"Analysis saved with ID: {analysis_id}")
    
    # Load analysis
    print("Loading analysis...")
    loaded_data = persistence_manager.load_analysis_results(analysis_id)
    
    if loaded_data:
        print("Analysis loaded successfully!")
        print(f"Model: {loaded_data['metadata']['model_name']}")
        print(f"Input text: {loaded_data['metadata']['input_text']}")
        print(f"Number of tokens: {loaded_data['model_info']['num_tokens']}")
        print(f"Number of layers: {loaded_data['model_info']['num_layers']}")
    else:
        print("Failed to load analysis!")
    
    return persistence_manager, analysis_id


def demo_visualization_states(persistence_manager, analysis_id):
    """Demonstrate saving and loading visualization states."""
    print("\\n=== Visualization States Demo ===")
    
    # Create sample visualization configs
    token_sankey_config = {
        "visualization_type": "token_sankey",
        "selected_tokens": [0, 2, 4],
        "selected_layers": ["layer_0", "layer_1", "layer_2"],
        "highlight_paths": True,
        "color_scheme": "qualitative"
    }
    
    attention_flow_config = {
        "visualization_type": "attention_flow",
        "selected_layers": ["layer_0", "layer_1"],
        "head_selection": [0, 1, 2, 3],
        "threshold": 0.1,
        "layout": "hierarchical"
    }
    
    # Save visualization states
    print("Saving visualization states...")
    token_state_id = persistence_manager.save_visualization_state(
        analysis_id=analysis_id,
        visualization_config=token_sankey_config,
        visualization_type="token_sankey",
        state_name="demo_token_view"
    )
    
    attention_state_id = persistence_manager.save_visualization_state(
        analysis_id=analysis_id,
        visualization_config=attention_flow_config,
        visualization_type="attention_flow",
        state_name="demo_attention_view"
    )
    
    print(f"Token Sankey state saved: {token_state_id}")
    print(f"Attention Flow state saved: {attention_state_id}")
    
    # Load visualization states
    print("Loading visualization states...")
    loaded_token_state = persistence_manager.load_visualization_state(token_state_id)
    loaded_attention_state = persistence_manager.load_visualization_state(attention_state_id)
    
    if loaded_token_state:
        print(f"Token state loaded: {loaded_token_state['visualization_type']}")
        print(f"Selected tokens: {loaded_token_state['config']['selected_tokens']}")
    
    if loaded_attention_state:
        print(f"Attention state loaded: {loaded_attention_state['visualization_type']}")
        print(f"Selected layers: {loaded_attention_state['config']['selected_layers']}")


def demo_sessions(persistence_manager, analysis_id):
    """Demonstrate session management."""
    print("\\n=== Session Management Demo ===")
    
    # Create a session
    print("Creating analysis session...")
    session_id = persistence_manager.create_session(
        session_name="GPT-2 Demo Session",
        analysis_ids=[analysis_id]
    )
    print(f"Session created: {session_id}")
    
    # Load session
    print("Loading session...")
    loaded_session = persistence_manager.load_session(session_id)
    
    if loaded_session:
        print(f"Session loaded: {loaded_session['session_name']}")
        print(f"Created at: {loaded_session['created_at']}")
        print(f"Analysis IDs: {loaded_session['analysis_ids']}")
    else:
        print("Failed to load session!")


def demo_export_import(persistence_manager, analysis_id):
    """Demonstrate export and import functionality."""
    print("\\n=== Export/Import Demo ===")
    
    # Export in different formats
    formats = ["json", "pickle", "csv"]
    
    for format_type in formats:
        print(f"Exporting analysis in {format_type.upper()} format...")
        try:
            export_path = persistence_manager.export_analysis(
                analysis_id=analysis_id,
                export_format=format_type,
                include_visualizations=True
            )
            print(f"Exported to: {export_path}")
        except Exception as e:
            print(f"Export failed: {e}")


def demo_list_and_cleanup(persistence_manager):
    \"\"\"Demonstrate listing analyses and cleanup functionality.\"\"\"
    print("\\n=== List and Cleanup Demo ===")
    
    # List all analyses
    print("Listing all analyses...")
    analyses = persistence_manager.list_analyses()
    
    for analysis in analyses:
        print(f"- {analysis['analysis_id']}: {analysis['model_name']} ({analysis['timestamp'][:19]})")
    
    # List analyses for specific model
    print("\\nListing analyses for gpt2-small...")
    gpt2_analyses = persistence_manager.list_analyses(model_name="gpt2-small")
    
    for analysis in gpt2_analyses:
        print(f"- {analysis['analysis_id']}: {analysis.get('user_name', 'Unnamed')} ({analysis['timestamp'][:19]})")
    
    # Cache statistics
    if persistence_manager.cache:
        stats = persistence_manager.cache.get_stats()
        print(f"\\nCache statistics:")
        print(f"- Size: {stats['size']} items")
        print(f"- Provider: {stats['provider']}")
        print(f"- Memory only: {stats['memory_only']}")
    
    # Cleanup cache
    print("\\nCleaning up cache...")
    persistence_manager.cleanup_cache(max_age_hours=0)  # Clean all for demo
    print("Cache cleaned up!")


def demo_convenience_functions():
    \"\"\"Demonstrate convenience functions.\"\"\"
    print("\\n=== Convenience Functions Demo ===")
    
    # Create sample data
    analysis_data = create_sample_analysis_data()
    model_name = "gpt2-medium"
    input_text = "This is a test with convenience functions."
    
    # Save using convenience function
    print("Saving analysis using convenience function...")
    analysis_id = save_gpt2_analysis(
        analysis_data=analysis_data,
        model_name=model_name,
        input_text=input_text
    )
    print(f"Analysis saved: {analysis_id}")
    
    # Load using convenience function
    print("Loading analysis using convenience function...")
    loaded_data = load_gpt2_analysis(analysis_id)
    
    if loaded_data:
        print("Analysis loaded successfully!")
        print(f"Model: {loaded_data['metadata']['model_name']}")
        print(f"Number of layers: {loaded_data['model_info']['num_layers']}")
    else:
        print("Failed to load analysis!")


def main():
    \"\"\"Run all demos.\"\"\"
    print("GPT-2 Analysis Persistence Demo")
    print("=" * 40)
    
    try:
        # Basic persistence
        persistence_manager, analysis_id = demo_basic_persistence()
        
        # Visualization states
        demo_visualization_states(persistence_manager, analysis_id)
        
        # Sessions
        demo_sessions(persistence_manager, analysis_id)
        
        # Export/Import
        demo_export_import(persistence_manager, analysis_id)
        
        # List and cleanup
        demo_list_and_cleanup(persistence_manager)
        
        # Convenience functions
        demo_convenience_functions()
        
        print("\\n=== Demo completed successfully! ===")
        print("Check the 'demo_gpt2_persistence' directory for saved files.")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()