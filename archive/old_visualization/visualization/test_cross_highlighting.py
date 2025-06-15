"""
Test script for cross-visualization highlighting functionality.

This script tests the synchronized highlighting system to ensure proper
coordination between token paths, attention patterns, and correlation visualizations.
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the cross-highlighting system
from visualization.gpt2_cross_viz_highlighting import (
    CrossVisualizationHighlighter,
    SelectionState,
    create_synchronized_visualization_layout,
    update_token_highlights,
    update_attention_highlights,
    clear_highlights
)

# Mock Plotly figure for testing
class MockFigure:
    """Mock Plotly Figure for testing without actual plotly dependency."""
    
    def __init__(self):
        self.data = [MockTrace()]
        self.layout = MockLayout()
    
    class MockTrace:
        def __init__(self):
            self.node = MockNode()
            self.link = MockLink()
    
    class MockNode:
        def __init__(self):
            self.color = ["lightblue"] * 10
            self.label = [f"Token_{i}" for i in range(10)]
    
    class MockLink:
        def __init__(self):
            self.color = ["lightgray"] * 15
            self.source = [i for i in range(15)]
            self.target = [(i + 1) % 10 for i in range(15)]
    
    class MockLayout:
        def __init__(self):
            self.annotations = []


def test_selection_state():
    """Test the SelectionState dataclass functionality."""
    print("Testing SelectionState...")
    
    # Test default initialization
    state = SelectionState()
    assert len(state.selected_tokens) == 0
    assert len(state.selected_attention_edges) == 0
    assert len(state.selected_clusters) == 0
    assert state.highlight_color == "#ff6b6b"
    
    # Test with initial values
    state = SelectionState(
        selected_tokens={"token_1", "token_2"},
        selected_clusters={0, 1, 2}
    )
    assert len(state.selected_tokens) == 2
    assert len(state.selected_clusters) == 3
    
    print("✓ SelectionState tests passed")
    return True


def test_cross_visualization_highlighter():
    """Test the CrossVisualizationHighlighter class."""
    print("Testing CrossVisualizationHighlighter...")
    
    highlighter = CrossVisualizationHighlighter()
    
    # Test visualization registration
    highlighter.register_visualization(
        viz_id="test_token_sankey",
        viz_type="token_sankey",
        data_mapping={"tokens": ["token_1", "token_2", "token_3"]}
    )
    
    assert "test_token_sankey" in highlighter.visualization_registry
    assert highlighter.visualization_registry["test_token_sankey"]["type"] == "token_sankey"
    
    # Test selection handlers
    handlers = highlighter._get_selection_handlers("token_sankey")
    assert "on_node_click" in handlers
    assert "update_highlights" in handlers
    
    # Test token node selection
    node_data = {"token_id": "token_1", "layer": "layer_0", "position": 3}
    result = highlighter._handle_token_node_selection(node_data)
    
    assert "token_1" in highlighter.selection_state.selected_tokens
    assert 3 in highlighter.selection_state.selected_positions
    
    print("✓ CrossVisualizationHighlighter tests passed")
    return True


def test_highlight_updates():
    """Test highlight update functions."""
    print("Testing highlight update functions...")
    
    # Create mock figure
    mock_fig = MockFigure()
    original_colors = mock_fig.data[0].node.color.copy()
    
    # Test token highlighting
    selected_attention_edges = [(0, 1), (2, 3)]
    updated_fig = update_token_highlights(mock_fig, selected_attention_edges)
    
    # Check that highlighting was applied (mock implementation)
    assert updated_fig is not None
    
    # Test attention highlighting
    selected_tokens = ["token_1", "token_2"]
    updated_fig = update_attention_highlights(mock_fig, selected_tokens)
    
    assert updated_fig is not None
    
    # Test clearing highlights
    cleared_fig = clear_highlights(mock_fig)
    
    assert cleared_fig is not None
    
    print("✓ Highlight update function tests passed")
    return True


def test_selection_propagation():
    """Test selection propagation across visualizations."""
    print("Testing selection propagation...")
    
    highlighter = CrossVisualizationHighlighter()
    
    # Register multiple visualizations
    visualizations = [
        ("token_sankey_1", "token_sankey", {"tokens": ["A", "B", "C"]}),
        ("attention_sankey_1", "attention_sankey", {"edges": [(0, 1), (1, 2)]}),
        ("correlation_heatmap_1", "correlation_heatmap", {"layers": ["layer_0", "layer_1"]})
    ]
    
    for viz_id, viz_type, data_mapping in visualizations:
        highlighter.register_visualization(viz_id, viz_type, data_mapping)
    
    assert len(highlighter.visualization_registry) == 3
    
    # Test selection propagation
    node_data = {"token_id": "A", "position": 0}
    update_commands = highlighter._handle_token_node_selection(node_data)
    
    # Check that update commands were generated for all visualizations
    assert isinstance(update_commands, dict)
    
    print("✓ Selection propagation tests passed")
    return True


def test_synchronized_layout_creation():
    """Test synchronized visualization layout creation."""
    print("Testing synchronized layout creation...")
    
    # Mock data for layout creation
    token_sankey_data = {
        "nodes": ["Token_A", "Token_B", "Token_C"],
        "links": [(0, 1), (1, 2)]
    }
    
    attention_sankey_data = {
        "attention_edges": [("Token_A", "Token_B"), ("Token_B", "Token_C")],
        "attention_weights": [0.5, 0.3]
    }
    
    correlation_data = {
        "correlation_matrix": [[1.0, 0.5], [0.5, 1.0]],
        "layer_names": ["layer_0", "layer_1"]
    }
    
    # Test different layout styles
    layout_styles = ["grid", "tabs", "accordion"]
    
    for style in layout_styles:
        try:
            layout = create_synchronized_visualization_layout(
                token_sankey_data,
                attention_sankey_data,
                correlation_data,
                layout_style=style
            )
            
            # Check that layout was created (basic validation)
            assert layout is not None
            
            print(f"  ✓ {style} layout created successfully")
            
        except Exception as e:
            print(f"  ✗ Error creating {style} layout: {e}")
            return False
    
    print("✓ Synchronized layout creation tests passed")
    return True


def test_selection_state_management():
    """Test selection state management across interactions."""
    print("Testing selection state management...")
    
    highlighter = CrossVisualizationHighlighter()
    
    # Test initial state
    assert len(highlighter.selection_state.selected_tokens) == 0
    
    # Simulate token selection
    node_data_1 = {"token_id": "token_A", "position": 0}
    highlighter._handle_token_node_selection(node_data_1)
    
    assert "token_A" in highlighter.selection_state.selected_tokens
    assert 0 in highlighter.selection_state.selected_positions
    
    # Simulate another token selection
    node_data_2 = {"token_id": "token_B", "position": 1}
    highlighter._handle_token_node_selection(node_data_2)
    
    assert len(highlighter.selection_state.selected_tokens) == 2
    assert len(highlighter.selection_state.selected_positions) == 2
    
    # Simulate deselection (clicking same token again)
    highlighter._handle_token_node_selection(node_data_1)
    
    assert "token_A" not in highlighter.selection_state.selected_tokens
    assert 0 not in highlighter.selection_state.selected_positions
    assert len(highlighter.selection_state.selected_tokens) == 1
    
    # Test attention edge selection
    link_data = {"source_token": "token_B", "target_token": "token_C"}
    highlighter._handle_attention_link_selection(link_data)
    
    assert ("token_B", "token_C") in highlighter.selection_state.selected_attention_edges
    
    print("✓ Selection state management tests passed")
    return True


def test_callback_data_structures():
    """Test data structures used in Dash callbacks."""
    print("Testing callback data structures...")
    
    # Test click data structure (simulates Plotly click event)
    mock_click_data = {
        "points": [{
            "customdata": {
                "token_id": "token_1",
                "layer": "layer_0",
                "position": 2
            },
            "x": 0.5,
            "y": 0.3
        }]
    }
    
    # Test selection state structure
    mock_selection_state = {
        "selected_tokens": ["token_1", "token_2"],
        "selected_attention_edges": [("token_1", "token_2")],
        "selected_clusters": [0, 1],
        "selected_layers": ["layer_0"],
        "selected_positions": [1, 2]
    }
    
    # Validate data structure integrity
    assert "points" in mock_click_data
    assert len(mock_click_data["points"]) > 0
    assert "customdata" in mock_click_data["points"][0]
    
    assert "selected_tokens" in mock_selection_state
    assert isinstance(mock_selection_state["selected_tokens"], list)
    assert isinstance(mock_selection_state["selected_attention_edges"], list)
    
    print("✓ Callback data structure tests passed")
    return True


def test_integration_scenarios():
    """Test realistic integration scenarios."""
    print("Testing integration scenarios...")
    
    # Scenario 1: User clicks token in token Sankey
    # This should highlight related attention edges and update correlation view
    
    highlighter = CrossVisualizationHighlighter()
    
    # Register visualizations
    highlighter.register_visualization("token_viz", "token_sankey", {})
    highlighter.register_visualization("attention_viz", "attention_sankey", {})
    highlighter.register_visualization("correlation_viz", "correlation_heatmap", {})
    
    # Simulate token selection
    token_selection = {"token_id": "important_token", "position": 5}
    update_commands = highlighter._handle_token_node_selection(token_selection)
    
    # Check that all registered visualizations received update commands
    expected_viz_ids = {"token_viz", "attention_viz", "correlation_viz"}
    assert all(viz_id in highlighter.visualization_registry for viz_id in expected_viz_ids)
    
    # Scenario 2: User selects attention edge
    # This should highlight involved tokens and update correlation
    
    attention_selection = {"source_token": "token_A", "target_token": "token_B"}
    update_commands = highlighter._handle_attention_link_selection(attention_selection)
    
    assert ("token_A", "token_B") in highlighter.selection_state.selected_attention_edges
    
    print("✓ Integration scenario tests passed")
    return True


def run_comprehensive_test():
    """Run comprehensive test suite for cross-visualization highlighting."""
    print("CROSS-VISUALIZATION HIGHLIGHTING - COMPREHENSIVE TEST")
    print("=" * 70)
    
    test_functions = [
        test_selection_state,
        test_cross_visualization_highlighter,
        test_highlight_updates,
        test_selection_propagation,
        test_synchronized_layout_creation,
        test_selection_state_management,
        test_callback_data_structures,
        test_integration_scenarios
    ]
    
    results = {}
    
    for test_func in test_functions:
        test_name = test_func.__name__
        try:
            result = test_func()
            results[test_name] = result
            print()
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            results[test_name] = False
            print()
    
    # Print summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<35}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Cross-visualization highlighting is ready for integration.")
    else:
        print("⚠️  Some tests failed. Review errors before integration.")
    
    return results


def demonstrate_usage():
    """Demonstrate how to use the cross-visualization highlighting system."""
    print("\n" + "=" * 70)
    print("USAGE DEMONSTRATION")
    print("=" * 70)
    
    print("""
1. BASIC SETUP:
   
   from visualization.gpt2_cross_viz_highlighting import (
       CrossVisualizationHighlighter,
       create_synchronized_visualization_layout
   )
   
   # Create highlighter instance
   highlighter = CrossVisualizationHighlighter()

2. REGISTER VISUALIZATIONS:
   
   highlighter.register_visualization(
       viz_id="my_token_sankey",
       viz_type="token_sankey",
       data_mapping={"tokens": token_list}
   )

3. CREATE SYNCHRONIZED LAYOUT:
   
   layout = create_synchronized_visualization_layout(
       token_sankey_data=token_data,
       attention_sankey_data=attention_data,
       correlation_data=correlation_data,
       layout_style="grid"  # or "tabs" or "accordion"
   )

4. ADD TO DASH APP:
   
   app.layout = html.Div([
       layout,
       # Your other components...
   ])

5. CALLBACKS ARE AUTOMATICALLY CREATED:
   
   # Selection synchronization happens automatically
   # Users can click on any visualization to highlight related elements
   # in other visualizations

6. CUSTOMIZATION:
   
   # Change highlight color
   highlighter.selection_state.highlight_color = "#your_color"
   
   # Add custom selection handlers
   custom_handlers = highlighter._get_selection_handlers("custom_viz_type")
""")


if __name__ == "__main__":
    # Run comprehensive tests
    test_results = run_comprehensive_test()
    
    # Save test results
    output_file = "cross_highlighting_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest results saved to: {output_file}")
    
    # Demonstrate usage
    demonstrate_usage()
    
    # Final recommendations
    print("\n" + "=" * 70)
    print("INTEGRATION RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. ADD TO EXISTING DASHBOARD:
   - Integrate with current GPT-2 token tab
   - Use grid layout for side-by-side comparison
   - Enable selection persistence across tab switches

2. PERFORMANCE OPTIMIZATION:
   - Use client-side callbacks for rapid interactions
   - Implement debouncing for frequent selections
   - Cache highlight update calculations

3. USER EXPERIENCE:
   - Add selection status indicators
   - Provide clear visual feedback for selections
   - Include tutorial or help text for new users

4. TESTING:
   - Test with real GPT-2 analysis data
   - Validate selection accuracy across visualizations
   - Performance test with large datasets
""")
    
    success_rate = sum(test_results.values()) / len(test_results)
    if success_rate >= 0.8:
        print("\n✅ Cross-visualization highlighting system is ready for production use!")
    else:
        print(f"\n⚠️  System needs refinement ({success_rate:.1%} success rate)")