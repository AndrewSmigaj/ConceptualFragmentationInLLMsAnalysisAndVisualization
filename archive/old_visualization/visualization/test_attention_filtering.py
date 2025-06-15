"""
Test script for GPT-2 attention interactive filtering functionality.

This script tests the new interactive filtering components with mock data
to ensure integration with existing visualization components works correctly.
"""

import numpy as np
import json
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the new interactive filtering components
from visualization.gpt2_attention_interactive import (
    filter_attention_data,
    create_filtered_attention_sankey,
    create_attention_head_analysis,
    create_filter_summary_stats
)

# Import existing components to test integration
from visualization.gpt2_attention_sankey import extract_attention_flow
from visualization.gpt2_attention_correlation import calculate_correlation_metrics


def create_mock_attention_data(batch_size=2, seq_len=10, n_layers=4, n_heads=6):
    """Create mock attention data for testing."""
    print(f"Creating mock attention data: {n_layers} layers, {n_heads} heads, {seq_len} tokens")
    
    attention_data = {}
    
    for layer_idx in range(n_layers):
        layer_name = f"layer_{layer_idx}"
        
        # Create attention tensor [batch_size, n_heads, seq_len, seq_len]
        attention = np.random.rand(batch_size, n_heads, seq_len, seq_len)
        
        # Normalize to create valid attention distributions
        for b in range(batch_size):
            for h in range(n_heads):
                for i in range(seq_len):
                    attention[b, h, i] = attention[b, h, i] / attention[b, h, i].sum()
        
        # Add some realistic patterns
        # Head 0: Strong self-attention
        for b in range(batch_size):
            np.fill_diagonal(attention[b, 0], attention[b, 0].diagonal() * 2)
            # Renormalize
            for i in range(seq_len):
                attention[b, 0, i] = attention[b, 0, i] / attention[b, 0, i].sum()
        
        # Head 1: Local attention pattern
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    if abs(i - j) <= 2:  # Local window
                        attention[b, 1, i, j] *= 1.5
            # Renormalize
            for i in range(seq_len):
                attention[b, 1, i] = attention[b, 1, i] / attention[b, 1, i].sum()
        
        attention_data[layer_name] = attention
    
    return attention_data


def create_mock_token_metadata(batch_size=2, seq_len=10):
    """Create mock token metadata for testing."""
    print(f"Creating mock token metadata: {batch_size} batches, {seq_len} tokens")
    
    token_words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]
    
    token_metadata = {
        "tokens": [token_words[:seq_len] for _ in range(batch_size)],
        "token_ids": np.arange(seq_len * batch_size).reshape(batch_size, seq_len),
        "attention_mask": np.ones((batch_size, seq_len), dtype=int)
    }
    
    return token_metadata


def test_attention_filtering():
    """Test the attention filtering functionality."""
    print("\n" + "="*60)
    print("TESTING ATTENTION FILTERING FUNCTIONALITY")
    print("="*60)
    
    # Create mock data
    attention_data = create_mock_attention_data()
    token_metadata = create_mock_token_metadata()
    
    print(f"\nOriginal attention data:")
    for layer_name, attention in attention_data.items():
        print(f"  {layer_name}: {attention.shape}")
    
    # Test 1: Basic threshold filtering
    print("\n1. Testing threshold filtering...")
    filter_result = filter_attention_data(
        attention_data=attention_data,
        attention_threshold=0.1
    )
    
    print(f"   Filter stats: {filter_result['filter_stats']}")
    print(f"   Retention rate: {filter_result['filter_stats']['retention_rate']:.2%}")
    
    # Test 2: Layer range filtering
    print("\n2. Testing layer range filtering...")
    filter_result = filter_attention_data(
        attention_data=attention_data,
        layer_range=(1, 2)
    )
    
    print(f"   Filtered layers: {filter_result['layer_names']}")
    print(f"   Original layers: {list(attention_data.keys())}")
    
    # Test 3: Head filtering
    print("\n3. Testing attention head filtering...")
    filter_result = filter_attention_data(
        attention_data=attention_data,
        attention_heads=[0, 1, 2]  # Select first 3 heads
    )
    
    print(f"   Filters applied: {filter_result['filter_stats']['filters_applied']}")
    
    # Test 4: Pattern type filtering
    print("\n4. Testing pattern type filtering...")
    filter_result = filter_attention_data(
        attention_data=attention_data,
        pattern_types=["local", "self"]
    )
    
    print(f"   Pattern filters: {filter_result['filter_stats']['filters_applied']}")
    
    # Test 5: Combined filtering
    print("\n5. Testing combined filtering...")
    filter_result = filter_attention_data(
        attention_data=attention_data,
        attention_threshold=0.05,
        layer_range=(0, 2),
        attention_heads=[0, 1],
        pattern_types=["local", "global", "self"]
    )
    
    print(f"   Combined filters: {filter_result['filter_stats']['filters_applied']}")
    print(f"   Retention rate: {filter_result['filter_stats']['retention_rate']:.2%}")
    
    return filter_result


def test_head_specialization_analysis():
    """Test the attention head specialization analysis."""
    print("\n" + "="*60)
    print("TESTING HEAD SPECIALIZATION ANALYSIS")
    print("="*60)
    
    # Create mock data
    attention_data = create_mock_attention_data()
    
    # Analyze head specialization
    head_analysis = create_attention_head_analysis(attention_data)
    
    print(f"\nHead analysis results:")
    print(f"  Total heads analyzed: {len(head_analysis['head_options'])}")
    print(f"  Layers analyzed: {len(head_analysis['specialization_by_layer'])}")
    
    # Show specialization breakdown
    specialization_counts = {}
    for layer_name, layer_heads in head_analysis['specialization_by_layer'].items():
        print(f"\n  {layer_name}:")
        for head_idx, head_info in layer_heads.items():
            spec = head_info['specialization']
            specialization_counts[spec] = specialization_counts.get(spec, 0) + 1
            print(f"    Head {head_idx}: {spec} (entropy: {head_info['mean_entropy']:.3f})")
    
    print(f"\nOverall specialization distribution:")
    for spec, count in specialization_counts.items():
        print(f"  {spec}: {count} heads")
    
    return head_analysis


def test_integration_with_existing_components():
    """Test integration with existing visualization components."""
    print("\n" + "="*60)
    print("TESTING INTEGRATION WITH EXISTING COMPONENTS")
    print("="*60)
    
    # Create mock data
    attention_data = create_mock_attention_data()
    token_metadata = create_mock_token_metadata()
    
    # Test 1: Integration with attention flow extraction
    print("\n1. Testing attention flow extraction integration...")
    
    # Filter attention data
    filter_result = filter_attention_data(
        attention_data=attention_data,
        attention_threshold=0.05,
        pattern_types=["local", "global"]
    )
    
    filtered_attention = filter_result["filtered_attention_data"]
    
    try:
        # Extract attention flow using existing component
        attention_flow = extract_attention_flow(
            attention_data=filtered_attention,
            token_metadata=token_metadata,
            min_attention=0.05
        )
        
        print(f"   ✓ Successfully extracted attention flow")
        print(f"   Layers processed: {len(attention_flow.get('layers', []))}")
        print(f"   Token importance data: {len(attention_flow.get('token_importance', {}))}")
        
    except Exception as e:
        print(f"   ✗ Error in attention flow extraction: {e}")
        return False
    
    # Test 2: Integration with filtered Sankey creation
    print("\n2. Testing filtered Sankey diagram creation...")
    
    try:
        # Create filtered Sankey diagram
        filter_params = {
            "attention_threshold": 0.05,
            "pattern_types": ["local", "global", "self"]
        }
        
        fig = create_filtered_attention_sankey(
            attention_data=attention_data,
            token_metadata=token_metadata,
            filter_params=filter_params,
            max_edges=50
        )
        
        print(f"   ✓ Successfully created filtered Sankey diagram")
        print(f"   Figure type: {type(fig).__name__}")
        print(f"   Has data: {len(fig.data) > 0}")
        
    except Exception as e:
        print(f"   ✗ Error in Sankey creation: {e}")
        return False
    
    print("\n✓ All integration tests passed!")
    return True


def test_filter_summary_stats():
    """Test filter summary statistics component."""
    print("\n" + "="*60)
    print("TESTING FILTER SUMMARY STATISTICS")
    print("="*60)
    
    # Create mock filter stats
    mock_stats = {
        "filters_applied": ["threshold_0.05", "layers_0_2", "heads_3", "patterns_2"],
        "retention_rate": 0.67,
        "filtered_layers": 3,
        "original_layers": 4,
        "total_attention_values": 1920,
        "filtered_attention_values": 1286
    }
    
    try:
        # Create filter summary component
        summary_component = create_filter_summary_stats(mock_stats)
        
        print(f"   ✓ Successfully created filter summary component")
        print(f"   Component type: {type(summary_component).__name__}")
        
        # Check if it's a valid Dash component
        if hasattr(summary_component, 'children'):
            print(f"   Has children: {len(summary_component.children) if summary_component.children else 0}")
        
    except Exception as e:
        print(f"   ✗ Error creating filter summary: {e}")
        return False
    
    return True


def run_comprehensive_test():
    """Run comprehensive test of all filtering functionality."""
    print("GPT-2 ATTENTION INTERACTIVE FILTERING - COMPREHENSIVE TEST")
    print("=" * 80)
    
    # Track test results
    test_results = {}
    
    # Run individual tests
    try:
        filter_result = test_attention_filtering()
        test_results["attention_filtering"] = True
    except Exception as e:
        print(f"✗ Attention filtering test failed: {e}")
        test_results["attention_filtering"] = False
    
    try:
        head_analysis = test_head_specialization_analysis()
        test_results["head_analysis"] = True
    except Exception as e:
        print(f"✗ Head specialization test failed: {e}")
        test_results["head_analysis"] = False
    
    try:
        integration_success = test_integration_with_existing_components()
        test_results["integration"] = integration_success
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        test_results["integration"] = False
    
    try:
        summary_success = test_filter_summary_stats()
        test_results["summary_stats"] = summary_success
    except Exception as e:
        print(f"✗ Summary stats test failed: {e}")
        test_results["summary_stats"] = False
    
    # Print final results
    print("\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:<25}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Interactive filtering is ready for integration.")
    else:
        print("⚠️  Some tests failed. Review errors before integration.")
    
    return test_results


if __name__ == "__main__":
    # Run the comprehensive test
    results = run_comprehensive_test()
    
    # Save test results
    output_file = "attention_filtering_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to: {output_file}")
    
    # Demonstrate usage
    print("\n" + "="*80)
    print("USAGE EXAMPLE")
    print("="*80)
    print("""
To integrate the interactive filtering with your dashboard:

1. Import the components:
   from visualization.gpt2_attention_interactive import (
       create_attention_filter_controls,
       filter_attention_data,
       create_filtered_attention_sankey
   )

2. Add filter controls to your layout:
   filter_controls = create_attention_filter_controls()

3. Use filtering in callbacks:
   filtered_data = filter_attention_data(
       attention_data=your_attention_data,
       attention_threshold=0.05,
       layer_range=(0, 5),
       pattern_types=["local", "global"]
   )

4. Create filtered visualizations:
   fig = create_filtered_attention_sankey(
       attention_data=your_attention_data,
       token_metadata=your_token_metadata,
       filter_params=your_filter_params
   )
""")