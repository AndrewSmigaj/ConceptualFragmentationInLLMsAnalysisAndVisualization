# GPT-2 Attention Interactive Filtering Integration Guide

This guide explains how to integrate the new interactive filtering capabilities with the existing GPT-2 attention visualizations in the dashboard.

## Overview

The interactive filtering system extends existing attention visualizations with the following capabilities:

- **Attention Head Filtering**: Select specific attention heads for analysis
- **Layer Range Filtering**: Focus on specific transformer layers
- **Pattern Type Filtering**: Filter by attention patterns (local, global, self, forward, backward)
- **Threshold Filtering**: Set minimum attention values to display
- **Advanced Filtering**: Entropy-based and correlation-based filtering

## Integration Steps

### Step 1: Import New Components

Add the following imports to your dashboard files:

```python
# In visualization/gpt2_token_tab.py or your main dashboard file
from visualization.gpt2_attention_interactive import (
    create_attention_filter_controls,
    filter_attention_data,
    create_filtered_attention_sankey,
    create_attention_head_analysis,
    create_filter_summary_stats,
    create_interactive_attention_tab
)
```

### Step 2: Add Filter Controls to Layout

#### Option A: Add to Existing Tab

```python
# In your existing GPT-2 token tab layout
def create_gpt2_token_tab_layout():
    return html.Div([
        # Existing components...
        
        # Add filter controls section
        html.Hr(),
        html.H4("Interactive Attention Filtering"),
        create_attention_filter_controls(),
        
        # Filtered visualization
        html.Div(id="filter-summary-stats"),
        dcc.Graph(
            id="filtered-attention-visualization",
            style={"height": "600px"}
        ),
        
        # Rest of existing layout...
    ])
```

#### Option B: Create New Tab

```python
# Add new tab to your dashboard tabs
dbc.Tab(
    label="Interactive Attention",
    tab_id="interactive-attention-tab",
    children=[
        create_interactive_attention_tab()
    ]
)
```

### Step 3: Add Callback for Filter Updates

```python
@app.callback(
    [Output("filtered-attention-visualization", "figure"),
     Output("filter-summary-stats", "children")],
    [Input("attention-head-filter", "value"),
     Input("layer-range-filter", "value"),
     Input("attention-threshold-filter", "value"),
     Input("max-edges-filter", "value"),
     Input("attention-pattern-filter", "value")],
    [State("gpt2-analysis-selector", "value")]
)
def update_filtered_attention_visualization(head_filter, layer_range, threshold, 
                                          max_edges, pattern_filter, selected_analysis):
    """Update attention visualization based on filter settings."""
    
    if not selected_analysis:
        return go.Figure(), html.Div()
    
    try:
        # Load your attention data (replace with actual data loading)
        attention_data, token_metadata = load_attention_data(selected_analysis)
        
        # Apply filters
        filter_params = {
            "attention_heads": head_filter,
            "layer_range": tuple(layer_range) if layer_range else None,
            "attention_threshold": threshold,
            "pattern_types": pattern_filter
        }
        
        # Create filtered visualization
        fig = create_filtered_attention_sankey(
            attention_data=attention_data,
            token_metadata=token_metadata,
            filter_params=filter_params,
            max_edges=max_edges
        )
        
        # Get filter statistics
        filter_result = filter_attention_data(attention_data, **filter_params)
        summary_stats = create_filter_summary_stats(filter_result["filter_stats"])
        
        return fig, summary_stats
        
    except Exception as e:
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return error_fig, html.Div()
```

### Step 4: Update Filter Options Dynamically

```python
@app.callback(
    [Output("attention-head-filter", "options"),
     Output("layer-range-filter", "max"),
     Output("layer-range-filter", "marks")],
    [Input("gpt2-analysis-selector", "value")]
)
def update_filter_options(selected_analysis):
    """Update filter options based on selected analysis."""
    
    if not selected_analysis:
        return [], 12, {}
    
    try:
        # Load analysis metadata to get available heads and layers
        metadata = load_analysis_metadata(selected_analysis)
        
        # Create head options
        head_options = []
        for layer_name in metadata.get("layers", []):
            n_heads = metadata.get("n_heads_per_layer", {}).get(layer_name, 12)
            for head_idx in range(n_heads):
                head_options.append({
                    "label": f"{layer_name} Head {head_idx}",
                    "value": f"{layer_name}_head_{head_idx}"
                })
        
        # Create layer range options
        max_layers = len(metadata.get("layers", [])) - 1
        layer_marks = {i: f"L{i}" for i in range(0, max_layers + 1, 2)}
        
        return head_options, max_layers, layer_marks
        
    except Exception as e:
        print(f"Error updating filter options: {e}")
        return [], 12, {}
```

## Data Loading Integration

### Modify Existing Data Loading

Update your existing data loading functions to include attention data:

```python
def load_gpt2_analysis_data(analysis_path):
    """Load GPT-2 analysis data including attention patterns."""
    
    # Load existing data
    with open(analysis_path, 'r') as f:
        analysis_data = json.load(f)
    
    # Load attention data if available
    attention_data = {}
    if "attention_files" in analysis_data:
        for layer_name, attention_file in analysis_data["attention_files"].items():
            attention_path = os.path.join(os.path.dirname(analysis_path), attention_file)
            if os.path.exists(attention_path):
                attention_data[layer_name] = np.load(attention_path)
    
    # Load token metadata
    token_metadata = analysis_data.get("token_metadata", {})
    
    return {
        "analysis_data": analysis_data,
        "attention_data": attention_data,
        "token_metadata": token_metadata
    }
```

## Advanced Integration Features

### 1. Head Specialization Analysis

```python
@app.callback(
    Output("head-specialization-content", "children"),
    [Input("gpt2-analysis-selector", "value")]
)
def update_head_specialization_analysis(selected_analysis):
    """Update head specialization analysis."""
    
    if not selected_analysis:
        return html.Div("Select an analysis to view head specialization")
    
    try:
        attention_data, _ = load_attention_data(selected_analysis)
        head_analysis = create_attention_head_analysis(attention_data)
        
        # Create specialization summary table
        specialization_data = []
        for layer_name, layer_heads in head_analysis["specialization_by_layer"].items():
            for head_idx, head_info in layer_heads.items():
                specialization_data.append({
                    "Layer": layer_name,
                    "Head": head_idx,
                    "Specialization": head_info["specialization"],
                    "Entropy": f"{head_info['mean_entropy']:.3f}",
                    "Local Attention": f"{head_info['local_attention']:.3f}",
                    "Global Attention": f"{head_info['global_attention']:.3f}"
                })
        
        return dash_table.DataTable(
            data=specialization_data,
            columns=[{"name": col, "id": col} for col in specialization_data[0].keys()],
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Specialization} = concept_formation'},
                    'backgroundColor': '#e8f5e8',
                },
                {
                    'if': {'filter_query': '{Specialization} = integration'},
                    'backgroundColor': '#fff2e8',
                }
            ]
        )
        
    except Exception as e:
        return html.Div(f"Error loading head specialization: {str(e)}", style={"color": "red"})
```

### 2. Real-time Filter Performance

```python
@app.callback(
    Output("filter-performance-indicator", "children"),
    [Input("attention-threshold-filter", "value"),
     Input("max-edges-filter", "value")],
    [State("gpt2-analysis-selector", "value")]
)
def update_filter_performance(threshold, max_edges, selected_analysis):
    """Show real-time filter performance indicators."""
    
    if not selected_analysis:
        return html.Div()
    
    # Estimate performance impact
    if threshold > 0.1:
        performance_color = "success"
        performance_text = "High performance (few edges)"
    elif threshold > 0.05:
        performance_color = "warning" 
        performance_text = "Medium performance"
    else:
        performance_color = "danger"
        performance_text = "Low performance (many edges)"
    
    return dbc.Alert([
        html.Strong("Filter Performance: "),
        performance_text,
        html.Br(),
        html.Small(f"Threshold: {threshold}, Max edges: {max_edges}")
    ], color=performance_color, className="mt-2")
```

## Testing Integration

### Unit Tests

Create tests for the integration:

```python
def test_filter_integration():
    """Test filter integration with existing components."""
    
    # Mock data
    attention_data = create_mock_attention_data()
    token_metadata = create_mock_token_metadata()
    
    # Test filter application
    filter_params = {"attention_threshold": 0.05}
    filtered_result = filter_attention_data(attention_data, **filter_params)
    
    assert "filtered_attention_data" in filtered_result
    assert "filter_stats" in filtered_result
    
    # Test visualization creation
    fig = create_filtered_attention_sankey(
        attention_data, token_metadata, filter_params
    )
    
    assert fig is not None
    assert len(fig.data) > 0
```

### Integration Tests

```python
def test_dashboard_integration():
    """Test full dashboard integration."""
    
    # Test callback functionality
    # Test filter option updates
    # Test visualization updates
    # Test error handling
    
    pass
```

## Performance Considerations

### Optimization Tips

1. **Lazy Loading**: Load attention data only when needed
2. **Caching**: Cache filtered results for common filter combinations
3. **Chunking**: Process large attention matrices in chunks
4. **Debouncing**: Debounce filter updates to avoid excessive recomputation

```python
import time
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_filter_attention_data(attention_data_hash, **filter_params):
    """Cached version of filter_attention_data for performance."""
    # Implementation with caching
    pass

# Debounced callback
@app.callback(
    Output("filtered-attention-visualization", "figure"),
    [Input("attention-threshold-filter", "value")],
    prevent_initial_call=True
)
def debounced_update_visualization(threshold):
    """Debounced visualization update."""
    time.sleep(0.3)  # Simple debouncing
    # Implementation
    pass
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce max_edges or increase threshold
2. **Slow Updates**: Enable caching and debouncing
3. **Missing Data**: Check attention data loading
4. **Filter Conflicts**: Validate filter combinations

### Debug Mode

```python
DEBUG_FILTERING = True

if DEBUG_FILTERING:
    @app.callback(
        Output("debug-info", "children"),
        [Input("attention-threshold-filter", "value")]
    )
    def show_debug_info(threshold):
        return html.Pre(f"Debug: threshold={threshold}")
```

## Complete Example

See `visualization/test_attention_filtering.py` for a complete working example that demonstrates:

- Filter functionality testing
- Integration with existing components
- Error handling
- Performance validation

The interactive filtering system is designed to seamlessly extend existing GPT-2 attention visualizations while maintaining backward compatibility and performance.