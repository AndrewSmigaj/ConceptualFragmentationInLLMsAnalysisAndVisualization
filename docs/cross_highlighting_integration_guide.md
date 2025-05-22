# Cross-Visualization Highlighting Integration Guide

This guide explains how to integrate the synchronized highlighting system with the existing GPT-2 APA dashboard to enable coordinated exploration across visualizations.

## Overview

The cross-visualization highlighting system enables users to:
- Click tokens in path visualizations to highlight related attention patterns
- Select attention edges to highlight involved tokens
- Synchronize selections across correlation heatmaps
- Maintain selection state across different visualization tabs
- Clear and manage selections with intuitive controls

## Integration Architecture

### Component Hierarchy

```
Dashboard
├── Existing GPT-2 Token Tab
│   ├── Token Path Visualizations
│   ├── Attention Visualizations
│   └── NEW: Cross-Highlighting System
├── Selection State Management
│   ├── Global Selection Store
│   ├── Selection Propagation Logic
│   └── Highlight Update Handlers
└── Synchronized Callbacks
    ├── Token Selection Callbacks
    ├── Attention Selection Callbacks
    └── Cross-Update Callbacks
```

## Step-by-Step Integration

### Step 1: Import Cross-Highlighting Components

Add imports to your main dashboard file:

```python
# In visualization/dash_app.py or your main dashboard file
from visualization.gpt2_cross_viz_highlighting import (
    CrossVisualizationHighlighter,
    SelectionState,
    create_synchronized_visualization_layout,
    create_selection_callbacks,
    update_token_highlights,
    update_attention_highlights,
    clear_highlights
)
```

### Step 2: Modify Existing GPT-2 Token Tab

Update your existing GPT-2 token tab to include cross-highlighting:

```python
# In visualization/gpt2_token_tab.py

def create_enhanced_gpt2_token_tab():
    """Create GPT-2 token tab with cross-visualization highlighting."""
    
    return html.Div([
        # Existing analysis selector
        create_analysis_selector(),
        
        html.Hr(),
        
        # NEW: Synchronized visualization section
        html.Div([
            html.H4("Synchronized Analysis", className="mb-3"),
            
            # Selection controls
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Clear All Selections",
                        id="clear-all-selections",
                        color="secondary",
                        size="sm",
                        className="me-2"
                    ),
                    dbc.Button(
                        "Highlight Connected Elements",
                        id="highlight-connected",
                        color="primary",
                        size="sm"
                    )
                ], width=6),
                dbc.Col([
                    html.Div(
                        id="selection-status-display",
                        className="text-muted small"
                    )
                ], width=6)
            ], className="mb-3"),
            
            # Synchronized visualizations
            html.Div(id="synchronized-visualizations-container")
            
        ]),
        
        html.Hr(),
        
        # Existing individual visualizations (can be kept for detailed analysis)
        html.Div([
            html.H4("Individual Analysis", className="mb-3"),
            # Your existing individual visualization components
        ])
    ])
```

### Step 3: Create Selection State Management

Add a global selection store to your app:

```python
# Global selection state store
app.layout = html.Div([
    # Existing layout components...
    
    # NEW: Global selection state
    dcc.Store(
        id="global-selection-state",
        data={
            "selected_tokens": [],
            "selected_attention_edges": [],
            "selected_clusters": [],
            "selected_layers": [],
            "selected_positions": [],
            "highlight_color": "#ff6b6b"
        }
    ),
    
    # Existing layout continues...
])
```

### Step 4: Add Main Synchronized Visualization Callback

```python
@app.callback(
    Output("synchronized-visualizations-container", "children"),
    [Input("gpt2-analysis-selector", "value")]
)
def create_synchronized_visualizations(selected_analysis):
    """Create synchronized visualization layout based on selected analysis."""
    
    if not selected_analysis:
        return html.Div("Select an analysis to view synchronized visualizations")
    
    try:
        # Load analysis data
        analysis_data = load_gpt2_analysis_data(selected_analysis)
        
        token_sankey_data = extract_token_path_data(analysis_data)
        attention_sankey_data = extract_attention_flow_data(analysis_data)
        correlation_data = extract_correlation_data(analysis_data)
        
        # Create synchronized layout
        synchronized_layout = create_synchronized_visualization_layout(
            token_sankey_data=token_sankey_data,
            attention_sankey_data=attention_sankey_data,
            correlation_data=correlation_data,
            layout_style="grid"  # Can be made configurable
        )
        
        return synchronized_layout
        
    except Exception as e:
        return dbc.Alert(f"Error creating synchronized visualizations: {str(e)}", color="danger")
```

### Step 5: Implement Selection Synchronization Callbacks

```python
# Token selection callback
@app.callback(
    [Output("global-selection-state", "data"),
     Output("attention-sankey-synchronized", "figure"),
     Output("correlation-heatmap-synchronized", "figure")],
    [Input("token-sankey-synchronized", "clickData")],
    [State("global-selection-state", "data"),
     State("attention-sankey-synchronized", "figure"),
     State("correlation-heatmap-synchronized", "figure")]
)
def handle_token_selection_sync(click_data, selection_state, attention_fig, correlation_fig):
    """Handle token selection and update related visualizations."""
    
    if not click_data:
        return selection_state, attention_fig, correlation_fig
    
    # Parse click data
    point = click_data["points"][0]
    
    # Update selection state
    updated_state = dict(selection_state)
    
    # Handle node click (token selection)
    if "customdata" in point:
        token_info = point["customdata"]
        token_id = token_info.get("token_id")
        position = token_info.get("position")
        
        if token_id:
            # Toggle token selection
            if token_id in updated_state["selected_tokens"]:
                updated_state["selected_tokens"].remove(token_id)
                if position in updated_state["selected_positions"]:
                    updated_state["selected_positions"].remove(position)
            else:
                updated_state["selected_tokens"].append(token_id)
                if position is not None:
                    updated_state["selected_positions"].append(position)
    
    # Update attention visualization
    updated_attention_fig = update_attention_highlights(
        attention_fig,
        updated_state["selected_tokens"]
    )
    
    # Update correlation visualization
    updated_correlation_fig = update_correlation_highlights(
        correlation_fig,
        updated_state["selected_tokens"]
    )
    
    return updated_state, updated_attention_fig, updated_correlation_fig


# Attention selection callback
@app.callback(
    [Output("global-selection-state", "data", allow_duplicate=True),
     Output("token-sankey-synchronized", "figure"),
     Output("correlation-heatmap-synchronized", "figure", allow_duplicate=True)],
    [Input("attention-sankey-synchronized", "clickData")],
    [State("global-selection-state", "data"),
     State("token-sankey-synchronized", "figure"),
     State("correlation-heatmap-synchronized", "figure")],
    prevent_initial_call=True
)
def handle_attention_selection_sync(click_data, selection_state, token_fig, correlation_fig):
    """Handle attention selection and update related visualizations."""
    
    if not click_data:
        return selection_state, token_fig, correlation_fig
    
    # Parse attention click data
    point = click_data["points"][0]
    
    updated_state = dict(selection_state)
    
    # Handle link click (attention edge selection)
    if "source" in point and "target" in point:
        source_idx = point["source"]
        target_idx = point["target"]
        
        # Get source and target information from customdata if available
        source_token = point.get("customdata", {}).get("source_token", f"token_{source_idx}")
        target_token = point.get("customdata", {}).get("target_token", f"token_{target_idx}")
        
        edge = (source_token, target_token)
        
        # Toggle edge selection
        if edge in updated_state["selected_attention_edges"]:
            updated_state["selected_attention_edges"].remove(edge)
        else:
            updated_state["selected_attention_edges"].append(edge)
        
        # Also select the involved tokens
        for token in [source_token, target_token]:
            if token not in updated_state["selected_tokens"]:
                updated_state["selected_tokens"].append(token)
    
    # Update token visualization
    updated_token_fig = update_token_highlights(
        token_fig,
        updated_state["selected_attention_edges"]
    )
    
    # Update correlation visualization
    updated_correlation_fig = update_correlation_highlights(
        correlation_fig,
        updated_state["selected_tokens"]
    )
    
    return updated_state, updated_token_fig, updated_correlation_fig


# Clear selections callback
@app.callback(
    [Output("global-selection-state", "data", allow_duplicate=True),
     Output("token-sankey-synchronized", "figure", allow_duplicate=True),
     Output("attention-sankey-synchronized", "figure", allow_duplicate=True),
     Output("correlation-heatmap-synchronized", "figure", allow_duplicate=True)],
    [Input("clear-all-selections", "n_clicks")],
    [State("token-sankey-synchronized", "figure"),
     State("attention-sankey-synchronized", "figure"),
     State("correlation-heatmap-synchronized", "figure")],
    prevent_initial_call=True
)
def clear_all_selections(n_clicks, token_fig, attention_fig, correlation_fig):
    """Clear all selections across visualizations."""
    
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Reset selection state
    empty_state = {
        "selected_tokens": [],
        "selected_attention_edges": [],
        "selected_clusters": [],
        "selected_layers": [],
        "selected_positions": [],
        "highlight_color": "#ff6b6b"
    }
    
    # Clear highlights from all visualizations
    cleared_token_fig = clear_highlights(token_fig)
    cleared_attention_fig = clear_highlights(attention_fig)
    cleared_correlation_fig = clear_highlights(correlation_fig)
    
    return empty_state, cleared_token_fig, cleared_attention_fig, cleared_correlation_fig


# Selection status display callback
@app.callback(
    Output("selection-status-display", "children"),
    [Input("global-selection-state", "data")]
)
def update_selection_status(selection_state):
    """Update selection status display."""
    
    if not selection_state:
        return "No selections"
    
    status_parts = []
    
    n_tokens = len(selection_state.get("selected_tokens", []))
    if n_tokens > 0:
        status_parts.append(f"{n_tokens} token{'s' if n_tokens != 1 else ''}")
    
    n_edges = len(selection_state.get("selected_attention_edges", []))
    if n_edges > 0:
        status_parts.append(f"{n_edges} attention edge{'s' if n_edges != 1 else ''}")
    
    n_clusters = len(selection_state.get("selected_clusters", []))
    if n_clusters > 0:
        status_parts.append(f"{n_clusters} cluster{'s' if n_clusters != 1 else ''}")
    
    if not status_parts:
        return "No selections active"
    
    return f"Selected: {', '.join(status_parts)}"
```

### Step 6: Enhanced Data Loading

Update your data loading functions to support cross-highlighting:

```python
def load_gpt2_analysis_data_with_mapping(analysis_path):
    """Load GPT-2 analysis data with element mapping for cross-highlighting."""
    
    # Load base analysis data
    analysis_data = load_gpt2_analysis_data(analysis_path)
    
    # Create element mappings for cross-highlighting
    token_mapping = {}
    attention_mapping = {}
    
    # Map tokens to their positions and clusters
    if "token_metadata" in analysis_data:
        tokens = analysis_data["token_metadata"].get("tokens", [])
        for batch_idx, batch_tokens in enumerate(tokens):
            for pos_idx, token in enumerate(batch_tokens):
                token_key = f"batch_{batch_idx}_pos_{pos_idx}"
                token_mapping[token_key] = {
                    "token_text": token,
                    "batch": batch_idx,
                    "position": pos_idx,
                    "token_id": token_key
                }
    
    # Map attention edges to their source/target tokens
    if "attention_data" in analysis_data:
        attention_data = analysis_data["attention_data"]
        for layer_name, attention_matrix in attention_data.items():
            # Process attention matrix to create edge mappings
            # This depends on your specific attention data structure
            pass
    
    # Add mappings to analysis data
    analysis_data["token_mapping"] = token_mapping
    analysis_data["attention_mapping"] = attention_mapping
    
    return analysis_data
```

### Step 7: Performance Optimization

For better performance with large datasets:

```python
# Client-side callback for rapid interactions
app.clientside_callback(
    """
    function(clickData, currentState) {
        if (!clickData) {
            return window.dash_clientside.no_update;
        }
        
        // Handle rapid selection updates on client side
        const point = clickData.points[0];
        const tokenId = point.customdata && point.customdata.token_id;
        
        if (tokenId && currentState) {
            const newState = {...currentState};
            const tokenIndex = newState.selected_tokens.indexOf(tokenId);
            
            if (tokenIndex > -1) {
                newState.selected_tokens.splice(tokenIndex, 1);
            } else {
                newState.selected_tokens.push(tokenId);
            }
            
            return newState;
        }
        
        return window.dash_clientside.no_update;
    }
    """,
    Output("global-selection-state", "data", allow_duplicate=True),
    [Input("token-sankey-synchronized", "clickData")],
    [State("global-selection-state", "data")],
    prevent_initial_call=True
)

# Debounced update for expensive operations
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

@app.long_callback(
    output=Output("expensive-correlation-update", "figure"),
    inputs=[Input("global-selection-state", "data")],
    manager=long_callback_manager,
    prevent_initial_call=True
)
def expensive_correlation_update(selection_state):
    """Handle expensive correlation updates with debouncing."""
    time.sleep(0.5)  # Debounce rapid updates
    # Perform expensive correlation calculation
    return updated_correlation_figure
```

### Step 8: Add Advanced Features

#### Selection Persistence

```python
# Store selections in browser local storage
app.clientside_callback(
    """
    function(selectionState) {
        if (selectionState) {
            localStorage.setItem('gpt2_selections', JSON.stringify(selectionState));
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("dummy-output", "children"),
    [Input("global-selection-state", "data")]
)

# Load selections on page load
app.clientside_callback(
    """
    function() {
        const saved = localStorage.getItem('gpt2_selections');
        if (saved) {
            return JSON.parse(saved);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("global-selection-state", "data", allow_duplicate=True),
    [Input("app-load-trigger", "children")],
    prevent_initial_call=True
)
```

#### Keyboard Shortcuts

```python
# Add keyboard shortcuts for common actions
app.clientside_callback(
    """
    function(n_intervals) {
        document.addEventListener('keydown', function(event) {
            if (event.ctrlKey || event.metaKey) {
                switch(event.key) {
                    case 'a':
                        // Select all tokens
                        event.preventDefault();
                        document.getElementById('select-all-tokens').click();
                        break;
                    case 'x':
                        // Clear selections
                        event.preventDefault();
                        document.getElementById('clear-all-selections').click();
                        break;
                }
            }
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output("keyboard-handler", "children"),
    [Input("keyboard-interval", "n_intervals")]
)
```

## Testing Integration

### Unit Tests

```python
def test_cross_highlighting_integration():
    """Test cross-highlighting integration with existing dashboard."""
    
    # Test data loading with mappings
    analysis_data = load_gpt2_analysis_data_with_mapping("test_analysis.json")
    assert "token_mapping" in analysis_data
    assert "attention_mapping" in analysis_data
    
    # Test synchronized layout creation
    layout = create_synchronized_visualization_layout(
        analysis_data["token_data"],
        analysis_data["attention_data"],
        analysis_data["correlation_data"]
    )
    assert layout is not None
    
    # Test selection state management
    selection_state = {
        "selected_tokens": ["token_1", "token_2"],
        "selected_attention_edges": [("token_1", "token_2")]
    }
    
    # Verify selections propagate correctly
    # Test implementation depends on your specific setup
```

### Integration Tests

```python
def test_dashboard_integration():
    """Test full dashboard integration."""
    
    # Test with real analysis data
    real_analysis_path = "results/gpt2/sample_analysis.json"
    
    if os.path.exists(real_analysis_path):
        analysis_data = load_gpt2_analysis_data_with_mapping(real_analysis_path)
        
        # Test synchronized visualization creation
        layout = create_synchronized_visualization_layout(
            analysis_data["token_data"],
            analysis_data["attention_data"],
            analysis_data["correlation_data"]
        )
        
        # Verify all required components are present
        # Test selection callback functionality
        # Validate highlight update performance
```

## User Experience Enhancements

### Tutorial Mode

```python
# Add tutorial overlay for new users
tutorial_steps = [
    {
        "target": "#token-sankey-synchronized",
        "content": "Click on tokens here to highlight related attention patterns"
    },
    {
        "target": "#attention-sankey-synchronized", 
        "content": "Click on attention edges to highlight involved tokens"
    },
    {
        "target": "#clear-all-selections",
        "content": "Use this button to clear all selections"
    }
]

tutorial_component = html.Div([
    dbc.Button("Show Tutorial", id="show-tutorial", color="info", size="sm"),
    html.Div(id="tutorial-overlay", style={"display": "none"})
])
```

### Selection History

```python
# Track selection history for undo/redo
@app.callback(
    Output("selection-history-store", "data"),
    [Input("global-selection-state", "data")],
    [State("selection-history-store", "data")]
)
def track_selection_history(current_state, history):
    """Track selection history for undo functionality."""
    
    if not history:
        history = []
    
    # Add current state to history (limit to last 10 states)
    history.append(current_state)
    if len(history) > 10:
        history.pop(0)
    
    return history

@app.callback(
    Output("global-selection-state", "data", allow_duplicate=True),
    [Input("undo-selection", "n_clicks")],
    [State("selection-history-store", "data")],
    prevent_initial_call=True
)
def undo_selection(n_clicks, history):
    """Undo last selection."""
    
    if n_clicks and history and len(history) > 1:
        # Return to previous state
        return history[-2]
    
    return dash.no_update
```

## Troubleshooting

### Common Issues

1. **Selections not synchronizing**: Check callback dependencies and IDs
2. **Performance issues**: Implement debouncing and client-side callbacks
3. **Memory usage**: Clear old selection states and limit history
4. **Browser compatibility**: Test client-side callbacks across browsers

### Debug Mode

```python
DEBUG_HIGHLIGHTING = True

if DEBUG_HIGHLIGHTING:
    @app.callback(
        Output("debug-selection-info", "children"),
        [Input("global-selection-state", "data")]
    )
    def debug_selection_info(selection_state):
        if not selection_state:
            return "No selection data"
        
        return html.Pre(json.dumps(selection_state, indent=2))
```

## Summary

The cross-visualization highlighting system provides:

✅ **Synchronized token and attention selection**  
✅ **Real-time highlight updates across visualizations**  
✅ **Intuitive selection management controls**  
✅ **Performance optimization with client-side callbacks**  
✅ **Extensible architecture for additional visualization types**  

This enhancement significantly improves the user experience by enabling coordinated exploration of token paths and attention patterns, making it easier to understand the relationships between different aspects of the GPT-2 analysis.