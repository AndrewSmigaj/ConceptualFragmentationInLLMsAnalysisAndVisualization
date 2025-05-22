# GPT-2 Analysis Persistence Guide

This guide explains how to use the GPT-2 analysis persistence functionality to save, load, and manage your analysis results.

## Overview

The GPT-2 analysis persistence system provides comprehensive data management capabilities including:

- **Analysis Results Storage**: Save complete analysis results with versioning
- **Visualization States**: Save and restore specific visualization configurations
- **Session Management**: Organize multiple analyses into sessions
- **Export/Import**: Export analyses in multiple formats (JSON, CSV, Pickle)
- **Caching**: Intelligent caching for faster data access

## Quick Start

### Basic Usage

```python
from concept_fragmentation.persistence import GPT2AnalysisPersistence

# Create persistence manager
persistence = GPT2AnalysisPersistence()

# Save analysis results
analysis_id = persistence.save_analysis_results(
    analysis_data=your_analysis_data,
    model_name="gpt2-small",
    input_text="Your input text here"
)

# Load analysis results
loaded_data = persistence.load_analysis_results(analysis_id)
```

### Convenience Functions

For simple operations, use the convenience functions:

```python
from concept_fragmentation.persistence import save_gpt2_analysis, load_gpt2_analysis

# Save
analysis_id = save_gpt2_analysis(analysis_data, "gpt2-small", "Input text")

# Load
loaded_data = load_gpt2_analysis(analysis_id)
```

## Dashboard Integration

The GPT-2 metrics dashboard includes integrated persistence controls:

### Saving Analysis
1. Navigate to the GPT-2 Metrics Dashboard tab
2. Scroll to the "Analysis Persistence" section  
3. Enter a name for your analysis
4. Click "Save Analysis"

### Loading Analysis
1. Select a saved analysis from the dropdown
2. Click "Load Analysis"
3. The dashboard will update with the loaded data

### Exporting Analysis
1. Select export format (JSON, CSV, or Pickle)
2. Click "Export"
3. The file will be saved to the exports directory

## Directory Structure

The persistence system creates the following directory structure:

```
data/gpt2_analysis/
├── analysis_results/     # Saved analysis files
├── visualizations/       # Visualization state files
├── sessions/            # Session files
├── cache/              # Cache files
├── exports/            # Exported files
└── metadata/           # Analysis metadata for quick lookup
```

## Advanced Features

### Versioning
- Multiple versions of the same analysis are automatically tracked
- Old versions are automatically cleaned up (configurable limit)
- Load specific versions using the `version` parameter

### Visualization States
Save and restore specific visualization configurations:

```python
# Save visualization state
state_id = persistence.save_visualization_state(
    analysis_id=analysis_id,
    visualization_config=config_dict,
    visualization_type="token_sankey",
    state_name="my_custom_view"
)

# Load visualization state
state_data = persistence.load_visualization_state(state_id)
```

### Sessions
Organize multiple analyses into sessions:

```python
# Create session
session_id = persistence.create_session(
    session_name="My Analysis Session",
    analysis_ids=[analysis1_id, analysis2_id]
)

# Load session
session_data = persistence.load_session(session_id)
```

### Export Options
Export analyses in multiple formats:

```python
# Export as JSON (default)
export_path = persistence.export_analysis(analysis_id, export_format="json")

# Export as CSV (creates multiple CSV files for different data types)
export_path = persistence.export_analysis(analysis_id, export_format="csv")

# Export as Pickle (for Python compatibility)
export_path = persistence.export_analysis(analysis_id, export_format="pickle")
```

## Configuration Options

### Persistence Manager Configuration

```python
persistence = GPT2AnalysisPersistence(
    base_dir="custom/path",          # Custom storage directory
    enable_cache=True,               # Enable/disable caching
    cache_ttl=3600,                  # Cache time-to-live in seconds
    max_versions=10                  # Maximum versions to keep per analysis
)
```

### Cache Management

```python
# Get cache statistics
stats = persistence.cache.get_stats()
print(f"Cache size: {stats['size']} items")

# Clean up old cache files
persistence.cleanup_cache(max_age_hours=24)

# Clear all cache
persistence.cache.clear()
```

## Data Formats

### Analysis Data Structure
The persistence system expects analysis data in the following format:

```python
{
    "model_type": "gpt2-small",
    "layers": ["layer_0", "layer_1", ...],
    "token_metadata": {
        "tokens": ["Hello", ",", " world", ...],
        "positions": [0, 1, 2, ...],
        "token_ids": [15496, 11, 995, ...]
    },
    "token_paths": {
        "0": {
            "token_text": "Hello",
            "position": 0,
            "cluster_path": [0, 1, 2, 1],
            "path_length": 3.2,
            "cluster_changes": 3,
            "mobility_score": 0.8
        }
    },
    "cluster_labels": {...},
    "attention_data": {...},
    "cluster_metrics": {...}
}
```

### Saved File Structure
Saved analyses include comprehensive metadata:

```python
{
    "metadata": {
        "model_name": "gpt2-small",
        "input_text": "Original input text",
        "analysis_id": "unique_analysis_id",
        "timestamp": "2023-...",
        "version": 1,
        "data_format": "gpt2_analysis_v1"
    },
    "analysis_data": { ... },
    "model_info": {
        "num_tokens": 42,
        "num_layers": 12,
        "input_length": 256
    }
}
```

## Error Handling

The persistence system includes comprehensive error handling:

- **File I/O errors**: Gracefully handled with informative messages
- **Data validation**: Ensures data integrity before saving
- **Version conflicts**: Automatically handles version numbering
- **Cache corruption**: Falls back to disk storage if cache is corrupted

## Best Practices

1. **Use descriptive names**: When saving analyses, use meaningful names
2. **Regular cleanup**: Periodically clean up old cache files
3. **Export important results**: Export critical analyses for backup
4. **Session organization**: Use sessions to group related analyses
5. **Version management**: Keep track of analysis versions for reproducibility

## Troubleshooting

### Common Issues

**"Analysis not found" error**
- Check that the analysis_id is correct
- Verify the analysis hasn't been deleted
- Check file permissions in the storage directory

**Cache performance issues**
- Increase cache_ttl for frequently accessed analyses
- Clean up old cache files regularly
- Consider using memory_only=True for temporary analyses

**Storage space concerns**
- Reduce max_versions for automatic cleanup
- Export and delete old analyses
- Use compression for long-term storage

## Examples

See `examples/gpt2_persistence_demo.py` for complete working examples of all persistence features.

## API Reference

### GPT2AnalysisPersistence Class

**Methods:**
- `save_analysis_results()`: Save complete analysis results
- `load_analysis_results()`: Load saved analysis results  
- `save_visualization_state()`: Save visualization configuration
- `load_visualization_state()`: Load visualization configuration
- `create_session()`: Create analysis session
- `load_session()`: Load analysis session
- `export_analysis()`: Export analysis in various formats
- `list_analyses()`: List all saved analyses
- `cleanup_cache()`: Clean up old cache files

### Convenience Functions

- `save_gpt2_analysis()`: Quick save function
- `load_gpt2_analysis()`: Quick load function

For detailed API documentation, see the docstrings in the source code.