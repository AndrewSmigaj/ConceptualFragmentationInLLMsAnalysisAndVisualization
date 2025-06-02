# LLM Analysis API Reference

## Overview

The `ClusterAnalysis` class in `concept_fragmentation.llm.analysis` provides a high-level API for analyzing neural network activation clusters using Large Language Models.

## Class: ClusterAnalysis

### Constructor

```python
ClusterAnalysis(
    provider: str = "openai",
    model: str = "gpt-4", 
    api_key: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    cache_ttl: Optional[int] = None,
    memory_only_cache: bool = False,
    save_interval: int = 10,
    optimize_prompts: bool = False,
    optimization_level: int = 1,
    debug: bool = False
)
```

**Parameters:**
- `provider`: LLM provider ("openai", "claude", "gemini", "grok")
- `model`: Model to use (provider-specific)
- `api_key`: API key (if None, loads from environment)
- `use_cache`: Enable response caching
- `cache_dir`: Directory for cache files
- `cache_ttl`: Cache time-to-live in seconds
- `memory_only_cache`: Don't persist cache to disk
- `save_interval`: Save cache every N items
- `optimize_prompts`: Enable prompt optimization
- `optimization_level`: Optimization aggressiveness (1-3)
- `debug`: Enable debug output

### Methods

#### generate_path_narratives_sync

```python
generate_path_narratives_sync(
    paths: Dict[int, List[str]],
    cluster_labels: Dict[str, str],
    convergent_points: Optional[Dict[int, List[Tuple[str, str, float]]]] = None,
    fragmentation_scores: Optional[Dict[int, float]] = None,
    path_demographic_info: Optional[Dict[int, Dict[str, Any]]] = None,
    per_cluster_stats_for_paths: Optional[Dict[int, Dict[str, str]]] = None,
    analysis_categories: Optional[List[str]] = None
) -> str
```

Generate comprehensive analysis of multiple paths in a single LLM call.

**Parameters:**
- `paths`: Dictionary mapping path IDs to lists of cluster IDs
- `cluster_labels`: Dictionary mapping cluster IDs to human-readable labels
- `convergent_points`: Optional convergent points in paths
- `fragmentation_scores`: Optional path fragmentation scores (0-1)
- `path_demographic_info`: Optional demographic/feature distributions per path
- `per_cluster_stats_for_paths`: Optional cluster statistics
- `analysis_categories`: Analysis types to include (default: ['interpretation', 'bias'])

**Returns:**
- Comprehensive analysis text covering all requested categories

**Example:**
```python
analyzer = ClusterAnalysis(provider="openai", api_key="sk-...")

# Analyze paths with bias detection
result = analyzer.generate_path_narratives_sync(
    paths={
        0: ["L0_C1", "L1_C2", "L2_C1"],
        1: ["L0_C2", "L1_C1", "L2_C3"]
    },
    cluster_labels={
        "L0_C1": "Low risk",
        "L0_C2": "High risk",
        "L1_C1": "Improving",
        "L1_C2": "Stable",
        "L2_C1": "Good outcome",
        "L2_C3": "Poor outcome"
    },
    path_demographic_info={
        0: {"age": {"<50": 0.8, ">=50": 0.2}},
        1: {"age": {"<50": 0.2, ">=50": 0.8}}
    },
    analysis_categories=['interpretation', 'bias']
)
```

#### label_clusters_sync

```python
label_clusters_sync(
    cluster_profiles: Dict[str, str],
    max_concurrency: int = 5
) -> Dict[str, str]
```

Generate human-readable labels for multiple clusters.

**Parameters:**
- `cluster_profiles`: Dictionary mapping cluster IDs to textual profiles
- `max_concurrency`: Maximum concurrent API requests

**Returns:**
- Dictionary mapping cluster IDs to generated labels

**Example:**
```python
labels = analyzer.label_clusters_sync({
    "L0_C1": "Age: 45±12, Gender: 60% female, BP: 120/80±10",
    "L0_C2": "Age: 65±8, Gender: 70% male, BP: 140/90±15"
})
# Returns: {"L0_C1": "Young healthy adults", "L0_C2": "Elderly hypertensive males"}
```

#### generate_with_cache

```python
generate_with_cache(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    **kwargs
) -> LLMResponse
```

Low-level method for cached LLM generation.

**Parameters:**
- `prompt`: The prompt to send to the LLM
- `temperature`: Randomness control (0-1)
- `max_tokens`: Maximum tokens to generate
- `**kwargs`: Additional API parameters

**Returns:**
- `LLMResponse` object with text and metadata

#### get_cache_stats

```python
get_cache_stats() -> Dict[str, Any]
```

Get cache statistics.

**Returns:**
- Dictionary with cache hit rate, size, and other stats

#### clear_cache

```python
clear_cache(force_save: bool = True) -> None
```

Clear the cache.

**Parameters:**
- `force_save`: Save empty cache to disk

#### close

```python
close() -> None
```

Properly close resources and save cache.

## Analysis Categories

### interpretation
Default analysis focusing on:
- Conceptual paths through the network
- How concepts transform across layers
- Decision-making patterns

### bias
Bias detection analysis including:
- Demographic routing differences
- Segregation patterns
- Statistical anomalies
- Fairness concerns

### efficiency
Model efficiency analysis:
- Redundant paths
- Compression opportunities
- Optimization suggestions

### robustness
Stability analysis:
- Path consistency
- Vulnerability assessment
- Adversarial robustness

## Complete Example

```python
from concept_fragmentation.llm.analysis import ClusterAnalysis
import asyncio

# Initialize analyzer
analyzer = ClusterAnalysis(
    provider="openai",
    api_key="sk-...",
    model="gpt-4",
    use_cache=True,
    cache_dir="./llm_cache",
    debug=True
)

# Prepare comprehensive data
paths = {
    0: ["L0_C1", "L1_C2", "L2_C1", "L3_C1"],
    1: ["L0_C2", "L1_C1", "L2_C3", "L3_C2"],
    2: ["L0_C1", "L1_C3", "L2_C2", "L3_C3"]
}

cluster_labels = {
    "L0_C1": "Normal baseline metrics",
    "L0_C2": "Elevated risk indicators",
    "L1_C1": "Improving trajectory",
    "L1_C2": "Stable progression",
    "L1_C3": "Mixed signals",
    "L2_C1": "Positive trends",
    "L2_C2": "Neutral state",
    "L2_C3": "Declining metrics",
    "L3_C1": "Healthy outcome",
    "L3_C2": "At-risk outcome", 
    "L3_C3": "Critical outcome"
}

path_demographic_info = {
    0: {
        "age_group": {"18-35": 0.6, "36-50": 0.3, "51+": 0.1},
        "gender": {"male": 0.4, "female": 0.6},
        "ethnicity": {"white": 0.5, "black": 0.2, "asian": 0.2, "other": 0.1}
    },
    1: {
        "age_group": {"18-35": 0.1, "36-50": 0.3, "51+": 0.6},
        "gender": {"male": 0.8, "female": 0.2},
        "ethnicity": {"white": 0.6, "black": 0.1, "asian": 0.1, "other": 0.2}
    },
    2: {
        "age_group": {"18-35": 0.3, "36-50": 0.4, "51+": 0.3},
        "gender": {"male": 0.5, "female": 0.5},
        "ethnicity": {"white": 0.4, "black": 0.3, "asian": 0.2, "other": 0.1}
    }
}

fragmentation_scores = {
    0: 0.12,  # Very stable
    1: 0.45,  # Moderately fragmented
    2: 0.78   # Highly fragmented
}

# Run comprehensive analysis
try:
    result = analyzer.generate_path_narratives_sync(
        paths=paths,
        cluster_labels=cluster_labels,
        path_demographic_info=path_demographic_info,
        fragmentation_scores=fragmentation_scores,
        analysis_categories=['interpretation', 'bias', 'robustness']
    )
    
    print("Analysis Results:")
    print(result)
    
    # Get cache statistics
    stats = analyzer.get_cache_stats()
    print(f"\nCache stats: {stats}")
    
finally:
    # Always close to save cache
    analyzer.close()
```

## Error Handling

```python
from concept_fragmentation.llm.analysis import ClusterAnalysis

try:
    analyzer = ClusterAnalysis(provider="invalid")
except ValueError as e:
    print(f"Invalid provider: {e}")

try:
    analyzer = ClusterAnalysis(provider="openai", api_key=None)
except ValueError as e:
    print(f"Missing API key: {e}")
```

## Performance Considerations

1. **Batching**: The comprehensive analysis processes all paths in one call, reducing API costs
2. **Caching**: Enable caching to avoid redundant API calls
3. **Path Limits**: For large models, limit to 20-50 archetypal paths
4. **Token Limits**: Monitor token usage, especially with many paths

## Migration from Old API

### Old (per-path) API:
```python
# DON'T DO THIS - Makes many API calls
narratives = {}
for path_id, path in paths.items():
    narrative = analyzer.generate_path_narrative(path, labels)
    narratives[path_id] = narrative
```

### New (comprehensive) API:
```python
# DO THIS - Single API call
result = analyzer.generate_path_narratives_sync(
    paths=paths,
    cluster_labels=labels,
    analysis_categories=['interpretation', 'bias']
)
```

## Thread Safety

The `ClusterAnalysis` class is thread-safe for read operations but not for write operations. If using in a multi-threaded environment:

1. Create separate instances per thread, OR
2. Use a single instance with external locking, OR
3. Use the async methods with proper async coordination