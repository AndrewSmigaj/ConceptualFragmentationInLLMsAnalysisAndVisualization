# Comprehensive LLM Analysis Usage Guide

## Overview

The comprehensive LLM analysis feature allows you to analyze multiple neural network activation paths in a single API call, enabling better pattern detection and bias analysis across paths.

## Quick Start

```python
from concept_fragmentation.llm.analysis import ClusterAnalysis

# Initialize the analyzer
analyzer = ClusterAnalysis(
    provider="openai",
    api_key="your-api-key",
    model="gpt-4",
    use_cache=True
)

# Run comprehensive analysis
result = analyzer.generate_path_narratives_sync(
    paths=paths_dict,
    cluster_labels=labels_dict,
    analysis_categories=['interpretation', 'bias']
)
```

## Data Format

### Required Inputs

1. **paths** (Dict[int, List[str]]): Maps path IDs to sequences of cluster IDs
```python
paths = {
    0: ["L0_C1", "L1_C2", "L2_C1", "L3_C3"],
    1: ["L0_C2", "L1_C1", "L2_C3", "L3_C1"],
    2: ["L0_C1", "L1_C3", "L2_C2", "L3_C2"]
}
```

2. **cluster_labels** (Dict[str, str]): Maps cluster IDs to human-readable labels
```python
cluster_labels = {
    "L0_C1": "Normal baseline",
    "L0_C2": "Elevated markers",
    "L1_C1": "Stable progression",
    # ... etc
}
```

### Optional Inputs

3. **path_demographic_info** (Dict[int, Dict[str, Any]]): Feature distributions for each path
```python
path_demographic_info = {
    0: {
        "age": {"<40": 0.6, "40-60": 0.3, ">60": 0.1},
        "gender": {"male": 0.3, "female": 0.7}
    }
}
```

4. **fragmentation_scores** (Dict[int, float]): Path consistency scores (0-1)
```python
fragmentation_scores = {
    0: 0.15,  # Low fragmentation = stable path
    1: 0.45,  # Medium fragmentation
    2: 0.85   # High fragmentation = unstable path
}
```

## Analysis Categories

You can request different types of analysis by specifying categories:

### 1. Interpretation (Default)
Analyzes conceptual paths and transformations:
```python
analysis_categories=['interpretation']
```

**Output includes:**
- Main conceptual paths through the network
- How concepts transform across layers
- Decision-making patterns

### 2. Bias Detection
Identifies demographic routing patterns and potential discrimination:
```python
analysis_categories=['bias']
```

**Output includes:**
- Systematic demographic routing differences
- Unexpected segregation patterns
- Statistical anomalies indicating potential bias
- Actionable recommendations

### 3. Efficiency
Analyzes redundancy and compression opportunities:
```python
analysis_categories=['efficiency']
```

**Output includes:**
- Redundant or similar paths
- Consolidation opportunities
- Model compression suggestions

### 4. Robustness
Evaluates stability and vulnerability:
```python
analysis_categories=['robustness']
```

**Output includes:**
- Path stability analysis
- Vulnerability to adversarial inputs
- Consistency of representations

## Complete Example

```python
from concept_fragmentation.llm.analysis import ClusterAnalysis

# Initialize
analyzer = ClusterAnalysis(
    provider="openai",
    api_key="sk-...",
    model="gpt-4"
)

# Prepare data
paths = {
    0: ["L0_C1", "L1_C2", "L2_C1"],
    1: ["L0_C2", "L1_C1", "L2_C3"],
}

cluster_labels = {
    "L0_C1": "Low risk indicators",
    "L0_C2": "High risk indicators",
    "L1_C1": "Improving trends",
    "L1_C2": "Stable state",
    "L2_C1": "Positive outcome",
    "L2_C3": "Negative outcome"
}

path_demographic_info = {
    0: {
        "age_group": {"young": 0.7, "middle": 0.2, "senior": 0.1},
        "gender": {"male": 0.4, "female": 0.6}
    },
    1: {
        "age_group": {"young": 0.1, "middle": 0.3, "senior": 0.6},
        "gender": {"male": 0.8, "female": 0.2}
    }
}

# Run analysis
result = analyzer.generate_path_narratives_sync(
    paths=paths,
    cluster_labels=cluster_labels,
    path_demographic_info=path_demographic_info,
    analysis_categories=['interpretation', 'bias']
)

print(result)
```

## Interpreting Results

The analysis returns a comprehensive text report with sections for each requested category:

```
INTERPRETATION:
- Path 0 represents a low-risk progression that stabilizes and leads to positive outcomes
- Path 1 shows high-risk cases that despite improving trends still lead to negative outcomes
- The model appears to make early risk assessments that strongly influence final predictions

BIAS ANALYSIS:
- Significant age-based routing: seniors are 6x more likely to follow high-risk paths
- Gender disparity detected: males are overrepresented in negative outcome paths (80% vs 40% in data)
- Recommendation: Review early risk assessment features for age and gender bias
```

## Performance Tips

1. **Use caching**: Set `use_cache=True` to avoid redundant API calls
2. **Batch paths**: Analyze multiple paths together for better pattern detection
3. **Limit paths**: For large models, select top 20-50 archetypal paths
4. **Pre-filter**: Remove rare paths (frequency < 5) before analysis

## Integration with Concept MRI

In the Concept MRI web interface:

1. Complete clustering analysis
2. Navigate to the "LLM Analysis" tab
3. Select analysis categories with checkboxes
4. Click "Run LLM Analysis"
5. Results appear in formatted cards
6. Use "Export Analysis" to save results

## Troubleshooting

**No paths found**: Ensure clustering has completed and generated path data

**API errors**: Check API key is valid and has sufficient credits

**Empty results**: Verify data format matches expected structure

**Timeout errors**: Reduce number of paths or use smaller model (gpt-3.5-turbo)