# Implementation Plan for Archetypal Path Analysis (Updated)

This document outlines the updated plan to implement the Archetypal Path Analysis (APA) framework described in "Foundations of Archetypal Path Analysis: Toward a Principled Geometry for Cluster-Based Interpretability."

## 1. Core Components to Implement

### 1.1 ✅ Unique Cluster IDs
- **Status:** Completed
- Unique ID system that preserves layer information
- Global identifiers for all clusters across layers
- Functions for cluster ID mapping and serialization

### 1.2 ✅ Centroid Similarity Matrix Calculation
- **Status:** Completed
- Compute similarity between cluster centroids across layers
- Normalize similarity matrices for comparison
- Calculate fragmentation scores for paths
- Identify similarity-convergent paths
- Support for cosine similarity and Euclidean distance

## 2. Dashboard Integration for Similarity Analysis

### 2.1 Similarity Network Visualization
- **Approach**: Create interactive network graph using NetworkX and Plotly
- **Integration**: Add as new dashboard tab using the similarity matrix data
- **Components**:
  - Clusters as nodes, similarities as weighted edges
  - Slider for similarity threshold filtering
  - Color coding by layer
  - Click interaction to show cluster details

### 2.2 ✅ Path Fragmentation View
- **Status**: Completed
- **Approach**: Enhance existing path list with fragmentation metrics
- **Integration**: Added a dedicated tab with comprehensive fragmentation analysis
- **Components**:
  - Fragmentation score histogram showing distribution
  - Interactive path table with fragmentation scores (sortable)
  - Color-coding for high/medium/low fragmentation
  - Filter option to show most/least fragmented paths
  - Detailed path view with demographics and convergence information
  - Visual representation of path structure and layer-to-layer fragmentation

### 2.3 Convergent Path Explorer
- **Approach**: Extend path visualization to highlight similarity convergence
- **Integration**: Add visual indicators to existing path view
- **Components**:
  - Highlight connections between similar clusters across non-adjacent layers
  - Add filter option to show only convergent paths
  - Visual indicators where concepts reappear in later layers

### 2.4 Layer Similarity Heatmap
- **Approach**: Create compact heatmap visualization of layer-layer similarities
- **Integration**: Add to the overview panel or metrics tab
- **Components**:
  - Square heatmap with layers on both axes
  - Color intensity showing average similarity
  - Click to drill down to specific layer pairs

### 2.5 Cluster Similarity Comparison & Concept Preservation
- **Approach**: Add detailed view when selecting similar clusters
- **Integration**: Show when selecting cluster pairs in network view
- **Components**:
  - Side-by-side visualization of cluster features
  - Summary metrics panel with concept preservation statistics
  - Comparison of member distributions

## 3. LLM Integration

### 3.1 xAI (Grok) API Integration
- **Status:** Planned
- Connect to xAI API for automatic path analysis
- Generate prompts for cluster labeling and path interpretation
- Implement caching for API responses

### 3.2 Automated Cluster Labeling
- **Status:** Planned
- Use LLM to generate human-readable labels for clusters
- Extract key features that define each cluster
- Keep track of concept evolution across layers

### 3.3 Path Narrative Generation
- **Status:** Planned
- Create detailed narratives explaining path meaning
- Highlight demographic characteristics of path members
- Identify potential biases or fairness issues

## 4. Experiment Scripts

### 4.1 Explainable Threshold Similarity (ETS)
- **Status:** Planned
- Implement ETS clustering algorithm
- Add ETS as an alternative clustering method
- Compare ETS to k-means for interpretability

### 4.2 Transition Matrix Analysis
- **Status:** Planned
- Compute transition matrices between layers
- Analyze transition entropy and sparsity
- Extend to multi-step transitions

### 4.3 Metrics Comparison
- **Status:** Planned
- Implement comprehensive metrics comparison
- Visualize different fragmentation metrics
- Evaluate correlation between metrics

## 5. Timeline

1. **Week 1: Dashboard Integration for Similarity Analysis**
   - Complete all five dashboard enhancement components
   - Ensure smooth integration with existing visualization

2. **Week 2: LLM Integration**
   - Implement xAI API connection
   - Create automated cluster and path labeling
   - Test with different prompt strategies

3. **Week 3: Experiment Scripts**
   - Implement ETS and transition matrix analysis
   - Add metrics comparison framework
   - Create comprehensive documentation

4. **Week 4: Testing & Refinement**
   - Write unit tests for all components
   - Optimize performance for large datasets
   - Prepare final demonstration

## 6. Requirements

Additional dependencies:
```
scipy>=1.7.0
scikit-learn>=1.0.0
networkx>=2.6.0
plotly>=5.3.0
```

Optional LLM integration:
```
xai-client>=1.0.0  # or appropriate xAI library
```