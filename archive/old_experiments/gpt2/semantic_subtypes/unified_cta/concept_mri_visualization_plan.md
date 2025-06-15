# Concept MRI: Visualization Plan for GPT-2 Semantic Organization

## Overview
Create an interpretable "Concept MRI" visualization that reveals how words flow through GPT-2's layers, focusing on the discovered grammatical organization and massive convergence patterns.

## Design Principles
1. **Build on existing infrastructure** - Use existing Sankey generators and visualization components
2. **Focus on interpretability** - Make patterns understandable to non-experts
3. **Highlight key findings** - Grammatical > semantic organization, massive convergence
4. **Interactive exploration** - Allow users to explore specific words and paths

## Core Components

### 1. Three-Window Sankey Visualization
**Use existing**: `visualization/generate_sankey_diagram.py` as base
**Enhance with**:
- Side-by-side layout for Early/Middle/Late windows
- Color by grammatical category (nouns=blue, adjectives=red, adverbs=green)
- Proportional path thickness
- Hover tooltips with example words

**Data preparation**:
- Already have sankey JSON files in `windowed_analysis/`
- Need to add grammatical category colors

### 2. Complete Path Analysis Table
**For each window, show ALL paths** (not just archetypal):

```
Early Window (L0-L3) - 19 paths
┌────────────────────────────────────────────┐
│ #1: [L0_C1→L1_C1→L2_C1→L3_C1]            │
│ 154 words (27.21%) - Concrete Objects     │
│ Examples: mouse, window, clock, computer   │
│ Stability: 100% (perfectly stable)        │
├────────────────────────────────────────────┤
│ #2: [L0_C0→L1_C1→L2_C1→L3_C1]            │
│ 131 words (23.14%) - Living Entities      │
│ Examples: cat, dog, bird, fish, horse      │
│ Stability: 66.7% (one transition)         │
└────────────────────────────────────────────┘
... all 19 paths ...
```

### 3. Key Insights Dashboard

**Convergence Visualization**:
```
  19 paths ──73.7% reduction──> 5 paths ──20% reduction──> 4 paths
  (Early)                        (Middle)                    (Late)
```

**Dominant Path Analysis**:
- Show the 72.79% path with ALL its words grouped by type
- Highlight that animals + objects travel together

**Grammatical vs Semantic Organization**:
- Split dominant path words by original semantic category
- Show they all converge despite different meanings

### 4. Interactive Word Explorer
**Simple search interface**:
- Input: Type any word
- Output: 
  - Full trajectory [L0_C?→L1_C?→...→L11_C?]
  - Which window paths it follows
  - Other words with identical trajectory

### 5. Layer-by-Layer Cluster Inspector
**Click any cluster label** (e.g., L4_C1) to see:
- All words in that cluster
- Breakdown by grammatical/semantic category
- Connections to previous/next layers

## Implementation Plan

### Phase 1: Data Preparation
1. Load existing data from `llm_analysis_data/all_paths_unique_labels.json`
2. Add grammatical categories to each word (noun/verb/adjective/adverb)
3. Prepare formatted data for visualizations

### Phase 2: Core Visualizations
1. Adapt `generate_sankey_diagram.py` for three-window display
2. Create path analysis tables with complete listings
3. Build convergence flow diagram

### Phase 3: Interactive Features
1. Word search functionality
2. Cluster inspection tooltips
3. Path highlighting on hover

### Phase 4: Insights & Interpretation
1. Add interpretation text for each major pattern
2. Highlight surprising findings
3. Create summary statistics

## Technical Approach

### Use Existing:
- `visualization/generate_sankey_diagram.py` - Sankey generation
- `visualization/gpt2_token_sankey.py` - Token flow visualization
- D3.js/Plotly from existing visualizations

### New Components:
- `concept_mri_dashboard.html` - Main integrated view
- `prepare_mri_data.py` - Format data for visualization
- `grammatical_tagger.py` - Add POS tags to words

### Data Flow:
```
all_paths_unique_labels.json
    ↓
prepare_mri_data.py (add grammatical tags)
    ↓
concept_mri_data.json
    ↓
concept_mri_dashboard.html (renders visualizations)
```

## Visual Design

### Color Scheme:
- **Nouns/Entities**: Blue gradient (#1f77b4 to #aec7e8)
- **Adjectives/Properties**: Red gradient (#ff7f0e to #ffbb78)
- **Adverbs/Modifiers**: Green gradient (#2ca02c to #98df8a)
- **Mixed/Ambiguous**: Gray (#7f7f7f)

### Layout:
```
┌─────────────────────────────────────────────┐
│          GPT-2 Concept MRI                  │
│    How Words Flow Through Neural Layers     │
├─────────────────────────────────────────────┤
│  [Early]      [Middle]       [Late]         │
│  Sankey       Sankey        Sankey          │
│  19 paths  →  5 paths    →  4 paths         │
├─────────────────────────────────────────────┤
│           Path Analysis Tables              │
│  [Early Paths] [Middle Paths] [Late Paths]  │
├─────────────────────────────────────────────┤
│          Key Insights & Patterns            │
│  • Grammatical > Semantic Organization      │
│  • 72.79% Convergence to Single Path        │
│  • Animals + Objects Travel Together        │
└─────────────────────────────────────────────┘
```

## Key Messages to Convey

1. **"GPT-2 sorts by grammar, not meaning"** - The primary organization is grammatical
2. **"Massive convergence creates efficiency"** - 566 words → 5 paths
3. **"Universal noun processor"** - One pathway handles all concrete things
4. **"Specialized paths for special cases"** - Rare paths handle ambiguous words

## Success Criteria

1. **Clarity**: Non-experts can understand the patterns
2. **Completeness**: Shows ALL paths, not just common ones
3. **Interactivity**: Users can explore specific words/paths
4. **Impact**: Reveals surprising organizational principles

## Files to Create

1. `prepare_mri_data.py` - Data preparation script
2. `concept_mri_dashboard.html` - Main visualization
3. `mri_styles.css` - Clean, scientific styling
4. `mri_interactions.js` - Interactive features
5. `grammatical_categories.json` - Word categorizations