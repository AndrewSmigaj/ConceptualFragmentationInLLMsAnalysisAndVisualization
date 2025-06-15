# Network Explorer Design Document

## Overview
The Network Explorer is a unified interface for exploring neural network behavior through Concept Trajectory Analysis (CTA). It provides multi-scale exploration from full network overview down to individual cluster/path details.

## UI Layout

```
┌────────────────────────────────────────────────────────────────────────────┐
│  [Model: example.pt]  [Dataset: data.csv]  [⚙️ Settings]  [📊 Export]       │
├────────────────────────────────────────────────────────────────────────────┤
│                           NETWORK OVERVIEW (15% height)                     │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Metrics Visualization                                               │  │
│  │  1.0 ┤ ━━━ Fragmentation  ╱╲    ╱╲                                 │  │
│  │      │ ┅┅┅ Cohesion      ╱  ╲__╱  ╲___╱╲___                       │  │
│  │  0.5 ┤ ⋯⋯⋯ Entropy   ___╱              ╲   ╲___                    │  │
│  │  0.0 ┼───┬────┬────┬────┬────┬────┬────┬────┬────┬────┬─────     │  │
│  │      L0  L1   L2   L3   L4   L5   L6   L7   L8   L9   L10        │  │
│  │                                                                     │  │
│  │  [Early Window]    [Middle Window]    [Late Window]    [Custom]    │  │
│  │    L0-L3 ●           L4-L7 ○           L8-L10 ○        Define...   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Metrics: ☑ Fragmentation ☑ Cohesion ☑ Entropy ☐ Path Diversity ☐ Stability│
├─────────────────────┬───────────────────────┬──────────────────────────────┤
│                     │                        │                             │
│  ARCHETYPAL PATHS   │   WINDOW VISUALIZATION │  DETAILS & ANALYSIS         │
│  (25% width)        │   (50% width)          │  (25% width)                │
│                     │                        │                             │
│  Window: Early (L0-L3)                       │  ┌──────────────────────┐   │
│  ┌─────────────┐    │  ┌──────────────────┐ │  │ Selected: Path #1    │   │
│  │ Path 1 (45%)│◀───┼──│                  │ │  ├──────────────────────┤   │
│  │ L0_C1→L1_C0 │    │  │  Sankey Diagram  │ │  │ Frequency: 45%       │   │
│  │ →L2_C1→L3_C2│    │  │  ================│ │  │ Samples: 450         │   │
│  │ ➤ Stable    │    │  │                  │ │  │ Fragmentation: 0.23  │   │
│  └─────────────┘    │  │  [Sankey|Traj|3D]│ │  │                      │   │
│                     │  │                  │ │  │ Demographics:        │   │
│  ┌─────────────┐    │  │  Highlighted:    │ │  │ • Age: 45-65 (72%)  │   │
│  │ Path 2 (23%)│    │  │  - Path 1        │ │  │ • Gender: F (68%)   │   │
│  │ L0_C0→L1_C1 │    │  │                  │ │  │                      │   │
│  │ →L2_C0→L3_C1│    │  └──────────────────┘ │  └──────────────────────┘   │
│  │ ⚠ Divergent │    │                        │                             │
│  └─────────────┘    │  Controls:             │  ┌──────────────────────┐   │
│                     │  Color: [Cluster ▼]    │  │ 🤖 LLM Analysis      │   │
│  Filters:           │  Layout: [Standard ▼]  │  ├──────────────────────┤   │
│  [🔍 Search...]     │  Highlight: [Path 1 ▼] │  │ ☑ Interpretation     │   │
│  Frequency: [>10% ▼]│  ☐ Show all paths     │  │ ☑ Bias Detection    │   │
│  Pattern: [All ▼]   │  ☐ Normalize          │  │ ☐ Efficiency         │   │
│                     │  ☐ Compare windows    │  │ ☐ Robustness         │   │
│  Coverage: 89%      │                        │  ├──────────────────────┤   │
│  Paths shown: 10/127│                        │  │ ⚠️ Bias Alert:       │   │
└─────────────────────┴───────────────────────┴──────────────────────────────┘
```

## Component Hierarchy

```
NetworkExplorer
├── NetworkOverview
│   ├── MetricsChart
│   │   ├── Multi-line plot (Plotly)
│   │   ├── Interactive window regions
│   │   └── Metric value tooltips
│   ├── WindowSelector
│   │   ├── Predefined windows (Early/Middle/Late)
│   │   ├── Custom window definition
│   │   └── Window comparison mode
│   └── MetricsSelector
│       └── Checkboxes for each metric
│
├── ExplorerWorkspace
│   ├── ArchetypalPathsPanel
│   │   ├── PathList
│   │   │   ├── PathCard (interactive)
│   │   │   ├── Stability indicator
│   │   │   └── Quick stats
│   │   ├── PathFilters
│   │   │   ├── Frequency threshold
│   │   │   ├── Pattern matching
│   │   │   └── Cluster inclusion
│   │   └── PathStatistics
│   │       ├── Coverage percentage
│   │       └── Distribution chart
│   │
│   ├── VisualizationPanel
│   │   ├── VisualizationSwitcher
│   │   │   ├── Sankey (default)
│   │   │   ├── Trajectory (UMAP)
│   │   │   └── 3D Network (future)
│   │   ├── VisualizationContainer
│   │   │   └── Active visualization component
│   │   └── VisualizationControls
│   │       ├── Color scheme selector
│   │       ├── Layout options
│   │       ├── Highlight controls
│   │       └── Comparison toggles
│   │
│   └── DetailsPanel
│       ├── EntityCard (Dynamic)
│       │   ├── PathCard
│       │   ├── ClusterCard
│       │   └── ComparisonCard
│       └── AnalysisPanel
│           ├── AnalysisTypeSelector
│           ├── AnalysisResults
│           └── AlertsDisplay
│
└── SelectionManager
    ├── Selection state management
    ├── Cross-component communication
    └── History tracking
```

## Data Flow

### 1. Window Selection Flow
```
User clicks window in NetworkOverview
  → window-selection-store updated
  → ArchetypalPathsPanel filters paths to selected window
  → VisualizationPanel updates to show only window's layers
  → AnalysisPanel refreshes with window-specific insights
  → MetricsChart highlights selected window region
```

### 2. Entity Selection Flow
```
User clicks path/cluster
  → entity-selection-store updated
  → EntityCard shows detailed information
  → Visualization highlights selected entity
  → Related entities are visually connected
  → Analysis shows entity-specific insights
```

### 3. Analysis Update Flow
```
User toggles analysis type
  → Triggers LLM analysis if needed
  → Updates AnalysisPanel content
  → Highlights relevant paths/clusters
  → Shows alerts for findings
```

## Key Features

### 1. Multi-Metric Visualization
- Overlay multiple metrics on single chart
- Different line styles/colors per metric
- Interactive legend with current values
- Correlation indicators between metrics

### 2. Path Evolution Tracking
- Path stability indicators (stable/divergent/convergent)
- Cross-window path comparison
- Ghost paths from other windows
- Path genealogy visualization

### 3. Smart Interactions
- **Click**: Select and show details
- **Double-click**: Isolate/focus
- **Shift+click**: Multi-select for comparison
- **Hover**: Quick tooltips with key info
- **Drag**: Create custom window
- **Right-click**: Context menu

### 4. Comparative Analysis
- Side-by-side window comparison
- Baseline model comparison
- Temporal analysis (epochs)
- A/B testing support

### 5. Export & Annotation
- Export current view as image/PDF
- Export selected data as CSV/JSON
- Add annotations to entities
- Save/load analysis sessions
- Findings log with screenshots

## Implementation Priority

### Phase 1: Core Explorer
1. NetworkOverview with basic metrics
2. ArchetypalPathsPanel with path list
3. Integration of existing visualizations
4. Basic EntityCard implementation

### Phase 2: Enhanced Interaction
1. SelectionManager with cross-highlighting
2. Multi-metric overlay
3. Window comparison mode
4. Enhanced entity cards

### Phase 3: Advanced Features
1. Path evolution tracking
2. Annotation system
3. Export functionality
4. Guided exploration mode

## Technical Considerations

### State Management
- Use Dash stores for persistent state
- SelectionManager as central coordinator
- Optimistic updates for responsiveness

### Performance
- Virtualized lists for many paths
- Debounced updates for smooth interaction
- Progressive data loading
- Caching of expensive computations

### Accessibility
- Keyboard navigation support
- Screen reader descriptions
- High contrast mode
- Configurable font sizes

## Success Metrics
1. Time to first insight < 30 seconds
2. All key information visible without scrolling
3. < 3 clicks to any detail
4. Smooth interactions (60 fps)
5. Clear visual hierarchy