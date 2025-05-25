# Concept MRI Visualization: TODO List

## Phase 1: Data Preparation ⏳
- [ ] Create `prepare_mri_data.py` script that:
  - [ ] Loads `all_paths_unique_labels.json`
  - [ ] Adds grammatical categories to each word (use simple heuristics or NLTK)
  - [ ] Formats data for Sankey diagrams (nodes + links format)
  - [ ] Outputs `concept_mri_data.json` with all visualization data

- [ ] Create `grammatical_categories.json`:
  - [ ] Categorize all 566 words as noun/verb/adjective/adverb/other
  - [ ] Can use simple suffix rules + manual verification for accuracy

## Phase 2: Sankey Visualizations 📊
- [ ] Adapt existing `generate_sankey_diagram.py`:
  - [ ] Modify to handle windowed data format
  - [ ] Add grammatical category coloring
  - [ ] Create three separate Sankey diagrams (early/middle/late)
  - [ ] Ensure proportional path thickness

- [ ] Generate static Sankey images:
  - [ ] `sankey_early_window.html`
  - [ ] `sankey_middle_window.html`
  - [ ] `sankey_late_window.html`

## Phase 3: Path Analysis Tables 📋
- [ ] Create path listing component:
  - [ ] Template for displaying all paths in a window
  - [ ] Include frequency, percentage, examples
  - [ ] Add stability indicators
  - [ ] Group by frequency (common/rare/singleton)

- [ ] Generate path tables for:
  - [ ] Early window (19 paths)
  - [ ] Middle window (5 paths)
  - [ ] Late window (4 paths)

## Phase 4: Main Dashboard 🎯
- [ ] Create `concept_mri_dashboard.html`:
  - [ ] Header with title and explanation
  - [ ] Three-column layout for Sankey diagrams
  - [ ] Tabbed interface for path analysis tables
  - [ ] Key insights section with findings

- [ ] Add CSS styling (`mri_styles.css`):
  - [ ] Clean, scientific appearance
  - [ ] Responsive layout
  - [ ] Print-friendly version

## Phase 5: Interactive Features ✨
- [ ] Implement word search:
  - [ ] Search box to find any word
  - [ ] Highlight word's path across all windows
  - [ ] Show similar words (same trajectory)

- [ ] Add cluster inspection:
  - [ ] Click cluster label to see all words
  - [ ] Tooltip with cluster statistics
  - [ ] Link to connected clusters

- [ ] Path highlighting:
  - [ ] Hover to highlight full path
  - [ ] Click to lock highlighting
  - [ ] Show path statistics

## Phase 6: Insights & Interpretation 💡
- [ ] Add interpretation panels:
  - [ ] Explain convergence pattern
  - [ ] Highlight grammatical organization
  - [ ] Note surprising findings

- [ ] Create summary statistics:
  - [ ] Convergence ratios
  - [ ] Path diversity metrics
  - [ ] Grammatical category distribution

## Phase 7: Export & Sharing 📤
- [ ] Add export functionality:
  - [ ] "Download as PNG" for visualizations
  - [ ] "Export data as CSV" for paths
  - [ ] "Generate PDF report" option

- [ ] Create standalone version:
  - [ ] Bundle all assets
  - [ ] Work offline
  - [ ] Easy to share

## Quick Start Commands 🚀

```bash
# 1. Prepare data
python prepare_mri_data.py

# 2. Generate Sankey diagrams
python generate_sankey_diagram.py --window early
python generate_sankey_diagram.py --window middle
python generate_sankey_diagram.py --window late

# 3. Open dashboard
open concept_mri_dashboard.html
```

## Priority Order 🎯

1. **High Priority** (Core functionality):
   - Data preparation script
   - Three Sankey diagrams
   - Path analysis tables
   - Basic dashboard layout

2. **Medium Priority** (Enhanced usability):
   - Word search
   - Interactive highlighting
   - Interpretation panels

3. **Low Priority** (Nice to have):
   - Export functionality
   - Advanced filtering
   - Animated transitions

## Time Estimates ⏰

- Phase 1: 2-3 hours
- Phase 2: 2-3 hours
- Phase 3: 1-2 hours
- Phase 4: 2-3 hours
- Phase 5: 3-4 hours
- Phase 6: 1-2 hours
- Phase 7: 2-3 hours

**Total: 13-20 hours**

## Next Step

Start with `prepare_mri_data.py` to get the data ready for visualization!