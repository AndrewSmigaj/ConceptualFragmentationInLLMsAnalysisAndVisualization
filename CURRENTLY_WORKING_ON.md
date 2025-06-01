# Currently Working On - Concept MRI Development

## 🎯 Current Focus: Building Concept MRI Tool

### Overview
We are developing **Concept MRI**, a web-based tool for analyzing and auditing neural networks to understand how they organize concepts internally. This tool **does NOT train models** - it only analyzes pre-trained models.

### Key Context
- **Main Tool Location**: `/concept_mri/`
- **Training Scripts** (separate): `/scripts/prepare_demo_models/`
- **Purpose**: Visualize concept fragmentation in neural networks
- **Status**: Core infrastructure complete, need to train demo models

## 📋 Current Task Status

### ✅ Completed Infrastructure
1. **Concept MRI App Structure**
   - Dash-based web application (`concept_mri/app.py`)
   - Feedforward Networks tab (`concept_mri/tabs/ff_networks.py`)
   - Model upload/dataset upload components
   - Clustering configuration panel
   - Sankey diagram visualization wrapper

2. **Component Verification** (Session: May 31, 2025)
   - Fixed all import issues
   - Installed missing dependencies
   - All components now load and instantiate correctly
   - Ready for enhancement

3. **Demo Model Training Infrastructure** (separate from Concept MRI)
   - `ModelTrainer` class - reuses existing data loaders
   - `OptimizableFFNet` - flexible architecture (any number of layers)
   - Multiple variants configured
   - CLI script: `train_demo_models.py`

### 🔧 Currently Working On: Extending LLM Analysis for Bias Detection

**What**: Refactoring the LLM analysis system to make comprehensive analysis across all archetypal paths in a single call, with support for multiple analysis categories including bias detection.

**Why**: 
- Single API call is more efficient and allows LLM to see patterns across paths
- Bias detection requires cross-path analysis to identify systematic patterns
- Current system makes individual calls per path, limiting pattern detection

**What's Been Done**:
- Modified `concept_fragmentation/llm/analysis.py`
- Changed `generate_path_narratives` to return a single comprehensive analysis string
- Added `analysis_categories` parameter (defaults to `['interpretation', 'bias']`)
- Removed backwards compatibility - doing it right this time
- Bias analysis looks for:
  - Systematic demographic routing differences
  - Unexpected segregation patterns  
  - Statistical anomalies across paths
  - Potential unfair treatment patterns

**Implementation Status**:
- ✅ Fully refactored `generate_path_narratives` method
- ✅ Removed old branching logic
- ✅ Updated synchronous wrapper
- ✅ Successfully tested with real API calls
- ✅ Bias detection working correctly

**Files Modified**:
- `concept_fragmentation/llm/analysis.py` - Main changes to ClusterAnalysis class
- `local_config.py` - Created for API keys (gitignored)
- `local_config.py.example` - Template for others
- `test_llm_comprehensive.py` - Test script demonstrating new API
- `CLAUDE.md` - Created with environment instructions

**Test Results**:
- LLM successfully detected gender bias in heart disease paths
- Identified age-related routing patterns
- Found ethnicity correlations with disease classification
- Analysis provided actionable insights

**Note**: Layer Window Manager was completed in previous session

### 🚧 Upcoming Tasks (Priority Order)

1. **Integrate with Concept MRI** (High Priority - NEXT)
   - Connect the new comprehensive analysis API to Concept MRI
   - Update UI to display comprehensive analysis results
   - Add analysis category selection to UI
   - Create BiasAuditor component that uses the new API

2. **Create Usage Documentation** (Medium Priority)
   - Document the new comprehensive analysis API
   - Create examples for different analysis categories
   - Show how to interpret bias detection results

3. **Train Demo Models** (Medium Priority)
   ```bash
   cd scripts/prepare_demo_models
   python train_demo_models.py --dataset titanic --variant optimal
   ```

4. **Document New LLM Analysis API** (Medium Priority)
   - Update docstrings
   - Create usage examples
   - Document analysis categories and their outputs

## 🏗️ Architecture Notes

### Separation of Concerns
- **Concept MRI** (`/concept_mri/`): Analysis and visualization only
- **Training Scripts** (`/scripts/prepare_demo_models/`): Model creation only
- **Existing Infrastructure** (`/concept_fragmentation/`): Core library

### Key Design Decisions
1. Reuse all existing data loaders from `concept_fragmentation`
2. Models saved with architecture info for easy loading
3. Multiple variants to demonstrate different behaviors
4. Clean separation between training and analysis

### Model Save Format
```python
{
    'model_state_dict': state_dict,
    'input_size': int,
    'output_size': int,
    'architecture': [hidden_sizes],
    'activation': str,
    'dropout_rate': float
}
```

## 💡 Suggestions for Better Continuity

### 1. **Session Start Protocol**
When starting a new session, you could say:
```
"Continue working on Concept MRI - check CURRENTLY_WORKING_ON.md"
```

### 2. **Key Information to Include**
- Current task from TODO list
- Any blocking issues
- Specific files you were working on
- Any decisions that need to be made

### 3. **Session End Protocol**
Before ending, I can update this file with:
- Last completed task
- Next immediate task
- Any important context or decisions made

### 4. **Alternative Approaches**
- **Branch-based context**: Create a branch for each work session
- **Issue tracking**: Use GitHub issues for task management
- **Session logs**: Keep a `SESSION_LOGS/` directory with dated entries

### 5. **Quick Commands**
You could use shorthand commands like:
- "Continue Concept MRI" - I'll read this file and continue
- "Update context" - I'll update this file with current state
- "Show progress" - I'll summarize what's been done

## 🔗 Related Documents
- Implementation Plan: `concept_mri/IMPLEMENTATION_PLAN.md`
- Architecture: `ARCHITECTURE.yaml`
- Demo Models: `concept_mri/demos/README.md`
- Training Scripts: `scripts/prepare_demo_models/README.md`

## 📝 Last Updated
- **Date**: May 31, 2025
- **Last Task**: Completed LLM analysis refactoring - comprehensive multi-path analysis with bias detection now working
- **Next Task**: Integrate the new comprehensive analysis API with Concept MRI UI
- **Session Notes**: Successfully refactored to single API call, tested bias detection which correctly identified gender/age/ethnicity patterns in test data. Created CLAUDE.md with environment rules.

---

*Note: This document should be updated at the end of each work session to maintain continuity.*