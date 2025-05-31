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

2. **Demo Model Training Infrastructure** (separate from Concept MRI)
   - `ModelTrainer` class - reuses existing data loaders
   - `OptimizableFFNet` - flexible architecture (any number of layers)
   - `model_presets.py` - configurations for different variants:
     - optimal (via hyperparameter search)
     - bottleneck (concept compression)
     - overfit (fragmentation demo)
     - underfit (poor concepts)
     - unstable (erratic training)
     - fragmented (redundant concepts)
   - Optuna integration for hyperparameter optimization
   - CLI script: `train_demo_models.py`

3. **Integration Updates**
   - Enhanced `model_interface.py` to load demo models
   - Auto-discovery of models in `concept_mri/demos/`
   - Metadata loading support

### 🚧 Next Steps (Priority Order)

1. **Train Demo Models** (Medium Priority)
   ```bash
   cd scripts/prepare_demo_models
   python train_demo_models.py --dataset titanic --variant optimal
   python train_demo_models.py --dataset titanic --all  # All variants
   ```

2. **Add Demo Model Dropdown** to FF tab (Medium Priority)
   - Add to `ff_networks.py`
   - Use `ModelInterface.list_demo_models()`

3. **Create Validation Script** (Medium Priority)
   - Test all models load correctly
   - Verify activations can be extracted

4. **Integration Testing** (High Priority)
   - Full pipeline: train → save → load → analyze

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
- **Date**: May 30, 2025
- **Last Task**: Completed demo model training infrastructure
- **Next Task**: Train actual demo models or add UI dropdown

---

*Note: This document should be updated at the end of each work session to maintain continuity.*