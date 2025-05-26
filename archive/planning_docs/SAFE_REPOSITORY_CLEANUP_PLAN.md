# Safe Repository Cleanup Plan (Revised)

## Overview
Based on dependency analysis, here's a safer approach that won't break anything.

## Phase 1: Safe Moves (No Dependencies)

### 1. Create Archive Structure First
```bash
mkdir -p archive/planning_docs
mkdir -p archive/session_contexts  
mkdir -p archive/old_plans
mkdir -p arxiv_submission/archive
```

### 2. Move Planning/Session Documents (100% Safe)
These have NO code dependencies:

**Root level → `archive/planning_docs/`:**
- `CLEANUP_GUIDE.md`
- `GPT2_APA_PIVOT_EXPERIMENT.md`
- `GPT2_APA_PIVOT_EXPERIMENT_REVISED.md`
- `GPT2_INTEGRATION_PLAN.md`
- `GPT2_SEMANTIC_SUBTYPES_APA_EXECUTION_PLAN.md`
- `GPT2_SEMANTIC_SUBTYPES_EXPERIMENT_PLAN.md`
- `LLM_SETUP_SUMMARY.md`
- `README_LLM_TESTING.md`
- `REORGANIZATION_SUMMARY.md`
- `REPOSITORY_REORGANIZATION_PLAN.md`
- `SESSION_CONTEXT_PAPER_POLISHING.md`
- `SESSION_PROGRESS_SUMMARY.md` (if exists)

**ArXiv planning → `arxiv_submission/archive/`:**
- All the planning docs in arxiv_submission/

### 3. Move Backup Directory (Safe)
- `safe_backup_20250521_064116/` → `archive/backups/`

## Phase 2: Careful Moves (Check First)

### 1. Before Moving Experiments
- **COMMIT CHANGES FIRST**: 
  - `experiments/gpt2/semantic_subtypes/unified_cta/generate_gpt2_paper_figures.py`
  - `experiments/heart_disease/generate_labeled_heart_sankey.py`

### 2. Test Persistence Directory
- `test_persistence/` → `archive/test_outputs/`
- This appears to be test output only, safe to move

## Phase 3: Keep in Place (For Now)

### 1. Root Test Files
Keep these in root for now since they test core functionality:
- `test_*.py` files
- Can consolidate into `tests/` in a future cleanup

### 2. Experiment Structure
The experiments/ directory is well-organized, just need to:
- Archive old results in `experiments/*/archive/`
- Keep current code structure intact

## What NOT to Move

1. **Active Code**: All Python modules in use
2. **Config Files**: `config.py`, `requirements.txt`
3. **Batch/Script Files**: `run_dashboard.bat`, etc.
4. **Paper Files**: Current tex files and figures
5. **Virtual Environment**: `venv311/` (add to .gitignore if not already)

## Recommended .gitignore Additions

```
# Archives
archive/
*/archive/

# Virtual environments
venv*/
env/
ENV/

# Test outputs
test_persistence/
test_outputs/

# Cache
cache/
*/__pycache__/
*.pyc

# Temporary files
*.tmp
*.bak
*~
```

## Benefits of This Approach

1. **Zero risk of breaking imports**: Only moving documentation
2. **Preserves git history**: Moving, not deleting
3. **Cleaner root**: Removes 12+ planning documents
4. **Easy to reverse**: Can move files back if needed
5. **Maintains working code**: All active code stays in place

## Execution Commands

```bash
# Create directories
mkdir -p archive/planning_docs archive/session_contexts archive/backups
mkdir -p arxiv_submission/archive

# Move planning docs (example)
mv CLEANUP_GUIDE.md GPT2_*.md *_SUMMARY.md *_PLAN.md SESSION_*.md archive/planning_docs/

# Move arxiv planning
cd arxiv_submission
mv *_PLAN.md *_SUMMARY.md CLEANUP_*.md COHERENCE_*.md PROOFREADING_*.md archive/

# Move backup
mv safe_backup_20250521_064116 archive/backups/

# Move test outputs
mv test_persistence archive/test_outputs/
```

This approach is much safer and focuses on cleaning documentation clutter without touching any code dependencies.