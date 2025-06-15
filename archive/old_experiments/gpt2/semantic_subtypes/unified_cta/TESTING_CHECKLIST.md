# Unified CTA Pipeline - Testing Checklist

## 🎯 Primary Goal
Fix the ETS single-cluster problem by implementing two-tier clustering with adaptive thresholds.

## ✅ Critical Success Criteria
1. **No more single mega-clusters** - Each layer should have multiple clusters
2. **Coverage >70%** - Up from the broken ~15%
3. **Maintains quality** - Purity stays above 80%
4. **Works with existing data** - Uses activations_by_layer.pkl without modification

## 📋 Testing Steps

### Step 1: Basic Sanity Check (5 min)
```bash
cd experiments/gpt2/semantic_subtypes/unified_cta
python test_pipeline_init.py
```
**Expected**: "SUCCESS: All pipeline components initialized correctly!"

### Step 2: Data Verification (5 min)
```bash
# Check input files exist
ls -la ../data/activations_by_layer.pkl
ls -la ../data/curated_word_lists.json
```
**Expected**: Both files exist

### Step 3: Quick Test Run (15 min)
```bash
# Test with 3 layers only
python run_unified_pipeline.py --experiment quick_test
```
**Expected**: 
- No errors
- Creates results/run_[timestamp] directory
- Each layer produces >1 cluster

### Step 4: Verify Key Improvements (10 min)
```python
# Check clustering results
import json
from pathlib import Path

# Get latest run
run_dir = sorted(Path('results').glob('run_*'))[-1]

# Check macro clustering
for f in run_dir.glob('clustering/layer_*/clustering_results.json'):
    with open(f) as fp:
        data = json.load(fp)
    print(f"{f.parent.name}: k={data['optimal_k']} clusters")

# Check micro clustering coverage
for f in run_dir.glob('micro_clustering/layer_*/micro_results.json'):
    with open(f) as fp:
        results = json.load(fp)
    coverages = [r['coverage'] for r in results]
    print(f"{f.parent.name}: avg coverage={sum(coverages)/len(coverages):.3f}")
```
**Expected**:
- Each layer: k > 1 (multiple clusters)
- Average coverage > 0.7

### Step 5: Full Pipeline Test (30 min)
```bash
# Run all 12 layers
python run_unified_pipeline.py --experiment full
```
**Expected**: Completes without errors, generates full results

### Step 6: Compare to Baseline (10 min)
```python
# Simple comparison
print("BEFORE (Broken ETS):")
print("- All words in 1 cluster per layer")
print("- Coverage: ~15%")
print("- No meaningful trajectories")

print("\nAFTER (Fixed Pipeline):")
# Load and print actual results
```

## 🔍 What to Look For

### In the Logs
- "Layer X: optimal k = Y" where Y > 1
- "coverage=0.7xx, purity=0.8xx"
- "Successfully clustered X/12 layers"

### In the Results
- `clustering/layer_*/clustering_results.json` - Check optimal_k > 1
- `micro_clustering/layer_*/micro_results.json` - Check coverage > 0.7
- `paths/trajectories.json` - Diverse paths, not all identical

### Red Flags 🚩
- Any layer with k=1 (single cluster)
- Coverage < 0.5
- All trajectories identical
- Memory errors or crashes

## 📊 Simple Success Check

```python
def check_success(run_dir):
    issues = []
    
    # Check each layer
    for layer in range(12):
        # Check macro clusters
        cluster_file = run_dir / f'clustering/layer_{layer}/clustering_results.json'
        if cluster_file.exists():
            with open(cluster_file) as f:
                data = json.load(f)
            if data['optimal_k'] == 1:
                issues.append(f"Layer {layer} has only 1 cluster!")
        
        # Check micro coverage
        micro_file = run_dir / f'micro_clustering/layer_{layer}/micro_results.json'
        if micro_file.exists():
            with open(micro_file) as f:
                results = json.load(f)
            avg_coverage = sum(r['coverage'] for r in results) / len(results)
            if avg_coverage < 0.6:
                issues.append(f"Layer {layer} coverage too low: {avg_coverage:.3f}")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ ALL CHECKS PASSED!")
        return True
```

## 🚀 Next Steps After Testing

If tests pass:
1. Generate visualization of improvements
2. Create summary report for paper
3. Run on additional datasets

If tests fail:
1. Check gap statistic parameters
2. Verify percentile thresholds
3. Review preprocessing steps

---

**Remember**: The main goal is fixing the single-cluster problem. Everything else is secondary.