# Activation Files Summary

## Current Activation Files

### 1. **All Tokens Chunked (KEEP)** - experiments/gpt2/all_tokens/activations/
- **Size**: 1.8GB total (11 chunks × 176MB each)
- **Content**: GPT-2 activations for all 50,257 tokens
- **Purpose**: Complete activation data for future experiments
- **Status**: ✅ KEEP - This is the canonical full dataset

### 2. **Frequent Token Activations (REMOVE)** - experiments/gpt2/all_tokens/frequent_token_activations.npy
- **Size**: 352MB
- **Content**: Subset of frequent tokens only
- **Status**: ❌ REMOVE - Redundant with all_tokens chunks

### 3. **Top 10k Activations (REMOVE)** - experiments/gpt2/all_tokens/top_10k_activations.npy
- **Size**: 352MB
- **Content**: Top 10k tokens subset
- **Status**: ❌ REMOVE - Redundant with all_tokens chunks

### 4. **5k Common Words (REMOVE)** - experiments/gpt2/semantic_subtypes/5k_common_words/activations.npy
- **Size**: 115MB
- **Content**: 5k common words experiment
- **Status**: ❌ REMOVE - Can be regenerated from all_tokens if needed

## Git LFS Consideration

**Current situation**:
- `.npy` files are in .gitignore (not tracked by git)
- Total activation data: ~2.6GB

**Git LFS recommendation**: NOT NEEDED
- Since activation files are already gitignored, they won't be pushed to the repository
- This is the correct approach for large intermediate data files
- Users who need the activations can regenerate them or obtain them separately

## Cleanup Actions

1. **Remove redundant activations** (saves ~819MB):
   ```bash
   rm experiments/gpt2/all_tokens/frequent_token_activations.npy
   rm experiments/gpt2/all_tokens/top_10k_activations.npy
   rm experiments/gpt2/semantic_subtypes/5k_common_words/activations.npy
   ```

2. **Keep only the canonical all_tokens chunks**:
   - These contain the complete data
   - Any subset can be extracted from these chunks
   - Essential for future experiments with different clustering parameters

## Verification Before Removal

- [ ] Confirm no active scripts depend on the standalone .npy files
- [ ] Verify that subset extraction from chunks works properly
- [ ] Document how to extract subsets from the chunked data