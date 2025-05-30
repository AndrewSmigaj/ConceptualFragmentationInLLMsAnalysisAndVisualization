# GPT-2 All Tokens Experiment Design

## Overview
Analyze the complete GPT-2 vocabulary (50,257 tokens) to discover how the model organizes its entire representational space.

## Motivation
- Previous experiments focused on single-token words (3,262 tokens)
- Missing 94% of the vocabulary including:
  - Subword tokens ("##ing", "##ed", "##tion")
  - Multi-character tokens (" the", " of", " and")
  - Special tokens (<|endoftext|>)
  - Numbers, punctuation, symbols
  - Non-English tokens

## Research Questions

1. **Subword Organization**: How do subword pieces cluster compared to full words?
   - Do prefixes cluster separately from suffixes?
   - Are morphological patterns (##ing, ##ed, ##ly) recognized?

2. **Token Length Effects**: Does token byte length correlate with clustering?
   - Single characters vs multi-character tokens
   - Leading space tokens (" the") vs no-space tokens ("the")

3. **Special Token Behavior**: Where do special tokens (<|endoftext|>) cluster?

4. **Cross-Linguistic Patterns**: Do non-English tokens form separate clusters?

5. **Frequency vs Representation**: Does token frequency predict cluster assignment?

6. **Punctuation & Symbols**: How are non-alphabetic tokens organized?

7. **Numeric Tokens**: Do number tokens show systematic organization?

8. **Capitalization Effects**: Do capitalized variants cluster differently?

9. **Byte-Pair Encoding Artifacts**: Can we detect BPE construction patterns?

10. **Semantic Leakage in Subwords**: Do semantically related subwords cluster?

## Technical Approach

### Phase 1: Token Characterization
```python
for each token in vocab:
    - Extract token string
    - Identify token type (word, subword, special, numeric, punctuation)
    - Calculate byte length
    - Check for leading space
    - Detect language (if applicable)
    - Get frequency from corpus
```

### Phase 2: Activation Extraction
- Use same approach as 5k experiment but for all 50,257 tokens
- Batch processing essential (memory constraints)
- Save activations in chunks

### Phase 3: Clustering Analysis
- Start with k=500, 1000, 2000 (scale with vocabulary size)
- Use MiniBatchKMeans for efficiency
- Focus on specific layers (0, 6, 11) initially

### Phase 4: Pattern Discovery
- Analyze cluster composition by token type
- Look for unexpected groupings
- Track subword → word relationships

## Expected Challenges

1. **Memory**: 50k tokens × 12 layers × 768 dims = ~1.8GB
2. **Computation**: Clustering 50k points is expensive
3. **Visualization**: Need sampling strategies
4. **Interpretation**: Many tokens are fragments

## Implementation Plan

### Step 1: Token Analysis Script
Create `analyze_all_tokens.py` to characterize the full vocabulary

### Step 2: Batch Activation Extraction
Create `extract_all_activations.py` with:
- Chunked processing (5000 tokens at a time)
- Progress tracking
- Checkpoint saving

### Step 3: Clustering Pipeline
Create `cluster_all_tokens.py` with:
- Scalable clustering (MiniBatchKMeans)
- Multiple k values
- Token-type aware analysis

### Step 4: Visualization
Create focused visualizations:
- Subword cluster networks
- Token length distributions per cluster
- Cross-linguistic cluster maps

## Success Metrics

1. **Discovery of subword organization principles**
2. **Identification of BPE artifacts in representation space**
3. **Understanding of special token placement**
4. **Insights into multilingual token organization**

## Timeline

- Day 1: Token characterization & analysis
- Day 2-3: Activation extraction (checkpointed)
- Day 4: Initial clustering analysis
- Day 5: Pattern discovery & visualization
- Day 6: Report generation

## Key Hypotheses

1. **Subword Hypothesis**: Subword tokens with same function (e.g., all "##ing" endings) will cluster together regardless of stem

2. **Length Hypothesis**: Token byte length will be a major organizing principle, with single-byte tokens clustering separately

3. **Space Hypothesis**: Leading-space tokens (" the") will cluster differently from no-space tokens ("the")

4. **Morphological Hypothesis**: Common morphological patterns will emerge as clusters (plurals, past tense, etc.)

5. **Cross-linguistic Hypothesis**: Non-English tokens will form language-specific clusters