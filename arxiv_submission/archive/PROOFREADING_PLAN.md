# Final Proofreading Plan

## Priority 1: GPT-2 Technical Accuracy ⚠️

### Terminology Check
- [ ] **"13 layers" vs "12 layers"**: GPT-2 has 12 transformer blocks + 1 embedding layer. We should be precise:
  - Embedding layer (layer 0)
  - 12 transformer blocks (layers 1-12)
  - Total: 13 activation extraction points
- [ ] **"Transformer layers" vs "layers"**: Be specific when referring to transformer blocks vs all layers
- [ ] **Parameter count**: Verify "117M parameters" is correct for GPT-2 small
- [ ] **Hidden dimension**: Verify 768-dimensional vectors

### Technical Claims
- [ ] Verify clustering happens on **hidden states** not attention weights
- [ ] Check if we're analyzing:
  - Token embeddings (after embedding layer)
  - Hidden states (after each transformer block)
  - Not: attention patterns, output logits
- [ ] Ensure we distinguish between:
  - Single token analysis (semantic subtypes: 774 words)
  - Sentence analysis (semantic pivot: 202 sentences)

### GPT-2 Architecture References
- [ ] Layer naming consistency:
  - "L0" = embedding layer output
  - "L1-L12" = transformer block outputs
  - Or clarify if different
- [ ] Activation extraction points - be explicit about:
  - Post-layernorm vs pre-layernorm
  - After full transformer block (attention + FFN)

## Priority 2: Mathematical Rigor 📐

### Metrics Verification
- [ ] **Stability metric formula**: Verify the formula matches implementation
- [ ] **Gap statistic**: Ensure mathematical notation is correct
- [ ] **Cross-layer metrics**: Check ρ^c, J, F formulas match code
- [ ] **Window boundaries**: Confirm L0-L3, L4-L7, L8-L11 are consistent

### Statistical Claims
- [ ] **72.8% convergence**: Verify this is calculated correctly (412/566)
- [ ] **Path counts**: 19→5→4 progression
- [ ] **Stability values**: 0.724→0.339→0.341
- [ ] **k values**: k=4 for L0, k=2 for L1-L11

## Priority 3: Consistency Checks ✓

### Terminology Consistency
- [ ] "Concept MRI" vs "concept MRI" - pick one capitalization
- [ ] "CTA" vs "Concept Trajectory Analysis" - use appropriately
- [ ] "L{layer}_C{cluster}" notation everywhere
- [ ] "Archetypal paths" vs "archetypical paths"

### Dataset Descriptions
- [ ] Heart disease: 303 patients, 13 features (verify)
- [ ] GPT-2 semantic subtypes: 774 words, 8 categories
- [ ] GPT-2 semantic pivot: 202 sentences

### Section Cross-References
- [ ] Figures referenced exist
- [ ] Tables referenced exist
- [ ] Section references are correct
- [ ] Equation numbers are correct

## Priority 4: Scientific Language 🔬

### Remove Casual Language
- [ ] Remove any remaining informal phrases
- [ ] Check for anthropomorphization of models
- [ ] Ensure claims are supported by evidence
- [ ] Avoid hyperbole ("groundbreaking" → "novel")

### Clarity and Precision
- [ ] Define all acronyms on first use
- [ ] Ensure technical terms are explained
- [ ] Check passive vs active voice consistency
- [ ] Verify causal claims are appropriate

## Priority 5: Grammar and Style 📝

### Basic Checks
- [ ] Spelling (especially technical terms)
- [ ] Punctuation consistency
- [ ] Hyphenation (cross-layer, sub-space, etc.)
- [ ] Citation format consistency

### LaTeX Specific
- [ ] Math mode for all variables ($k$, $\rho^c$, etc.)
- [ ] Consistent use of \textbf{} vs \emph{}
- [ ] Table formatting and alignment
- [ ] Figure caption style

## Priority 6: Final Verification ✅

### Compile Checks
- [ ] LaTeX compiles without errors
- [ ] Bibliography entries complete
- [ ] No overfull hboxes
- [ ] Page count reasonable for venue

### Content Verification
- [ ] Abstract accurately summarizes findings
- [ ] Introduction motivates the work
- [ ] Conclusion doesn't overstate findings
- [ ] Future work is realistic

## Execution Plan

### Phase 1: GPT-2 Technical Review (30 min)
1. Read GPT-2 sections with architecture diagram handy
2. Verify all technical claims
3. Fix terminology issues
4. Ensure we're describing actual GPT-2, not generic transformer

### Phase 2: Mathematical Review (20 min)
1. Check all formulas against implementation
2. Verify numerical results
3. Ensure notation consistency

### Phase 3: Consistency Pass (20 min)
1. Global search/replace for terminology
2. Verify all cross-references
3. Check figure/table references

### Phase 4: Scientific Language (15 min)
1. Remove casual language
2. Tone down hyperbole
3. Ensure precision

### Phase 5: Grammar/Style (15 min)
1. Spell check
2. Grammar check
3. LaTeX formatting

### Phase 6: Final Compile (10 min)
1. Full compile
2. Check output
3. Final read-through

## Key GPT-2 Facts to Verify

1. **Architecture**:
   - 12 transformer blocks
   - 768 hidden dimensions
   - 12 attention heads per block
   - 117M parameters (GPT-2 small)

2. **Our Analysis**:
   - Analyzing hidden states (not attention)
   - 13 extraction points (embedding + 12 blocks)
   - Single tokens, not sequences
   - Clustering in 768-dim space

3. **Findings**:
   - Grammar > semantics organization
   - Happens in middle layers (4-7)
   - Binary clustering (k=2) in most layers
   - Initial semantic awareness (k=4) in embedding layer

## Red Flags to Watch For

1. Claiming we analyze "attention patterns" (we don't)
2. Confusing embedding layer with first transformer block
3. Saying "13 transformer layers" (it's 12 + embedding)
4. Implying causation where we show correlation
5. Overstating what clustering can tell us about model behavior