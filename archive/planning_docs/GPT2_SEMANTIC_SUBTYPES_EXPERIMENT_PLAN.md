# GPT-2 Semantic Subtypes and Contextual Influences Experiment Plan

## Executive Summary

This experiment extends our GPT-2 clustering analysis to probe semantic subtypes within parts-of-speech categories and examine contextual influences on token representations. We will analyze how GPT-2 organizes 800 semantically subtyped words and 200 contextual sentences across its 13 layers using advanced clustering methods and probing classifiers.

## Objectives

### Primary Goals
1. **Semantic subtypes within POS** - Do concrete vs abstract nouns cluster separately?
2. **Context effects** - How does "cat" in "The big cat slept" differ from isolated "cat"?
3. **Model scaling** - Does GPT-2 Medium (345M) show different patterns than Small (117M)?
4. **Advanced clustering** - HDBSCAN vs k-means for natural cluster discovery
5. **Probing alignment** - Do clusters match supervised classification boundaries?

### Scientific Questions
- Do semantic subtypes (e.g., concrete vs abstract nouns) form distinct clusters, or are they subsumed by syntactic roles?
- How do sentence contexts alter token-level vs phrase-level clustering compared to single words?
- Do manner vs degree adverbs retain distinct paths in sentences?
- Which layers best separate subtypes vs syntactic roles, and does this vary by model size?
- Does sentence context enhance semantic clustering in Layer 12?
- Do clustering patterns align with probing classifier accuracy for POS/subtypes?

## Architecture Plan

### Current Infrastructure Inventory

**What we already have:**
- ✅ `SimpleGPT2ActivationExtractor` - handles single words/sentences
- ✅ `GPT2PivotClusterer` - k-means with silhouette optimization  
- ✅ `GPT2APAMetricsCalculator` - basic APA metrics
- ✅ Data pipeline - word lists → activations → clustering → metrics → JSON

**What needs extension:**
- 🔧 GPT-2 Medium support (345M model)
- 🔧 HDBSCAN clustering method
- 🔧 Advanced metrics (AMI, ARI, path curvature)
- 🔧 Probing classifiers
- 🔧 Single-token validation

### Validated Architecture Plan

**Phase 1: Robust Dataset Generation**
- ✅ Single-token validation with fallback strategy
- ✅ Frequency-balanced word curation  
- ✅ Systematic sentence construction
- ✅ Multi-token reporting and replacement

**Phase 2: Memory-Efficient Infrastructure Extensions**
- ✅ Adaptive batch sizing for GPT-2 Medium
- ✅ Progressive degradation on OOM
- ✅ HDBSCAN with automated parameter tuning
- ✅ Robust error handling and recovery

**Phase 3: Validated Analysis Pipeline**
- ✅ Proven sklearn metrics (AMI/ARI)
- ✅ Systematic probing classifier training
- ✅ Context extraction with position validation
- ✅ Comprehensive result validation

## Risk Analysis and Mitigation

### 🚨 High Risk Items

#### 1. Single-token validation complexity
**Risk:** GPT-2 BPE tokenizer splits many words into multiple tokens
**Mitigation:** 
- Pre-validate ALL words with both GPT-2 Small and Medium tokenizers
- Maintain backup word lists (150+ per category to ensure 100 valid)
- Use frequency-balanced replacement for multi-token words

#### 2. HDBSCAN parameter tuning
**Risk:** Poor parameter selection leads to all noise points or single cluster
**Mitigation:**
- Automated parameter tuning with validation
- Fallback to k-means if HDBSCAN fails
- Document parameter selection rationale

#### 3. GPT-2 Medium memory requirements
**Risk:** 345M model requires ~4-6GB peak memory, may cause OOM errors
**Mitigation:**
- Automated batch sizing based on available memory
- Progressive degradation (smaller batches if OOM)
- Memory monitoring and garbage collection

### ⚠️ Medium Risk Items

#### 1. Advanced metrics implementation
**Mitigation:** Use proven sklearn implementations for AMI/ARI, custom path curvature

#### 2. Probing classifier accuracy
**Mitigation:** Standard logistic regression with cross-validation

#### 3. Context extraction alignment
**Mitigation:** Systematic token position validation and error handling

## Dataset Specification

### Word Dataset (800 total)
- **Concrete nouns** (100): cat, dog, house, car, book, tree, chair, etc.
- **Abstract nouns** (100): love, hate, fear, joy, freedom, justice, truth, etc.
- **Physical adjectives** (100): big, small, hot, cold, rough, smooth, etc.
- **Emotive adjectives** (100): happy, sad, good, bad, beautiful, ugly, etc.
- **Manner adverbs** (100): quickly, slowly, carefully, gently, etc.
- **Degree adverbs** (100): very, quite, extremely, barely, etc.
- **Action verbs** (100): run, walk, jump, eat, drive, etc.
- **Stative verbs** (100): be, think, love, know, seem, etc.

### Sentence Dataset (200 total)
- **Noun-focused** (50): "The big cat slept", "A small dog barked", etc.
- **Verb-focused** (50): "He walked slowly", "She ran quickly", etc.
- **Adjective-focused** (50): "The very big cat sat", "A quite small dog ran", etc.
- **Adverb-focused** (50): "She ran quite fast", "He walked very slowly", etc.
- **Ungrammatical controls** (50): "Cat big the slept", "Dog small a barked", etc.

## Implementation Plan

### Phase 1: Dataset Generation & Validation

**TODO-1.1: Single-Token Validation System**
- [ ] Implement `validate_single_token_robust()` with detailed feedback
- [ ] Create `curate_validated_wordlist()` with backup strategy
- [ ] Test validation on both GPT-2 Small and Medium tokenizers
- [ ] Generate tokenization statistics report

**TODO-1.2: Semantic Subtypes Word Curation**
- [ ] Generate 150+ candidate words per subtype (8 subtypes)
- [ ] Validate and select 100 single-token words per subtype
- [ ] Ensure frequency balance using SUBTLEX or similar
- [ ] Create replacement mapping for multi-token words

**TODO-1.3: Contextual Sentences Generation**
- [ ] Generate 50 noun-focused sentences
- [ ] Generate 50 verb-focused sentences  
- [ ] Generate 50 adjective-focused sentences
- [ ] Generate 50 adverb-focused sentences
- [ ] Generate 50 ungrammatical control sentences
- [ ] Validate target words are single tokens in context

**TODO-1.4: Dataset Integration & Export**
- [ ] Combine 800 validated words + 200 sentences
- [ ] Create comprehensive dataset metadata
- [ ] Export in format compatible with existing infrastructure
- [ ] Generate dataset validation report

### Phase 2: Infrastructure Extensions

**TODO-2.1: Memory-Efficient GPT-2 Extractor**
- [ ] Extend `SimpleGPT2ActivationExtractor` for GPT-2 Medium support
- [ ] Implement `MemoryEfficientGPT2Extractor` with adaptive batching
- [ ] Add memory monitoring and garbage collection
- [ ] Implement progressive degradation on OOM errors
- [ ] Test with both model sizes and memory limits

**TODO-2.2: Advanced Clustering System**
- [ ] Create `AdvancedGPT2Clusterer` extending existing clusterer
- [ ] Implement `tune_hdbscan_parameters()` with systematic search
- [ ] Add HDBSCAN clustering with fallback to k-means
- [ ] Implement advanced metrics computation (AMI, ARI, curvature)
- [ ] Add clustering method comparison and validation

**TODO-2.3: Context-Aware Token Extraction**
- [ ] Implement `extract_target_token_from_context()` function
- [ ] Add token position validation and error handling
- [ ] Create alignment verification between single-word and context
- [ ] Test with complex tokenization cases

**TODO-2.4: Probing Classifier System**
- [ ] Implement `train_probing_classifiers()` with cross-validation
- [ ] Create POS classification probes for each layer
- [ ] Create subtype classification probes for each layer
- [ ] Add probe accuracy vs clustering alignment analysis
- [ ] Implement feature importance analysis

### Phase 3: Analysis Pipeline

**TODO-3.1: Advanced Metrics Calculator**
- [ ] Extend existing `GPT2APAMetricsCalculator` for subtypes
- [ ] Implement path curvature calculation
- [ ] Add AMI/ARI computation with statistical significance
- [ ] Create subtype-specific fragmentation analysis
- [ ] Add model comparison metrics (Small vs Medium)

**TODO-3.2: Comprehensive Results Analysis**
- [ ] Adapt existing LLM analysis JSON compilation
- [ ] Add subtype-specific research questions
- [ ] Create clustering method comparison analysis
- [ ] Add context vs single-word comparison metrics
- [ ] Generate probe vs clustering alignment reports

**TODO-3.3: Visualization and Reporting**
- [ ] Create AMI/ARI heatmaps across layers
- [ ] Generate clustering method comparison charts
- [ ] Add path curvature visualization
- [ ] Create model size comparison plots
- [ ] Generate comprehensive methodology report

### Phase 4: Validation & Testing

**TODO-4.1: Component Testing**
- [ ] Unit tests for single-token validation
- [ ] Memory stress tests for GPT-2 Medium extraction
- [ ] HDBSCAN parameter tuning validation
- [ ] Probing classifier accuracy validation
- [ ] Context extraction accuracy verification

**TODO-4.2: Integration Testing**
- [ ] End-to-end pipeline test with small dataset
- [ ] Memory usage profiling across full dataset
- [ ] Cross-validation of clustering results
- [ ] Comparison with baseline POS experiment results
- [ ] Error handling and recovery validation

**TODO-4.3: Results Validation**
- [ ] Statistical significance testing of findings
- [ ] Replication testing with different random seeds
- [ ] Comparison with existing literature benchmarks
- [ ] Peer review of methodology and findings
- [ ] Documentation of limitations and caveats

### Phase 5: Deployment & Documentation

**TODO-5.1: Code Organization**
- [ ] Refactor code into modular, reusable components
- [ ] Add comprehensive docstrings and type hints
- [ ] Create configuration files for different experiment setups
- [ ] Add command-line interface for easy execution
- [ ] Create unit tests and CI/CD pipeline

**TODO-5.2: Documentation**
- [ ] Write comprehensive methodology documentation
- [ ] Create user guide for running experiments
- [ ] Document hardware requirements and recommendations
- [ ] Add troubleshooting guide for common issues
- [ ] Create replication package with all code and data

## Priority Sequence

### High Priority (Critical Path)
- TODO-1.1, 1.2, 1.3, 1.4 (Dataset generation - foundation)
- TODO-2.1 (Memory-efficient extraction - technical risk)
- TODO-2.2 (Advanced clustering - core methodology)

### Medium Priority (Core Features)
- TODO-2.3, 2.4 (Context extraction and probing)
- TODO-3.1, 3.2 (Advanced metrics and analysis)
- TODO-4.1, 4.2 (Testing and validation)

### Lower Priority (Enhancement)
- TODO-3.3 (Visualization)
- TODO-4.3 (Results validation)
- TODO-5.1, 5.2 (Documentation and deployment)

## Decision Points

**After TODO-1.4:** Validate dataset quality before proceeding
**After TODO-2.1:** Confirm memory efficiency with GPT-2 Medium
**After TODO-2.2:** Evaluate HDBSCAN vs k-means clustering quality
**After TODO-3.2:** Review results for scientific significance

## Expected Outcomes

### Technical Outcomes
- 1000 inputs processed (800 words + 200 sentences)
- Both models working (117M + 345M)
- Both clustering methods working (k-means + HDBSCAN)
- Advanced metrics computed (AMI, ARI, curvature)

### Scientific Outcomes
- Clear evidence of subtype separation (or lack thereof)
- Quantified context vs single-word clustering differences
- Model scaling effects on clustering patterns
- Probing vs clustering alignment analysis

## Hardware Requirements

### Minimum Requirements
- 8GB RAM for GPT-2 Small processing
- 16GB RAM recommended for GPT-2 Medium
- CPU-only inference supported (slower)

### Recommended Requirements
- 16GB+ RAM for efficient processing
- GPU with 6GB+ VRAM for faster inference
- SSD storage for faster data loading

## File Structure

```
gpt2_semantic_subtypes_experiment/
├── dataset/
│   ├── gpt2_subtypes_words.txt
│   ├── gpt2_subtypes_word_labels.txt
│   ├── gpt2_subtypes_word_subtypes.txt
│   ├── gpt2_subtypes_sentences.txt
│   └── gpt2_subtypes_sentence_labels.txt
├── infrastructure/
│   ├── memory_efficient_extractor.py
│   ├── advanced_clusterer.py
│   ├── context_extractor.py
│   └── probing_classifier.py
├── analysis/
│   ├── advanced_metrics.py
│   ├── subtype_analysis.py
│   └── visualization.py
├── results/
│   ├── activations/
│   ├── clustering/
│   ├── metrics/
│   └── reports/
└── tests/
    ├── unit_tests/
    ├── integration_tests/
    └── validation_tests/
```

---

**Status:** Ready for implementation
**Next Step:** Begin TODO-1.1 Single-Token Validation System
**Last Updated:** 2025-05-22