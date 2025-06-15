# Choosing the Best Prediction Task for Apple CTA Paper

## Current Options Analysis

### 1. Premium/Standard/Juice Routing (Original)
**Pros:**
- Clear business value
- Good narrative about misrouting
- Only 324 labeled samples

**Cons:**
- Very few samples when split 3 ways
- Subjective labels ("good" vs "very good")
- Severe class imbalance (14%/11%/5%)

### 2. Binary: Premium vs Non-Premium
**Pros:**
- Larger sample per class
- Clearer business case
- Simpler story

**Cons:**
- Still only 324 labeled samples
- Loses nuance of juice routing

### 3. Storage Quality Prediction (from text features)
**Pros:**
- Rich text descriptions available
- Clear value prop (predict storage life)
- Could use more samples

**Cons:**
- Need NLP preprocessing
- Storage quality often missing
- Less clear CTA story

### 4. Brix (Sugar) Prediction from Other Features
**Pros:**
- 1,071 samples available!
- Continuous target (regression)
- Can bin into categories for CTA
- Clear importance to industry

**Cons:**
- Less dramatic than misrouting story
- Need to justify why predict Brix

### 5. Variety Classification (Top 10 varieties)
**Pros:**
- Directly addresses "can AI tell varieties apart?"
- Good sample sizes for top varieties
- Natural multi-class problem
- CTA can show where varieties converge/diverge

**Cons:**
- Need to subset to well-represented varieties
- Less direct business impact

## 🎯 RECOMMENDED: Variety Classification with Economic Framing

**The Setup:**
- Predict top 10 varieties (Honeycrisp, Buckeye Gala, Macoun, etc.)
- Each has 20+ samples
- Total ~400 samples, balanced classes
- Frame as: "Understanding how AI sees variety differences"

**Why This Works Best:**
1. **Sufficient data**: 40+ samples per class after train/test split
2. **Natural CTA story**: Track where varieties merge/separate
3. **Business relevance**: Variety determines price
4. **Clear insights**: "Honeycrisp converges with Gala at layer 2"
5. **Actionable**: Shows exactly which varieties get confused

**The Narrative:**
"We use CTA to understand how neural networks distinguish apple varieties, revealing why premium cultivars like Honeycrisp (worth 5x more) get confused with standard varieties. By tracking variety representations through network layers, we identify where and why these costly confusions occur."

## Implementation Plan

1. **Select top 10 varieties by sample count**
2. **Create balanced dataset (~400 samples)**
3. **Use same 8 chemical/physical features**
4. **4-layer network (8→32→64→32→10)**
5. **Apply CTA to track variety trajectories**
6. **Focus analysis on premium variety confusion**