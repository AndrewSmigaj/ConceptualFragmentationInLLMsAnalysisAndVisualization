# GPT-2 Archetypal Path Analysis: Step-by-Step Tutorials

This tutorial guide provides comprehensive, hands-on instructions for performing GPT-2 Archetypal Path Analysis (APA). Each tutorial builds on the previous ones, progressing from basic analysis to advanced techniques.

## Table of Contents

1. [Tutorial 1: Getting Started (15 minutes)](#tutorial-1-getting-started)
2. [Tutorial 2: Intermediate Analysis with Attention (30 minutes)](#tutorial-2-intermediate-analysis-with-attention)
3. [Tutorial 3: Advanced Batch Processing (45 minutes)](#tutorial-3-advanced-batch-processing)
4. [Tutorial 4: Custom Analysis and Troubleshooting (30 minutes)](#tutorial-4-custom-analysis-and-troubleshooting)
5. [Quick Reference](#quick-reference)

---

## Tutorial 1: Getting Started

**Time Required**: 15 minutes  
**Prerequisites**: Basic Python knowledge  
**Goal**: Perform your first GPT-2 APA analysis and understand the basic workflow

### Step 1: Environment Setup

First, ensure you have the required dependencies:

```bash
# Check Python version (should be 3.8+)
python --version

# Install required packages if not already installed
pip install torch transformers numpy matplotlib plotly scikit-learn
```

### Step 2: Your First Analysis

Let's analyze a simple sentence to understand concept flow:

```bash
# Navigate to the project directory
cd /path/to/ConceptualFragmentationInLLMsAnalysisAndVisualization

# Run basic analysis
python run_gpt2_analysis.py --text "The quick brown fox jumps over the lazy dog." --visualize --output-dir tutorial_results/getting_started
```

**What happens**: 
1. GPT-2 small model loads
2. Text is tokenized and processed
3. Activations are extracted from all layers
4. 3-layer sliding windows are analyzed
5. Clusters are formed and paths are tracked
6. Visualizations are generated

### Step 3: Understanding the Output

Check your results directory:

```bash
ls -la tutorial_results/getting_started/
```

You should see:
```
gpt2_20241121_123456/
├── activations/           # Raw activation data
├── clusters/             # Cluster assignments
├── results/              # Analysis metrics
├── visualizations/       # Generated plots
└── gpt2_apa_results.json # Summary file
```

### Step 4: Interpreting Your First Results

Open the summary file:

```python
import json

# Load results
with open('tutorial_results/getting_started/gpt2_20241121_123456/gpt2_apa_results.json', 'r') as f:
    results = json.load(f)

# Check basic metrics
print("Analysis Summary:")
print(f"Text analyzed: {results['input_text']}")
print(f"Number of tokens: {results['num_tokens']}")
print(f"Number of windows: {len(results['windows'])}")
print(f"Layers analyzed: {results['layers_analyzed']}")

# Look at fragmentation metrics
if 'fragmentation_metrics' in results:
    print(f"\nFragmentation Metrics:")
    for metric, value in results['fragmentation_metrics'].items():
        print(f"  {metric}: {value:.4f}")
```

### Step 5: View Your First Visualization

Open the visualization files in your browser:

```bash
# Open the main Sankey diagram
open tutorial_results/getting_started/gpt2_*/visualizations/token_sankey_window_0_2.html
```

**What to look for**:
- **Thick flows**: Tokens that maintain consistent cluster assignments
- **Thin/broken flows**: Tokens that change clusters (fragmentation)
- **Color consistency**: How well token identities are preserved

### Step 6: Experiment with Different Texts

Try different types of text to see how complexity affects fragmentation:

```bash
# Simple factual statement
python run_gpt2_analysis.py --text "Water boils at 100 degrees Celsius." --visualize --output-dir tutorial_results/simple

# Complex sentence with multiple concepts
python run_gpt2_analysis.py --text "The artificial intelligence researcher published groundbreaking work on neural network interpretability." --visualize --output-dir tutorial_results/complex

# Narrative text
python run_gpt2_analysis.py --text "Once upon a time, in a galaxy far far away, a young hero embarked on an epic journey." --visualize --output-dir tutorial_results/narrative
```

**Observation Exercise**: Compare the fragmentation patterns between these different text types.

### Tutorial 1 Takeaways

✅ **You've learned**:
- How to run basic GPT-2 APA analysis
- Understanding the output structure
- Reading basic fragmentation metrics
- Interpreting Sankey visualizations

**Next**: Tutorial 2 will show you how to incorporate attention patterns for deeper insights.

---

## Tutorial 2: Intermediate Analysis with Attention

**Time Required**: 30 minutes  
**Prerequisites**: Completed Tutorial 1  
**Goal**: Integrate attention patterns with APA analysis for enhanced interpretation

### Step 1: Enabling Attention Analysis

Attention integration provides deeper insights into concept formation:

```bash
# Run analysis with attention integration
python run_gpt2_analysis.py \
    --text "Scientists discovered that climate change affects ocean temperatures significantly." \
    --model gpt2-medium \
    --visualize \
    --include-attention \
    --output-dir tutorial_results/attention_analysis
```

**New flags explained**:
- `--include-attention`: Extracts attention patterns alongside activations
- `--model gpt2-medium`: Uses larger model for richer attention patterns

### Step 2: Advanced Visualization with Highlighted Tokens

Focus analysis on specific concepts:

```bash
# Highlight key conceptual tokens
python run_gpt2_analysis.py \
    --text "Machine learning algorithms process data to identify patterns and make predictions." \
    --visualize \
    --include-attention \
    --highlight-tokens "learning" "algorithms" "data" "patterns" \
    --output-dir tutorial_results/highlighted_analysis
```

### Step 3: Interpreting Attention-Integrated Results

Load and analyze the enhanced results:

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load attention-integrated results  
with open('tutorial_results/attention_analysis/gpt2_*/gpt2_apa_results.json', 'r') as f:
    results = json.load(f)

# Check for attention metrics
if 'attention_metrics' in results:
    print("Attention Integration Metrics:")
    attention_metrics = results['attention_metrics']
    
    # Print attention-path correlation
    if 'attention_path_correlation' in attention_metrics:
        print(f"Attention-Path Correlation: {attention_metrics['attention_path_correlation']:.4f}")
        
        # Interpret correlation strength
        correlation = attention_metrics['attention_path_correlation']
        if correlation > 0.7:
            print("  → Strong alignment between attention and concept paths")
        elif correlation > 0.4:
            print("  → Moderate alignment - attention partially supports concept flow")
        else:
            print("  → Weak alignment - attention may serve other purposes")
    
    # Print attention entropy by layer
    if 'attention_entropy_by_layer' in attention_metrics:
        print("\nAttention Entropy by Layer:")
        for layer, entropy in attention_metrics['attention_entropy_by_layer'].items():
            print(f"  {layer}: {entropy:.4f}")
            
            # Interpret entropy levels
            if entropy > 0.8:
                print("    → High fragmentation - dispersed attention")
            elif entropy < 0.3:
                print("    → Strong focus - concentrated attention")
            else:
                print("    → Moderate focus")
```

### Step 4: Multi-Head Attention Analysis

Analyze how different attention heads contribute to concept formation:

```python
# If multi-head data is available
if 'attention_heads' in results:
    heads_data = results['attention_heads']
    
    print("\nAttention Head Specialization:")
    for layer, head_info in heads_data.items():
        if 'head_specialization' in head_info:
            specialization = head_info['head_specialization']
            
            print(f"\n{layer}:")
            for head_idx, role in specialization.items():
                print(f"  Head {head_idx}: {role}")
```

### Step 5: Comparing Standard vs Attention-Weighted Analysis

Understanding how attention affects fragmentation metrics:

```python
# Compare fragmentation metrics
if 'standard_fragmentation' in results and 'attention_weighted_fragmentation' in results:
    standard = results['standard_fragmentation']
    weighted = results['attention_weighted_fragmentation']
    
    print("\nFragmentation Comparison:")
    print(f"Standard fragmentation: {standard:.4f}")
    print(f"Attention-weighted fragmentation: {weighted:.4f}")
    
    difference = abs(weighted - standard)
    if difference > 0.1:
        print(f"  → Significant difference ({difference:.4f}) - attention strongly affects concept flow")
    elif difference > 0.05:
        print(f"  → Moderate difference ({difference:.4f}) - attention moderately affects concept flow")
    else:
        print(f"  → Small difference ({difference:.4f}) - attention weakly affects concept flow")
```

### Step 6: Advanced Visualization Interpretation

Open the attention-enhanced visualizations:

```bash
# View attention flow Sankey diagram
open tutorial_results/attention_analysis/gpt2_*/visualizations/attention_sankey_*.html

# View attention-path comparison
open tutorial_results/attention_analysis/gpt2_*/visualizations/attention_comparison_*.html
```

**Key interpretation techniques**:

1. **Attention Sankey Diagrams**:
   - Flow thickness = attention strength
   - Color consistency = attention pattern stability
   - Missing flows = attention gaps (potential fragmentation)

2. **Attention Heatmaps**:
   - Diagonal patterns = local attention (adjacent tokens)
   - Block patterns = concept boundary attention
   - Scattered patterns = fragmented attention

### Step 7: Bias Detection Through Attention

Analyze potential biases in attention patterns:

```bash
# Analyze text with potential gender bias
python run_gpt2_analysis.py \
    --text "The doctor examined the patient while the nurse prepared his medication." \
    --visualize \
    --include-attention \
    --highlight-tokens "doctor" "nurse" "his" \
    --output-dir tutorial_results/bias_analysis
```

Examine the results for attention biases:

```python
# Load bias analysis results
with open('tutorial_results/bias_analysis/gpt2_*/gpt2_apa_results.json', 'r') as f:
    bias_results = json.load(f)

# Check for bias metrics
if 'bias_analysis' in bias_results:
    bias_data = bias_results['bias_analysis']
    
    print("Bias Analysis Results:")
    for bias_type, bias_info in bias_data.items():
        if bias_info.get('bias_detected', False):
            print(f"  {bias_type}: {bias_info['bias_direction']} bias detected (magnitude: {bias_info['bias_magnitude']:.4f})")
        else:
            print(f"  {bias_type}: No significant bias detected")
```

### Tutorial 2 Takeaways

✅ **You've learned**:
- How to integrate attention patterns with APA analysis
- Interpreting attention-path correlations
- Understanding multi-head attention specialization
- Detecting biases through attention patterns
- Reading advanced visualizations

**Next**: Tutorial 3 covers batch processing and production workflows.

---

## Tutorial 3: Advanced Batch Processing

**Time Required**: 45 minutes  
**Prerequisites**: Completed Tutorials 1-2  
**Goal**: Process multiple texts efficiently and generate comparative analyses

### Step 1: Preparing Batch Input Data

Create a dataset for batch analysis:

```python
# Create sample dataset
import json

# Create diverse text samples
texts = [
    {
        "id": "factual_1",
        "category": "factual", 
        "text": "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius."
    },
    {
        "id": "narrative_1",
        "category": "narrative",
        "text": "The brave knight rode through the dark forest to rescue the captured princess."
    },
    {
        "id": "technical_1", 
        "category": "technical",
        "text": "Neural networks use backpropagation to adjust weights during training iterations."
    },
    {
        "id": "argumentative_1",
        "category": "argumentative", 
        "text": "Renewable energy sources are essential for sustainable economic development."
    },
    {
        "id": "descriptive_1",
        "category": "descriptive",
        "text": "The ancient oak tree stood majestically in the center of the peaceful meadow."
    }
]

# Save as JSONL file
with open('tutorial_data/batch_texts.jsonl', 'w') as f:
    for item in texts:
        f.write(json.dumps(item) + '\n')

print("Created batch input file: tutorial_data/batch_texts.jsonl")
```

### Step 2: Running Batch Analysis

Process multiple texts with consistent settings:

```bash
# Create batch processing script
cat > batch_analysis.py << 'EOF'
import json
import subprocess
import os
from pathlib import Path

def run_batch_analysis(input_file, output_base_dir, model="gpt2-medium"):
    """Run GPT-2 APA analysis on a batch of texts."""
    
    # Read input texts
    texts = []
    with open(input_file, 'r') as f:
        for line in f:
            texts.append(json.loads(line.strip()))
    
    # Process each text
    results = []
    
    for i, item in enumerate(texts):
        text_id = item['id']
        category = item['category']
        text = item['text']
        
        print(f"\nProcessing {i+1}/{len(texts)}: {text_id}")
        
        # Create output directory for this text
        output_dir = f"{output_base_dir}/{category}/{text_id}"
        
        # Run analysis
        cmd = [
            "python", "run_gpt2_analysis.py",
            "--text", text,
            "--model", model,
            "--visualize",
            "--include-attention",
            "--output-dir", output_dir
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            results.append({
                "id": text_id,
                "category": category,
                "status": "success",
                "output_dir": output_dir
            })
            print(f"  ✓ Success: {text_id}")
        
        except subprocess.CalledProcessError as e:
            results.append({
                "id": text_id,
                "category": category, 
                "status": "error",
                "error": str(e)
            })
            print(f"  ✗ Error: {text_id} - {e}")
    
    # Save batch results summary
    summary_file = f"{output_base_dir}/batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBatch processing complete. Summary saved to: {summary_file}")
    return results

if __name__ == "__main__":
    # Run batch analysis
    results = run_batch_analysis(
        input_file="tutorial_data/batch_texts.jsonl",
        output_base_dir="tutorial_results/batch_analysis",
        model="gpt2-medium"
    )
    
    # Print summary
    success_count = len([r for r in results if r['status'] == 'success'])
    error_count = len([r for r in results if r['status'] == 'error'])
    
    print(f"\nBatch Summary:")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(results)}")
EOF

# Run the batch analysis
python batch_analysis.py
```

### Step 3: Comparative Analysis Across Text Types

Analyze patterns across different text categories:

```python
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def collect_batch_results(base_dir):
    """Collect and aggregate results from batch analysis."""
    
    results = []
    base_path = Path(base_dir)
    
    # Find all result files
    for result_file in base_path.rglob("gpt2_apa_results.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract key metrics
            result = {
                'file_path': str(result_file),
                'category': result_file.parent.parent.parent.name,
                'text_id': result_file.parent.parent.name,
                'num_tokens': data.get('num_tokens', 0),
                'num_windows': len(data.get('windows', [])),
            }
            
            # Extract fragmentation metrics
            if 'fragmentation_metrics' in data:
                frag_metrics = data['fragmentation_metrics']
                result.update({
                    'trajectory_fragmentation': frag_metrics.get('trajectory_fragmentation', 0),
                    'average_path_length': frag_metrics.get('average_path_length', 0),
                    'unique_paths': frag_metrics.get('unique_paths', 0)
                })
            
            # Extract attention metrics
            if 'attention_metrics' in data:
                attn_metrics = data['attention_metrics']
                result.update({
                    'attention_path_correlation': attn_metrics.get('attention_path_correlation', 0),
                    'mean_attention_entropy': np.mean(list(attn_metrics.get('attention_entropy_by_layer', {}).values()))
                })
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
    
    return pd.DataFrame(results)

# Collect results
df = collect_batch_results("tutorial_results/batch_analysis")
print("Collected results from batch analysis")
print(f"Shape: {df.shape}")
print(f"Categories: {df['category'].unique()}")

# Display summary statistics by category
print("\nSummary by Category:")
summary = df.groupby('category').agg({
    'trajectory_fragmentation': ['mean', 'std'],
    'attention_path_correlation': ['mean', 'std'],
    'mean_attention_entropy': ['mean', 'std'],
    'num_tokens': ['mean', 'std']
}).round(4)

print(summary)
```

### Step 4: Visualizing Comparative Results

Create visualizations comparing text categories:

```python
# Create comparative visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Fragmentation by category
axes[0,0].boxplot([df[df['category']==cat]['trajectory_fragmentation'].values 
                   for cat in df['category'].unique()],
                  labels=df['category'].unique())
axes[0,0].set_title('Trajectory Fragmentation by Text Category')
axes[0,0].set_ylabel('Fragmentation Score')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Attention-path correlation by category  
axes[0,1].boxplot([df[df['category']==cat]['attention_path_correlation'].values 
                   for cat in df['category'].unique()],
                  labels=df['category'].unique())
axes[0,1].set_title('Attention-Path Correlation by Text Category')
axes[0,1].set_ylabel('Correlation Score')
axes[0,1].tick_params(axis='x', rotation=45)

# 3. Attention entropy by category
axes[1,0].boxplot([df[df['category']==cat]['mean_attention_entropy'].values 
                   for cat in df['category'].unique()],
                  labels=df['category'].unique())
axes[1,0].set_title('Mean Attention Entropy by Text Category')
axes[1,0].set_ylabel('Entropy Score')
axes[1,0].tick_params(axis='x', rotation=45)

# 4. Scatter plot: fragmentation vs attention correlation
for category in df['category'].unique():
    cat_data = df[df['category']==category]
    axes[1,1].scatter(cat_data['trajectory_fragmentation'], 
                      cat_data['attention_path_correlation'],
                      label=category, alpha=0.7)

axes[1,1].set_xlabel('Trajectory Fragmentation')
axes[1,1].set_ylabel('Attention-Path Correlation')
axes[1,1].set_title('Fragmentation vs Attention Correlation')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('tutorial_results/batch_analysis/comparative_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Comparative analysis visualization saved to: tutorial_results/batch_analysis/comparative_analysis.png")
```

### Step 5: Generating Batch Report

Create an automated report summarizing batch results:

```python
def generate_batch_report(df, output_file):
    """Generate a comprehensive batch analysis report."""
    
    report = []
    report.append("# GPT-2 APA Batch Analysis Report")
    report.append(f"\nGenerated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total texts analyzed: {len(df)}")
    report.append(f"Categories: {', '.join(df['category'].unique())}")
    
    report.append("\n## Summary Statistics")
    
    # Overall statistics
    report.append("\n### Overall Metrics")
    overall_stats = df.describe().round(4)
    report.append(overall_stats.to_string())
    
    # Category comparison
    report.append("\n### By Category")
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        report.append(f"\n#### {category.title()} Category ({len(cat_data)} texts)")
        
        report.append(f"- Average Fragmentation: {cat_data['trajectory_fragmentation'].mean():.4f} ± {cat_data['trajectory_fragmentation'].std():.4f}")
        report.append(f"- Average Attention-Path Correlation: {cat_data['attention_path_correlation'].mean():.4f} ± {cat_data['attention_path_correlation'].std():.4f}")
        report.append(f"- Average Attention Entropy: {cat_data['mean_attention_entropy'].mean():.4f} ± {cat_data['mean_attention_entropy'].std():.4f}")
    
    # Key findings
    report.append("\n## Key Findings")
    
    # Find category with highest/lowest fragmentation
    frag_by_category = df.groupby('category')['trajectory_fragmentation'].mean()
    highest_frag = frag_by_category.idxmax()
    lowest_frag = frag_by_category.idxmin()
    
    report.append(f"- **Highest Fragmentation**: {highest_frag} texts ({frag_by_category[highest_frag]:.4f})")
    report.append(f"- **Lowest Fragmentation**: {lowest_frag} texts ({frag_by_category[lowest_frag]:.4f})")
    
    # Find category with best attention-path alignment
    corr_by_category = df.groupby('category')['attention_path_correlation'].mean()
    best_alignment = corr_by_category.idxmax()
    worst_alignment = corr_by_category.idxmin()
    
    report.append(f"- **Best Attention-Path Alignment**: {best_alignment} texts ({corr_by_category[best_alignment]:.4f})")
    report.append(f"- **Worst Attention-Path Alignment**: {worst_alignment} texts ({corr_by_category[worst_alignment]:.4f})")
    
    # Interpretation guidelines
    report.append("\n## Interpretation Guidelines")
    report.append("- **High Fragmentation (>0.6)**: Complex concept mixing, potential ambiguity")
    report.append("- **Low Fragmentation (<0.3)**: Clear concept boundaries, stable representations")
    report.append("- **High Attention-Path Correlation (>0.7)**: Strong alignment between attention and concept flow")
    report.append("- **Low Attention-Path Correlation (<0.4)**: Attention serves non-conceptual purposes")
    
    # Save report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Batch report generated: {output_file}")

# Generate the report
generate_batch_report(df, "tutorial_results/batch_analysis/batch_report.md")
```

### Step 6: Automated Dashboard Integration

Set up results for dashboard viewing:

```bash
# Copy batch results for dashboard integration
mkdir -p visualization/data/gpt2_apa/batch_analysis

# Create dashboard-compatible metadata
cat > tutorial_results/batch_analysis/dashboard_metadata.json << 'EOF'
{
  "analysis_type": "batch_gpt2_apa",
  "timestamp": "2024-11-21T12:00:00Z",
  "description": "Batch analysis of diverse text types",
  "categories": ["factual", "narrative", "technical", "argumentative", "descriptive"],
  "total_texts": 5,
  "model_used": "gpt2-medium",
  "analysis_features": ["attention_integration", "bias_detection", "multi_head_analysis"]
}
EOF

# Link for dashboard access
ln -sf ../../../tutorial_results/batch_analysis visualization/data/gpt2_apa/

echo "Batch results prepared for dashboard integration"
echo "Start dashboard with: python visualization/dash_app.py"
```

### Tutorial 3 Takeaways

✅ **You've learned**:
- How to set up and run batch processing workflows
- Collecting and aggregating results across multiple texts
- Comparative analysis techniques across text categories
- Automated report generation
- Integration with dashboard visualization

**Next**: Tutorial 4 covers troubleshooting and custom analysis techniques.

---

## Tutorial 4: Custom Analysis and Troubleshooting

**Time Required**: 30 minutes  
**Prerequisites**: Completed Tutorials 1-3  
**Goal**: Handle edge cases, customize analysis parameters, and troubleshoot common issues

### Step 1: Custom Analysis Parameters

Learn to fine-tune analysis for specific research questions:

```bash
# Analysis with custom clustering parameters
python run_gpt2_analysis.py \
    --text "The complex philosophical argument regarding consciousness and artificial intelligence remains unresolved." \
    --model gpt2-large \
    --window-size 4 \
    --stride 2 \
    --n-clusters 15 \
    --visualize \
    --include-attention \
    --output-dir tutorial_results/custom_analysis
```

**Parameter optimization guide**:
- `--window-size`: Larger windows (4-5) for complex texts, smaller (2-3) for simple texts
- `--stride`: Larger stride (2-3) for efficiency, smaller (1) for comprehensive coverage  
- `--n-clusters`: More clusters (12-20) for complex concepts, fewer (6-10) for simple concepts

### Step 2: Handling Different Text Lengths

Optimize analysis for various text lengths:

```python
# Create texts of different lengths for testing
test_texts = {
    "short": "AI is powerful.",
    "medium": "Artificial intelligence systems are becoming increasingly sophisticated and capable of performing complex tasks that were once thought to require human intelligence.",
    "long": """The development of artificial intelligence has been one of the most significant technological advances of the 21st century. From simple rule-based systems to complex neural networks capable of learning and adaptation, AI has transformed numerous industries including healthcare, finance, transportation, and education. However, as these systems become more powerful and autonomous, important questions arise about their impact on society, employment, privacy, and human agency. Researchers and policymakers must work together to ensure that AI development proceeds in a responsible manner that maximizes benefits while minimizing potential risks."""
}

# Analyze each with optimized parameters
for length, text in test_texts.items():
    # Adjust parameters based on text length
    if length == "short":
        window_size, n_clusters = 2, 6
    elif length == "medium":  
        window_size, n_clusters = 3, 10
    else:  # long
        window_size, n_clusters = 4, 15
    
    cmd = f"""python run_gpt2_analysis.py \
        --text "{text}" \
        --window-size {window_size} \
        --n-clusters {n_clusters} \
        --visualize \
        --output-dir tutorial_results/length_optimization/{length}"""
    
    print(f"Running analysis for {length} text...")
    print(cmd)
    # os.system(cmd)  # Uncomment to run
```

### Step 3: Common Issues and Solutions

#### Issue 1: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Solution 1: Use smaller model
python run_gpt2_analysis.py --text "Your text" --model gpt2 --device cuda

# Solution 2: Use CPU
python run_gpt2_analysis.py --text "Your text" --model gpt2-medium --device cpu

# Solution 3: Reduce context window
python run_gpt2_analysis.py --text "Your text" --context-window 256 --device cuda

# Solution 4: Process in smaller chunks
python run_gpt2_analysis.py --text "Your very long text..." --window-size 2 --stride 2
```

#### Issue 2: Poor Clustering Results

**Symptoms**: All tokens in one cluster or random cluster assignments

**Diagnostic script**:
```python
import json
import numpy as np

def diagnose_clustering_issues(results_file):
    """Diagnose clustering quality issues."""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    issues = []
    
    # Check cluster distribution
    if 'clusters' in results:
        for window_name, window_data in results['clusters'].items():
            if 'labels' in window_data:
                labels = window_data['labels']
                unique_clusters = len(set(labels))
                
                if unique_clusters == 1:
                    issues.append(f"{window_name}: All tokens in single cluster")
                elif unique_clusters == len(labels):
                    issues.append(f"{window_name}: Each token in separate cluster")
                
                # Check cluster size distribution
                cluster_sizes = {}
                for label in labels:
                    cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
                
                max_cluster_size = max(cluster_sizes.values())
                total_tokens = len(labels)
                
                if max_cluster_size > 0.8 * total_tokens:
                    issues.append(f"{window_name}: Dominant cluster contains {max_cluster_size}/{total_tokens} tokens")
    
    return issues

# Example usage
issues = diagnose_clustering_issues("tutorial_results/custom_analysis/gpt2_*/gpt2_apa_results.json")
for issue in issues:
    print(f"⚠️  {issue}")
```

**Solutions for clustering issues**:
```bash
# Increase number of clusters
python run_gpt2_analysis.py --text "Your text" --n-clusters 20

# Use different model size
python run_gpt2_analysis.py --text "Your text" --model gpt2-medium

# Adjust window size
python run_gpt2_analysis.py --text "Your text" --window-size 5
```

#### Issue 3: Low Attention-Path Correlation

**Symptoms**: Correlation < 0.3 between attention patterns and cluster paths

**Investigation script**:
```python
def investigate_low_correlation(results_file):
    """Investigate reasons for low attention-path correlation."""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("Low Correlation Investigation:")
    
    # Check if attention data is present
    if 'attention_metrics' not in results:
        print("❌ No attention data found - enable --include-attention")
        return
    
    attention_metrics = results['attention_metrics']
    correlation = attention_metrics.get('attention_path_correlation', 0)
    
    print(f"Attention-Path Correlation: {correlation:.4f}")
    
    # Check attention entropy levels
    if 'attention_entropy_by_layer' in attention_metrics:
        entropies = list(attention_metrics['attention_entropy_by_layer'].values())
        mean_entropy = np.mean(entropies)
        
        print(f"Mean Attention Entropy: {mean_entropy:.4f}")
        
        if mean_entropy > 0.8:
            print("➤ High entropy suggests dispersed attention - may indicate:")
            print("  - Text complexity beyond model capacity")
            print("  - Need for larger model")
            print("  - Attention serving syntactic vs semantic roles")
        
        elif mean_entropy < 0.2:
            print("➤ Very low entropy suggests overly focused attention - may indicate:")
            print("  - Simple text with obvious patterns")
            print("  - Model attention artifacts") 
            print("  - Need for different analysis approach")
    
    # Check fragmentation levels
    if 'fragmentation_metrics' in results:
        frag = results['fragmentation_metrics'].get('trajectory_fragmentation', 0)
        print(f"Trajectory Fragmentation: {frag:.4f}")
        
        if frag > 0.7 and correlation < 0.3:
            print("➤ High fragmentation with low correlation suggests:")
            print("  - Complex concept mixing")
            print("  - Attention processing different information than clustering")
            print("  - Consider multi-head attention analysis")

# Run investigation
investigate_low_correlation("tutorial_results/custom_analysis/gpt2_*/gpt2_apa_results.json")
```

### Step 4: Custom Attention Analysis

Advanced attention pattern investigation:

```python
# Custom attention head analysis script
import torch
import json
from pathlib import Path

def analyze_attention_heads_custom(model_path, text, output_dir):
    """Perform custom multi-head attention analysis."""
    
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path, output_attentions=True)
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        attentions = outputs.attentions  # Tuple of attention tensors
    
    # Analyze each layer and head
    head_analysis = {}
    
    for layer_idx, layer_attention in enumerate(attentions):
        layer_name = f"layer_{layer_idx}"
        head_analysis[layer_name] = {}
        
        # Shape: [batch_size, num_heads, seq_len, seq_len]
        batch_size, num_heads, seq_len, _ = layer_attention.shape
        
        for head_idx in range(num_heads):
            head_attention = layer_attention[0, head_idx].numpy()  # Remove batch dim
            
            # Calculate head-specific metrics
            # 1. Attention entropy
            entropy = calculate_attention_entropy(head_attention)
            
            # 2. Diagonal vs off-diagonal attention
            diagonal_strength = np.mean(np.diag(head_attention))
            off_diagonal_strength = np.mean(head_attention - np.diag(np.diag(head_attention)))
            
            # 3. Local vs global attention pattern
            local_attention = calculate_local_attention(head_attention, window=3)
            global_attention = calculate_global_attention(head_attention, window=3)
            
            head_analysis[layer_name][f"head_{head_idx}"] = {
                "entropy": float(entropy),
                "diagonal_strength": float(diagonal_strength),
                "off_diagonal_strength": float(off_diagonal_strength),
                "local_attention": float(local_attention),
                "global_attention": float(global_attention)
            }
    
    # Save detailed analysis
    output_file = Path(output_dir) / "detailed_attention_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(head_analysis, f, indent=2)
    
    print(f"Detailed attention analysis saved: {output_file}")
    return head_analysis

def calculate_attention_entropy(attention_matrix):
    """Calculate entropy of attention distribution."""
    # Flatten and normalize
    flat_attention = attention_matrix.flatten()
    flat_attention = flat_attention[flat_attention > 1e-10]  # Remove zeros
    flat_attention = flat_attention / flat_attention.sum()
    
    # Calculate entropy
    entropy = -np.sum(flat_attention * np.log2(flat_attention))
    return entropy

def calculate_local_attention(attention_matrix, window=3):
    """Calculate strength of local attention patterns."""
    seq_len = attention_matrix.shape[0]
    local_sum = 0
    local_count = 0
    
    for i in range(seq_len):
        for j in range(max(0, i-window), min(seq_len, i+window+1)):
            local_sum += attention_matrix[i, j]
            local_count += 1
    
    return local_sum / local_count if local_count > 0 else 0

def calculate_global_attention(attention_matrix, window=3):
    """Calculate strength of global attention patterns.""" 
    seq_len = attention_matrix.shape[0]
    global_sum = 0
    global_count = 0
    
    for i in range(seq_len):
        for j in range(seq_len):
            if abs(i - j) > window:
                global_sum += attention_matrix[i, j]
                global_count += 1
    
    return global_sum / global_count if global_count > 0 else 0

# Run custom analysis
custom_results = analyze_attention_heads_custom(
    model_path="gpt2-medium",
    text="The artificial intelligence researcher discovered fascinating patterns in neural network behavior.",
    output_dir="tutorial_results/custom_attention"
)
```

### Step 5: Performance Optimization

Optimize analysis for production use:

```bash
# Create performance-optimized analysis script
cat > optimized_analysis.py << 'EOF'
import time
import psutil
import json
from pathlib import Path

def performance_optimized_analysis(text, model_size="gpt2", use_gpu=True):
    """Run performance-optimized GPT-2 APA analysis."""
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Optimize parameters based on text length and model size
    text_length = len(text.split())
    
    if text_length < 10:
        window_size, n_clusters, stride = 2, 6, 1
    elif text_length < 50:
        window_size, n_clusters, stride = 3, 10, 1  
    elif text_length < 200:
        window_size, n_clusters, stride = 3, 12, 2
    else:
        window_size, n_clusters, stride = 4, 15, 3
    
    # Adjust for model size
    if model_size in ["gpt2-large", "gpt2-xl"] and not use_gpu:
        # Reduce parameters for large models on CPU
        window_size = max(2, window_size - 1)
        n_clusters = max(6, n_clusters - 3)
    
    device = "cuda" if use_gpu else "cpu"
    
    # Build command
    cmd = f"""python run_gpt2_analysis.py \
        --text "{text}" \
        --model {model_size} \
        --device {device} \
        --window-size {window_size} \
        --n-clusters {n_clusters} \
        --stride {stride} \
        --visualize \
        --output-dir tutorial_results/performance_test"""
    
    print(f"Optimized parameters: window_size={window_size}, n_clusters={n_clusters}, stride={stride}")
    print(f"Command: {cmd}")
    
    # Run analysis (uncomment to execute)
    # import os
    # os.system(cmd)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    performance_stats = {
        "execution_time": end_time - start_time,
        "memory_usage": end_memory - start_memory,
        "text_length": text_length,
        "model_size": model_size,
        "optimized_params": {
            "window_size": window_size,
            "n_clusters": n_clusters,
            "stride": stride
        }
    }
    
    return performance_stats

# Test with different configurations
test_text = "Machine learning algorithms analyze patterns in data to make predictions about future outcomes."

configs = [
    ("gpt2", True),
    ("gpt2-medium", True),
    ("gpt2-medium", False)
]

for model, use_gpu in configs:
    print(f"\nTesting: {model} ({'GPU' if use_gpu else 'CPU'})")
    stats = performance_optimized_analysis(test_text, model, use_gpu)
    print(f"Optimized parameters: {stats['optimized_params']}")
EOF

python optimized_analysis.py
```

### Step 6: Quality Assurance Checklist

Create a comprehensive quality check:

```python
def quality_assurance_check(results_directory):
    """Comprehensive quality assurance for GPT-2 APA results."""
    
    qa_results = {
        "passed": [],
        "warnings": [], 
        "errors": []
    }
    
    results_path = Path(results_directory)
    result_file = None
    
    # Find results file
    for json_file in results_path.rglob("gpt2_apa_results.json"):
        result_file = json_file
        break
    
    if not result_file:
        qa_results["errors"].append("No results file found")
        return qa_results
    
    # Load results
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Test 1: Basic data presence
    required_fields = ['input_text', 'num_tokens', 'windows', 'layers_analyzed']
    for field in required_fields:
        if field in results:
            qa_results["passed"].append(f"✓ {field} present")
        else:
            qa_results["errors"].append(f"✗ {field} missing")
    
    # Test 2: Clustering quality
    if 'clusters' in results:
        for window_name, cluster_data in results['clusters'].items():
            if 'labels' in cluster_data:
                labels = cluster_data['labels']
                unique_clusters = len(set(labels))
                
                if 1 < unique_clusters < len(labels):
                    qa_results["passed"].append(f"✓ {window_name}: Good cluster distribution")
                elif unique_clusters == 1:
                    qa_results["warnings"].append(f"⚠ {window_name}: All tokens in one cluster")
                else:
                    qa_results["warnings"].append(f"⚠ {window_name}: Each token in separate cluster")
    
    # Test 3: Attention analysis (if present)
    if 'attention_metrics' in results:
        attention_metrics = results['attention_metrics']
        
        correlation = attention_metrics.get('attention_path_correlation', 0)
        if correlation > 0.5:
            qa_results["passed"].append(f"✓ Good attention-path correlation ({correlation:.3f})")
        elif correlation > 0.3:
            qa_results["warnings"].append(f"⚠ Moderate attention-path correlation ({correlation:.3f})")
        else:
            qa_results["warnings"].append(f"⚠ Low attention-path correlation ({correlation:.3f})")
    
    # Test 4: Visualization files
    viz_dir = results_path / "visualizations"
    if viz_dir.exists():
        html_files = list(viz_dir.glob("*.html"))
        if html_files:
            qa_results["passed"].append(f"✓ {len(html_files)} visualization files generated")
        else:
            qa_results["warnings"].append("⚠ No visualization files found")
    else:
        qa_results["warnings"].append("⚠ Visualization directory not found")
    
    return qa_results

# Run QA check
qa_report = quality_assurance_check("tutorial_results/custom_analysis")

print("Quality Assurance Report:")
print("=" * 40)

for check in qa_report["passed"]:
    print(check)

for warning in qa_report["warnings"]:
    print(warning)

for error in qa_report["errors"]:
    print(error)

print(f"\nSummary: {len(qa_report['passed'])} passed, {len(qa_report['warnings'])} warnings, {len(qa_report['errors'])} errors")
```

### Tutorial 4 Takeaways

✅ **You've learned**:
- How to customize analysis parameters for different text types
- Troubleshooting common issues and their solutions
- Advanced attention pattern investigation techniques
- Performance optimization strategies
- Quality assurance procedures

---

## Quick Reference

### Essential Commands

```bash
# Basic analysis  
python run_gpt2_analysis.py --text "Your text" --visualize

# With attention
python run_gpt2_analysis.py --text "Your text" --include-attention --visualize

# Custom parameters
python run_gpt2_analysis.py --text "Your text" --model gpt2-medium --window-size 4 --n-clusters 15

# Batch processing
python batch_analysis.py  # (using script from Tutorial 3)
```

### Parameter Guidelines

| Text Type | Window Size | N Clusters | Stride | Model |
|-----------|-------------|------------|--------|-------|
| Simple/Short | 2-3 | 6-8 | 1 | gpt2 |
| Medium Complexity | 3-4 | 10-12 | 1-2 | gpt2-medium |
| Complex/Long | 4-5 | 12-20 | 2-3 | gpt2-large |

### Interpretation Thresholds

| Metric | Excellent | Good | Poor |
|--------|-----------|------|------|
| Attention-Path Correlation | >0.7 | 0.4-0.7 | <0.4 |
| Trajectory Fragmentation | <0.3 | 0.3-0.6 | >0.6 |
| Attention Entropy | 0.3-0.7 | 0.2-0.3, 0.7-0.8 | <0.2, >0.8 |

### Troubleshooting Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| CUDA OOM | `--device cpu` or `--model gpt2` |
| Poor clustering | `--n-clusters 15` or `--model gpt2-medium` |
| Low correlation | Check text complexity, try `--window-size 4` |
| No visualizations | Install `pip install plotly` |

### File Locations

- **Results**: `tutorial_results/[analysis_name]/gpt2_*/gpt2_apa_results.json`
- **Visualizations**: `tutorial_results/[analysis_name]/gpt2_*/visualizations/`
- **Dashboard Data**: `visualization/data/gpt2_apa/`

This comprehensive tutorial series provides everything needed to successfully perform GPT-2 Archetypal Path Analysis, from basic concepts to advanced production workflows.