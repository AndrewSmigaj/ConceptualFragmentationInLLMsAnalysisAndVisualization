# Concept MRI Testing Checklist

## Prerequisites
- [ ] Ensure `OPENAI_KEY` is set in `local_config.py`
- [ ] Python environment activated: `venv311\Scripts\activate` (Windows)
- [ ] Demo model created: Run `python concept_mri/demos/create_simple_demo.py`

## Running the App
```bash
# From the project root directory
python concept_mri/app.py

# OR from within concept_mri directory
python -m app
```

The app will start at: http://localhost:8050

## Testing Steps

### 1. Model Upload
- [ ] Navigate to "Feedforward Networks" tab
- [ ] Click "Upload Model"
- [ ] Select: `concept_mri/demos/synthetic_demo/model_*.pt`
- [ ] Verify model info displays correctly

### 2. Dataset Upload
- [ ] Click "Upload Dataset"
- [ ] Select: `concept_mri/demos/synthetic_demo/dataset.npz`
- [ ] Verify dataset info shows 100 samples, 10 features

### 3. Clustering Configuration
- [ ] Select "K-Means" algorithm
- [ ] Choose "Manual" K selection
- [ ] Set k=3
- [ ] Click "Run Clustering"
- [ ] Verify success message appears

### 4. View Results
- [ ] Check "Concept Flow" tab - Sankey diagram should appear
- [ ] Check "Trajectories" tab - Trajectory plot should appear
- [ ] Check "Cluster Details" tab - Cluster cards should appear
- [ ] Check "Metrics" tab - Basic metrics should display

### 5. LLM Analysis
- [ ] Navigate to "LLM Analysis" tab
- [ ] Verify checkboxes for analysis categories appear
- [ ] Select "Interpretation" and "Bias Detection"
- [ ] Click "Run LLM Analysis"
- [ ] Wait for results (may take 10-30 seconds)
- [ ] Verify analysis text appears in formatted cards

## Expected Results

### Clustering Output Format
```python
{
    'completed': True,
    'paths': {
        0: ['L0_C1', 'L1_C2', 'L2_C0'],
        # ... more paths
    },
    'cluster_labels': {
        'L0_C1': 'Layer 0 Cluster 1',
        # ... placeholder labels
    },
    'fragmentation_scores': {
        0: 0.23,
        # ... scores
    }
}
```

### LLM Analysis Output
Should contain sections for:
- **INTERPRETATION**: Conceptual paths and transformations
- **BIAS ANALYSIS**: Demographic routing patterns (if applicable)

## Troubleshooting

### Import Errors
- Check virtual environment is activated
- Run: `pip install -r requirements.txt`

### No Activations Found
- Ensure model was uploaded successfully
- Check browser console for errors

### LLM Analysis Fails
- Verify `OPENAI_KEY` in `local_config.py`
- Check API key has credits
- Look for error messages in terminal

### Clustering Takes Too Long
- Reduce number of clusters (k)
- Use smaller dataset for testing