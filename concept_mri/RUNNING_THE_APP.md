# Running Concept MRI

## Prerequisites

1. **Python Environment**: Make sure the virtual environment is activated:
   ```bash
   # Windows
   venv311\Scripts\activate
   
   # Linux/Mac
   source venv311/bin/activate
   ```

2. **API Keys**: Ensure `local_config.py` exists with your OpenAI API key:
   ```python
   OPENAI_KEY = "your-api-key-here"
   ```

3. **Demo Model**: The demo model should already exist at:
   ```
   concept_mri/demos/synthetic_demo/
   ```

## Starting the App

### Method 1: Using the run script (Recommended)
```bash
python run_concept_mri.py
```

### Method 2: Direct execution
```bash
cd concept_mri
python -m app
```

The app will start at: http://localhost:8050

## Testing the Pipeline

### Automated Testing
To test the full pipeline programmatically:
```bash
python test_full_pipeline.py
```

This will:
1. Load the demo model
2. Load dataset and activations
3. Run clustering
4. Extract cluster paths
5. Format data for LLM
6. Run LLM analysis (if API key is available)

### Manual Testing via UI

1. **Navigate to the app**: http://localhost:8050
2. **Select "Feedforward Networks" tab**
3. **Upload Model**: Click "Upload Model" and select `concept_mri/demos/synthetic_demo/model_*.pt`
4. **Upload Dataset**: Click "Upload Dataset" and select `concept_mri/demos/synthetic_demo/dataset.npz`
5. **Configure Clustering**:
   - Algorithm: K-Means
   - K Selection: Manual
   - Number of clusters: 3
6. **Run Clustering**: Click "Run Clustering" button
7. **View Results**:
   - Concept Flow: Sankey diagram
   - Trajectories: Trajectory plot
   - Cluster Details: Cluster information cards
   - Metrics: Basic metrics
8. **Run LLM Analysis**:
   - Go to "LLM Analysis" tab
   - Select analysis categories (Interpretation, Bias, etc.)
   - Click "Run LLM Analysis"
   - Wait for results (10-30 seconds)

## Troubleshooting

### Import Errors
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`

### LLM Analysis Fails
- Check that `OPENAI_KEY` is set in `local_config.py`
- Verify API key has credits
- Check terminal for error messages

### App Won't Start
- Check if port 8050 is already in use
- Try a different port: `python run_concept_mri.py --port 8051`

## What's Working

✅ Model upload and parsing
✅ Dataset upload and validation
✅ Activation extraction
✅ Clustering with multiple algorithms
✅ Cluster path extraction
✅ LLM analysis integration
✅ Results visualization (Sankey, trajectories, etc.)
✅ Analysis category selection
✅ Export functionality

## Known Limitations

- Only supports feedforward networks currently
- GPT analysis tab is a placeholder
- Some advanced metrics not yet implemented
- UI performance can be slow with large datasets