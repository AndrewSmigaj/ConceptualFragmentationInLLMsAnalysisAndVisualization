# Concept MRI Manual Testing Checklist

## 🚀 Starting the Application

- [ ] Run the app: `python -m concept_mri.app`
- [ ] Verify app starts without errors
- [ ] Open browser at http://localhost:8050
- [ ] Verify page loads without console errors (check browser DevTools)

## 📤 Model Upload Tab

### Basic Upload
- [ ] Click "Upload Model" button
- [ ] Select a demo model file (check `/concept_mri/demos/synthetic_demo/`)
- [ ] Verify model loads successfully
- [ ] Check that model architecture is displayed correctly
- [ ] Verify no console errors during upload

### Error Handling
- [ ] Try uploading a non-model file (e.g., .txt file)
- [ ] Verify appropriate error message appears
- [ ] Try uploading corrupted model file
- [ ] Verify error is handled gracefully

## 📊 Dataset Upload Tab

### Basic Upload
- [ ] Click "Upload Dataset" button
- [ ] Select a demo dataset file
- [ ] Verify dataset loads successfully
- [ ] Check that dataset info is displayed (number of samples, features, etc.)
- [ ] Verify no console errors during upload

### Error Handling
- [ ] Try uploading invalid dataset format
- [ ] Verify appropriate error message appears
- [ ] Try uploading very large dataset
- [ ] Check memory usage doesn't spike excessively

## 🔬 Activation Extraction

### Successful Extraction
- [ ] With model and dataset loaded, click "Extract Activations"
- [ ] Verify progress indicator appears
- [ ] Check that extraction completes without errors
- [ ] Verify activations are stored (check session storage)
- [ ] Confirm numpy arrays are maintained (not converted to lists)

### Session Storage Verification
- [ ] Check browser DevTools for session-id-store
- [ ] Verify activation_session_id is present in model-store
- [ ] Confirm activations are NOT directly in Dash stores

## 📐 Window Detection

### Layer Window Manager
- [ ] Verify layer selection dropdown populates with correct layers
- [ ] Select different window sizes
- [ ] Click "Detect Windows"
- [ ] Verify window detection completes without errors
- [ ] Check that window metrics are displayed
- [ ] Verify visualization updates correctly

### Enhanced Features
- [ ] Test adaptive window size detection
- [ ] Verify critical window identification works
- [ ] Check overlap analysis displays correctly
- [ ] Test different window overlap percentages

## 🎯 Clustering Configuration

### Basic Clustering
- [ ] Select clustering method (K-means)
- [ ] Set number of clusters (try 5)
- [ ] Click "Run Clustering"
- [ ] Verify clustering completes without errors
- [ ] Check that cluster assignments are displayed
- [ ] Verify "list has no attribute shape" error does NOT occur

### Advanced Options
- [ ] Try different clustering methods
- [ ] Test various cluster numbers (3, 5, 10)
- [ ] Verify distance metric selection works
- [ ] Check initialization method options

## 📈 Visualizations

### Sankey Diagram
- [ ] Verify Sankey diagram appears after clustering
- [ ] Check that paths are displayed correctly
- [ ] Test interactive features (hover, click)
- [ ] Verify node labels are readable
- [ ] Check color coding is consistent

### Trajectory Visualization
- [ ] Switch to trajectory view
- [ ] Verify trajectories display correctly
- [ ] Test time window selection
- [ ] Check layer progression is accurate
- [ ] Verify legend and labels

## 🤖 LLM Analysis Tab

### Analysis Categories
- [ ] Select "Interpretation" category
- [ ] Click "Run Analysis"
- [ ] Verify analysis completes
- [ ] Check results display correctly
- [ ] Test "Bias Detection" category
- [ ] Test "Efficiency" category
- [ ] Test "Robustness" category

### Results Display
- [ ] Verify results are formatted properly
- [ ] Check that insights are meaningful
- [ ] Test copy/export functionality
- [ ] Verify no encoding errors with special characters

## 🔄 Full Workflow Test

### End-to-End Pipeline
- [ ] Upload fresh model
- [ ] Upload fresh dataset
- [ ] Extract activations
- [ ] Run window detection
- [ ] Configure and run clustering
- [ ] View Sankey diagram
- [ ] Run LLM analysis
- [ ] Verify all results are consistent

### Multi-Session Test
- [ ] Open app in two different browser tabs
- [ ] Upload different models in each tab
- [ ] Verify sessions don't interfere with each other
- [ ] Check that each session maintains its own data

## 🐛 Error Recovery

### Graceful Degradation
- [ ] Disconnect from internet, try LLM analysis
- [ ] Verify appropriate error message
- [ ] Test with missing API keys
- [ ] Check fallback behavior

### Memory Management
- [ ] Upload multiple large models sequentially
- [ ] Verify old sessions are cleaned up
- [ ] Check memory usage stays reasonable
- [ ] Test 2GB session limit enforcement

## 📱 Browser Compatibility

### Cross-Browser Testing
- [ ] Test in Chrome
- [ ] Test in Firefox
- [ ] Test in Edge
- [ ] Test in Safari (if available)
- [ ] Verify all features work consistently

## 🎨 UI/UX Polish

### Visual Consistency
- [ ] Check all buttons have consistent styling
- [ ] Verify loading states are clear
- [ ] Check error messages are user-friendly
- [ ] Verify tooltips appear where needed
- [ ] Test responsive layout at different screen sizes

### Accessibility
- [ ] Test keyboard navigation
- [ ] Verify focus indicators are visible
- [ ] Check color contrast for readability
- [ ] Test with screen reader (if available)

## 📝 Notes Section

### Issues Found:
```
1. 
2. 
3. 
```

### Suggestions for Improvement:
```
1. 
2. 
3. 
```

### Performance Observations:
```
- Model upload time:
- Activation extraction time:
- Clustering time:
- LLM analysis time:
```

## ✅ Sign-off

- [ ] All critical features tested
- [ ] No blocking bugs found
- [ ] Performance is acceptable
- [ ] Ready for demo recording

**Tester:** ______________________
**Date:** ________________________
**Version:** _____________________