#!/bin/bash
# Run the 5000 common words experiment with WordNet enrichment

echo "Running GPT-2 5000 Common Words Experiment"
echo "========================================"

# Ensure we're in the project root
cd /mnt/c/Repos/ConceptualFragmentationInLLMsAnalysisAndVisualization

# Run the experiment
python experiments/gpt2/semantic_subtypes/gpt2_5k_common_words_experiment.py

echo "Experiment complete! Check experiments/gpt2/semantic_subtypes/5k_common_words/ for results."