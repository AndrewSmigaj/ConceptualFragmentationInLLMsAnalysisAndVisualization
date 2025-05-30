#!/usr/bin/env python
"""
Simplified test script for LLM analysis
"""

import os
import json
import sys
import numpy as np

# Ensure we can import from the project root
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from concept_fragmentation.llm.analysis import ClusterAnalysis

# Load the data
data_path = "data/cluster_paths/titanic_seed_0_paths.json"
print(f"Loading data from {data_path}")
with open(data_path, "r") as f:
    data = json.load(f)

# Extract a few centroids for testing
centroids = {}
count = 0
for unique_id, centroid_values in data["unique_centroids"].items():
    if unique_id in data["id_mapping"]:
        mapping = data["id_mapping"][unique_id]
        layer_name = mapping["layer_name"]
        original_id = mapping["original_id"]
        
        # Construct a human-readable cluster ID
        cluster_id = f"L{mapping['layer_idx']}C{original_id}"
        centroids[cluster_id] = np.array(centroid_values)
        count += 1
        if count >= 4:  # Just get a few for testing
            break

# Initialize LLM client
analyzer = ClusterAnalysis(
    provider="grok",
    model="default",
    use_cache=True
)

# Generate labels for the centroids
print(f"Generating labels for {len(centroids)} sample clusters...")
labels = analyzer.label_clusters_sync(centroids)

# Print the results
for cluster_id, label in labels.items():
    print(f"{cluster_id}: {label}")

# Save results to a file for later analysis
results = {"cluster_labels": labels}
with open("sample_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Sample results saved to sample_analysis_results.json")