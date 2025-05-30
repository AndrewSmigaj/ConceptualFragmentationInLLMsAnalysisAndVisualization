#!/usr/bin/env python3
"""Check label consistency for similar clusters."""

import json
from collections import defaultdict

with open('llm_labels_k10/cluster_labels_k10.json', 'r') as f:
    data = json.load(f)

# Check if clusters with very similar tokens have the same label
with open('llm_labels_k10/llm_labeling_data.json', 'r') as f:
    cluster_data = json.load(f)

# Find clusters with high token overlap
similar_pairs = []
clusters = cluster_data['clusters']

for k1 in clusters:
    for k2 in clusters:
        if k1 < k2:  # Avoid duplicates
            tokens1 = set(clusters[k1]['common_tokens'][:20])
            tokens2 = set(clusters[k2]['common_tokens'][:20])
            overlap = len(tokens1 & tokens2)
            if overlap >= 10:  # High overlap
                layer1 = int(k1.split('_')[0][1:])
                layer2 = int(k2.split('_')[0][1:])
                label1 = data['labels'][f'layer_{layer1}'][k1]['label']
                label2 = data['labels'][f'layer_{layer2}'][k2]['label']
                if label1 != label2:
                    similar_pairs.append((k1, k2, overlap, label1, label2))

print(f'Found {len(similar_pairs)} inconsistent similar cluster pairs:')
for k1, k2, overlap, l1, l2 in sorted(similar_pairs[:10], key=lambda x: x[2], reverse=True):
    print(f'  {k1} ({l1}) vs {k2} ({l2}) - {overlap} common tokens')
    
# Show some example tokens
if similar_pairs:
    print("\nExample of inconsistent cluster:")
    k1, k2, overlap, l1, l2 = similar_pairs[0]
    print(f"\n{k1} labeled as '{l1}':")
    print(f"  Tokens: {', '.join(clusters[k1]['common_tokens'][:10])}")
    print(f"\n{k2} labeled as '{l2}':")
    print(f"  Tokens: {', '.join(clusters[k2]['common_tokens'][:10])}")
    
    # Show common tokens
    tokens1 = set(clusters[k1]['common_tokens'][:20])
    tokens2 = set(clusters[k2]['common_tokens'][:20])
    common = tokens1 & tokens2
    print(f"\nCommon tokens ({len(common)}): {', '.join(list(common)[:15])}")