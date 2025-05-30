#!/usr/bin/env python3
"""
Create consistent labels for k=10 clusters using alphabetical consistency.
When clusters share tokens, they get the same label (alphabetically first).
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def create_consistent_labels():
    base_dir = Path(__file__).parent
    
    # Load cluster data
    with open(base_dir / "llm_labels_k10" / "llm_labeling_data.json", 'r') as f:
        cluster_data = json.load(f)
    
    # First, assign initial labels based on token analysis
    initial_labels = {}
    
    for cluster_key, data in cluster_data["clusters"].items():
        tokens = [t.strip().lower() for t in data["common_tokens"][:30]]
        
        # Simple categorization
        if any(t in [".", ",", "?", "!", ";", ":"] for t in tokens[:10]):
            label = "Punctuation"
        elif sum(1 for t in tokens[:10] if t in ["the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with"]) >= 5:
            label = "Function Words"
        elif sum(1 for t in tokens[:10] if t in ["he", "she", "it", "they", "we", "i", "you", "me", "him", "her", "us", "them"]) >= 3:
            label = "Pronouns"
        elif sum(1 for t in tokens[:10] if t in ["is", "was", "are", "were", "be", "been", "being", "have", "has", "had"]) >= 3:
            label = "Auxiliaries"
        elif sum(1 for t in tokens[:10] if t.isdigit()) >= 2:
            label = "Numbers"
        elif sum(1 for t in tokens[:10] if len(t) <= 2) >= 5:
            label = "Short Tokens"
        else:
            label = "Content Words"
        
        initial_labels[cluster_key] = label
    
    # Now find similar clusters and ensure they have consistent labels
    # Group clusters with >50% token overlap
    cluster_groups = []
    processed = set()
    
    for k1, d1 in cluster_data["clusters"].items():
        if k1 in processed:
            continue
        
        group = [k1]
        tokens1 = set(d1["common_tokens"][:20])
        
        for k2, d2 in cluster_data["clusters"].items():
            if k2 != k1 and k2 not in processed:
                tokens2 = set(d2["common_tokens"][:20])
                overlap = len(tokens1 & tokens2) / min(len(tokens1), len(tokens2))
                
                if overlap > 0.5:  # 50% similarity
                    group.append(k2)
        
        if len(group) > 1:
            cluster_groups.append(group)
            processed.update(group)
    
    # For each group, use the alphabetically first label
    final_labels = initial_labels.copy()
    
    for group in cluster_groups:
        # Get all labels in the group
        group_labels = [initial_labels[k] for k in group]
        # Pick the alphabetically first one
        consistent_label = sorted(group_labels)[0]
        # Apply to all clusters in group
        for k in group:
            final_labels[k] = consistent_label
    
    return final_labels


def main():
    base_dir = Path(__file__).parent
    
    # Get consistent labels
    labels = create_consistent_labels()
    
    # Load cluster data for metadata
    with open(base_dir / "llm_labels_k10" / "llm_labeling_data.json", 'r') as f:
        cluster_data = json.load(f)
    
    # Create results
    results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "alphabetical_consistency",
            "k": 10,
            "total_clusters": 120,
            "method": "alphabetical_consistent_labeling",
            "description": "Labels assigned with alphabetical consistency for similar clusters"
        },
        "labels": {}
    }
    
    # Organize by layer
    for layer in range(12):
        layer_key = f"layer_{layer}"
        results["labels"][layer_key] = {}
        
        for cluster_idx in range(10):
            cluster_key = f"L{layer}_C{cluster_idx}"
            
            if cluster_key in labels and cluster_key in cluster_data["clusters"]:
                cluster_info = cluster_data["clusters"][cluster_key]
                results["labels"][layer_key][cluster_key] = {
                    "label": labels[cluster_key],
                    "description": f"Cluster containing tokens like: {', '.join(repr(t) for t in cluster_info['common_tokens'][:5])}",
                    "size": cluster_info["size"],
                    "percentage": cluster_info["percentage"]
                }
    
    # Save
    output_path = base_dir / "llm_labels_k10" / "cluster_labels_k10.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    label_counts = defaultdict(int)
    for label in labels.values():
        label_counts[label] += 1
    
    print(f"Saved consistent labels to {output_path}")
    print("\nLabel Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} clusters")
    
    print(f"\nTotal unique labels: {len(label_counts)}")


if __name__ == "__main__":
    main()