#!/usr/bin/env python3
"""
Create consistent primary and secondary labels for k=10 clusters.
Primary labels for consistency, secondary labels for distinction.
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def analyze_tokens_for_secondary_label(tokens, primary_label):
    """Analyze tokens to determine appropriate secondary label."""
    
    clean_tokens = [t.strip().lower() for t in tokens[:30]]
    
    if primary_label == "Function Words":
        # Distinguish types of function words
        if sum(1 for t in clean_tokens if t in ["the", "a", "an", "this", "that", "these", "those"]) >= 3:
            return "Determiners"
        elif sum(1 for t in clean_tokens if t in ["and", "or", "but", "nor", "yet", "so"]) >= 2:
            return "Conjunctions"
        elif sum(1 for t in clean_tokens if t in ["to", "of", "in", "on", "at", "by", "for", "with"]) >= 3:
            return "Prepositions"
        else:
            return "Core Grammar"
    
    elif primary_label == "Content Words":
        # Distinguish types of content words
        # Check for specific patterns
        if any(t in clean_tokens for t in ["time", "day", "year", "hour", "minute", "week", "month"]):
            return "Temporal"
        elif any(t in clean_tokens for t in ["place", "location", "area", "region", "city", "country"]):
            return "Spatial"
        elif any(t in clean_tokens for t in ["man", "woman", "people", "person", "child", "human"]):
            return "Human/Social"
        elif any(t in clean_tokens for t in ["mind", "thought", "idea", "feeling", "sense", "power"]):
            return "Abstract Concepts"
        elif any(t in clean_tokens for t in ["hand", "head", "body", "eye", "face", "water", "air"]):
            return "Physical/Concrete"
        elif sum(1 for t in tokens if t.strip().endswith("ing")) >= 5:
            return "Action/Process"
        else:
            return "General"
    
    elif primary_label == "Pronouns":
        # Distinguish types of pronouns
        if sum(1 for t in clean_tokens if t in ["i", "you", "he", "she", "it", "we", "they"]) >= 3:
            return "Personal"
        elif sum(1 for t in clean_tokens if t in ["my", "your", "his", "her", "its", "our", "their"]) >= 2:
            return "Possessive"
        elif sum(1 for t in clean_tokens if t in ["which", "that", "who", "whom", "whose"]) >= 2:
            return "Relative"
        else:
            return "Mixed"
    
    elif primary_label == "Punctuation":
        # Distinguish types of punctuation patterns
        if "." in tokens[:5] or "?" in tokens[:5] or "!" in tokens[:5]:
            return "Sentence-Final"
        elif "," in tokens[:5]:
            return "Comma-Dominated"
        elif any(t in tokens[:10] for t in ["(", ")", "[", "]", "{", "}"]):
            return "Brackets/Parens"
        elif any(t in tokens[:10] for t in ["'", '"', "``", "''"]):
            return "Quotes"
        else:
            return "Mixed Punctuation"
    
    elif primary_label == "Auxiliaries":
        # Distinguish types of auxiliaries
        if sum(1 for t in clean_tokens if t in ["is", "are", "was", "were", "be", "been", "being"]) >= 3:
            return "Be-Forms"
        elif sum(1 for t in clean_tokens if t in ["have", "has", "had", "having"]) >= 2:
            return "Have-Forms"
        elif sum(1 for t in clean_tokens if t in ["will", "would", "can", "could", "may", "might", "shall", "should"]) >= 2:
            return "Modals"
        else:
            return "Mixed Auxiliary"
    
    elif primary_label == "Short Tokens":
        # Distinguish types of short tokens
        if sum(1 for t in tokens[:10] if len(t.strip()) == 1) >= 5:
            return "Single Chars"
        elif sum(1 for t in tokens[:10] if t.strip().endswith("'s") or t.strip().endswith("n't")) >= 2:
            return "Contractions"
        else:
            return "Fragments"
    
    elif primary_label == "Numbers":
        return "Numeric"
    
    else:
        return "Other"


def create_consistent_labels_with_secondary():
    base_dir = Path(__file__).parent
    
    # Load cluster data
    with open(base_dir / "llm_labels_k10" / "llm_labeling_data.json", 'r') as f:
        cluster_data = json.load(f)
    
    # First, assign initial primary labels
    initial_labels = {}
    
    for cluster_key, data in cluster_data["clusters"].items():
        tokens = [t.strip().lower() for t in data["common_tokens"][:30]]
        
        # Simple primary categorization
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
    
    # Find similar clusters and ensure primary label consistency
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
    
    # For each group, use alphabetically first primary label
    primary_labels = initial_labels.copy()
    
    for group in cluster_groups:
        group_labels = [initial_labels[k] for k in group]
        consistent_label = sorted(group_labels)[0]
        for k in group:
            primary_labels[k] = consistent_label
    
    # Now assign secondary labels
    final_labels = {}
    
    for cluster_key, primary in primary_labels.items():
        tokens = cluster_data["clusters"][cluster_key]["common_tokens"]
        secondary = analyze_tokens_for_secondary_label(tokens, primary)
        final_labels[cluster_key] = {
            "primary": primary,
            "secondary": secondary,
            "full_label": f"{primary} ({secondary})"
        }
    
    return final_labels


def main():
    base_dir = Path(__file__).parent
    
    # Get labels with primary and secondary
    labels = create_consistent_labels_with_secondary()
    
    # Load cluster data for metadata
    with open(base_dir / "llm_labels_k10" / "llm_labeling_data.json", 'r') as f:
        cluster_data = json.load(f)
    
    # Create results
    results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "primary_secondary_labeling",
            "k": 10,
            "total_clusters": 120,
            "method": "consistent_primary_with_distinctive_secondary",
            "description": "Primary labels for consistency, secondary labels for distinction"
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
                label_info = labels[cluster_key]
                
                results["labels"][layer_key][cluster_key] = {
                    "label": label_info["full_label"],
                    "primary_label": label_info["primary"],
                    "secondary_label": label_info["secondary"],
                    "description": f"Cluster containing tokens like: {', '.join(repr(t) for t in cluster_info['common_tokens'][:5])}",
                    "size": cluster_info["size"],
                    "percentage": cluster_info["percentage"]
                }
    
    # Save
    output_path = base_dir / "llm_labels_k10" / "cluster_labels_k10.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    primary_counts = defaultdict(int)
    secondary_counts = defaultdict(int)
    full_counts = defaultdict(int)
    
    for label_data in labels.values():
        primary_counts[label_data["primary"]] += 1
        secondary_counts[f"{label_data['primary']} - {label_data['secondary']}"] += 1
        full_counts[label_data["full_label"]] += 1
    
    print(f"Saved labels to {output_path}")
    print("\nPrimary Label Distribution:")
    for label, count in sorted(primary_counts.items()):
        print(f"  {label}: {count} clusters")
    
    print(f"\nTotal unique primary labels: {len(primary_counts)}")
    print(f"Total unique full labels: {len(full_counts)}")
    
    # Show some examples of secondary distinctions
    print("\nExample Secondary Distinctions for Content Words:")
    content_secondaries = [k for k in secondary_counts.keys() if k.startswith("Content Words")]
    for label in sorted(content_secondaries)[:5]:
        print(f"  {label}: {secondary_counts[label]} clusters")


if __name__ == "__main__":
    main()