#!/usr/bin/env python3
"""
Create consistent differentiated labels for k=5 clusters.
Same concepts get the same label across layers.
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Consistent labels for concepts that appear across layers
CONSISTENT_LABELS = {
    "layer_0": {
        "L0_C0": ("Mixed Content & Punctuation", "Articles (the), punctuation marks, and common nouns"),
        "L0_C1": ("Copulas & Prepositions", "Be-verbs (is, was), prepositions (for, with, on, by)"),
        "L0_C2": ("Articles & Conjunctions", "Determiners (a, the), conjunctions (and, that, as), core prepositions (of, to, in)"),
        "L0_C3": ("Pronouns & Short Words", "Personal pronouns (he, I, she) and very short tokens"),
        "L0_C4": ("Plural Pronouns & Time", "Plural pronouns (it, they, we) and temporal nouns (time, years)")
    },
    "layer_1": {
        "L1_C0": ("Grammar Connectives", "Articles, prepositions (of, at, from), and conjunctions"),
        "L1_C1": ("Punctuation & Capitals", "Sentence punctuation, pronouns, and capitalized words"),
        "L1_C2": ("Quotes & Titles", "Quotation marks and titles (Mr, Mrs)"),
        "L1_C3": ("Core Prepositions", "Primary prepositions (to, in, for, with, on) and pronouns"),
        "L1_C4": ("Auxiliary System", "Auxiliary verbs (was, had, has) and modal verbs (can)")
    },
    "layer_2": {
        "L2_C0": ("Punctuation & Pronouns", "Punctuation marks and personal pronouns"),
        "L2_C1": ("Prepositions & Copulas", "Prepositions and be-verbs"),
        "L2_C2": ("Content Nouns", "Common nouns (man, way, year, life, part)"),
        "L2_C3": ("Articles & Core Grammar", "Determiners and essential conjunctions"),
        "L2_C4": ("Quotes & Sentence Starters", "Quotation marks and capitalized sentence starters")
    },
    "layer_3": {
        "L3_C0": ("Conjunctions & Determiners", "Connecting words (and, that, as, or, but) and determiners"),
        "L3_C1": ("Sentence Initiators", "Capitalized sentence starters (In, There, They, If, When)"),
        "L3_C2": ("Human & Location Nouns", "Nouns for people and places"),
        "L3_C3": ("Auxiliary System", "Copular verbs and auxiliary verbs"),
        "L3_C4": ("Punctuation & Prepositions", "Sentence punctuation and basic prepositions")
    },
    "layer_4": {
        "L4_C0": ("Function Core", "Essential grammatical connectives"),
        "L4_C1": ("Abstract Nouns", "Abstract concepts and states (way, life, number)"),
        "L4_C2": ("Grammar Connectives", "Core determiners and function words"),
        "L4_C3": ("Auxiliary System", "Complete auxiliary verb system"),
        "L4_C4": ("Discourse Markers", "Quotation marks and discourse markers")
    },
    "layer_5": {
        "L5_C0": ("Grammar Connectives", "Core grammatical infrastructure"),
        "L5_C1": ("Abstract Nouns", "Abstract nouns and concepts"),
        "L5_C2": ("Discourse Markers", "Quotation and discourse elements"),
        "L5_C3": ("Auxiliary System", "Modal and auxiliary verbs"),
        "L5_C4": ("Function Core", "Essential function words")
    },
    "layer_6": {
        "L6_C0": ("Abstract Nouns", "Abstract and conceptual nouns"),
        "L6_C1": ("Auxiliary System", "Auxiliary and modal verbs"),
        "L6_C2": ("Discourse Markers", "Discourse and quotation markers"),
        "L6_C3": ("Function Core", "Core grammatical elements"),
        "L6_C4": ("Grammar Connectives", "Articles and determiners")
    },
    "layer_7": {
        "L7_C0": ("Function Core", "Core function word system"),
        "L7_C1": ("Grammar Connectives", "Grammatical connectives"),
        "L7_C2": ("Auxiliary System", "Auxiliary verb system"),
        "L7_C3": ("Discourse Markers", "Discourse markers"),
        "L7_C4": ("Abstract Nouns", "Abstract noun system")
    },
    "layer_8": {
        "L8_C0": ("Grammar Connectives", "Central grammatical elements"),
        "L8_C1": ("Auxiliary System", "Auxiliary verb network"),
        "L8_C2": ("Abstract Nouns", "Abstract concepts"),
        "L8_C3": ("Discourse Markers", "Discourse elements"),
        "L8_C4": ("Function Core", "Function word system")
    },
    "layer_9": {
        "L9_C0": ("Grammar Connectives", "Core grammar"),
        "L9_C1": ("Abstract Nouns", "Abstract concepts"),
        "L9_C2": ("Discourse Markers", "Discourse markers"),
        "L9_C3": ("Function Core", "Function words"),
        "L9_C4": ("Auxiliary System", "Auxiliary verbs")
    },
    "layer_10": {
        "L10_C0": ("Grammar Connectives", "Grammatical core"),
        "L10_C1": ("Abstract Nouns", "Abstract concepts"),
        "L10_C2": ("Discourse Markers", "Discourse elements"),
        "L10_C3": ("Function Core", "Function words"),
        "L10_C4": ("Auxiliary System", "Auxiliary system")
    },
    "layer_11": {
        "L11_C0": ("Sentence Boundaries", "Punctuation and sentence delimiters"),
        "L11_C1": ("Grammar Connectives", "Core grammatical words"),
        "L11_C2": ("Function Core", "Essential connectives"),
        "L11_C3": ("Short Forms", "Abbreviated and short forms"),
        "L11_C4": ("Morphological Elements", "Word endings and morphology")
    }
}

def create_consistent_labels():
    """Create consistent differentiated labels."""
    base_dir = Path(__file__).parent
    
    # Load full data for examples
    with open(base_dir / "llm_labels_k5" / "llm_labeling_data.json", 'r') as f:
        full_data = json.load(f)
    
    # Create new label structure
    new_labels = {
        "metadata": {
            "generated_at": "2025-05-29T00:40:00",
            "model": "claude",
            "k": 5,
            "total_clusters": 60,
            "method": "consistent_differentiated_analysis"
        },
        "labels": {}
    }
    
    # Apply consistent labels
    for layer in range(12):
        layer_key = f"layer_{layer}"
        new_labels["labels"][layer_key] = {}
        
        for cluster_idx in range(5):
            cluster_key = f"L{layer}_C{cluster_idx}"
            
            # Get the consistent label
            label, base_desc = CONSISTENT_LABELS[layer_key][cluster_key]
            
            # Get examples from full data
            cluster_data = full_data["clusters"][cluster_key]
            examples = ", ".join(cluster_data["common_tokens"][:5])
            
            new_labels["labels"][layer_key][cluster_key] = {
                "label": label,
                "description": f"{base_desc}. Examples: {examples}",
                "size": cluster_data["size"],
                "percentage": cluster_data["percentage"]
            }
    
    # Save the consistent labels
    output_path = base_dir / "llm_labels_k5" / "cluster_labels_k5.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_labels, f, indent=2)
    
    logging.info(f"Saved consistent differentiated labels to {output_path}")
    
    # Create tracking summary
    create_tracking_summary(new_labels, base_dir)


def create_tracking_summary(labels, base_dir):
    """Create a summary showing how each label tracks across layers."""
    lines = ["CONSISTENT LABEL TRACKING ACROSS LAYERS", "=" * 50, ""]
    
    # Track where each label appears
    label_tracking = {}
    
    for layer in range(12):
        layer_key = f"layer_{layer}"
        for cluster_idx in range(5):
            cluster_key = f"L{layer}_C{cluster_idx}"
            label = labels["labels"][layer_key][cluster_key]["label"]
            
            if label not in label_tracking:
                label_tracking[label] = []
            label_tracking[label].append((layer, cluster_idx))
    
    # Sort by frequency
    sorted_labels = sorted(label_tracking.items(), key=lambda x: len(x[1]), reverse=True)
    
    for label, occurrences in sorted_labels:
        lines.append(f"\n{label}:")
        lines.append(f"  Appears {len(occurrences)} times")
        
        # Group by consecutive layers
        layer_runs = []
        current_run = [occurrences[0]]
        
        for i in range(1, len(occurrences)):
            if occurrences[i][0] == occurrences[i-1][0] + 1:
                current_run.append(occurrences[i])
            else:
                layer_runs.append(current_run)
                current_run = [occurrences[i]]
        layer_runs.append(current_run)
        
        # Show runs
        for run in layer_runs:
            if len(run) > 1:
                lines.append(f"  Layers {run[0][0]}-{run[-1][0]}: " + 
                           ", ".join(f"C{r[1]}" for r in run))
            else:
                lines.append(f"  Layer {run[0][0]}: C{run[0][1]}")
    
    # Save tracking summary
    output_path = base_dir / "llm_labels_k5" / "consistent_label_tracking.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logging.info(f"Saved tracking summary to {output_path}")


if __name__ == "__main__":
    create_consistent_labels()
    print("\nConsistent labeling complete!")
    print("Key consistent labels:")
    print("  - Grammar Connectives: appears across layers")
    print("  - Function Core: consistent naming for core function words")
    print("  - Auxiliary System: consistent for auxiliary verbs")
    print("  - Abstract Nouns: consistent for abstract concepts")
    print("  - Discourse Markers: consistent for quotes/discourse")