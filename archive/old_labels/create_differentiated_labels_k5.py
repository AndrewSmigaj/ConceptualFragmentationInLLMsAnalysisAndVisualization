#!/usr/bin/env python3
"""
Create differentiated labels for k=5 clusters based on their unique characteristics.
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Based on the detailed analysis, here are the differentiated labels
DIFFERENTIATED_LABELS = {
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
        "L1_C4": ("Auxiliaries & Modals", "Auxiliary verbs (was, had, has) and modal verbs (can)")
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
        "L3_C3": ("Be-verbs & Auxiliaries", "Copular verbs and auxiliary verbs"),
        "L3_C4": ("Punctuation & Prepositions", "Sentence punctuation and basic prepositions")
    },
    "layer_4": {
        "L4_C0": ("Core Grammar Words", "Essential grammatical connectives"),
        "L4_C1": ("Abstract Nouns", "Abstract concepts and states (way, life, number)"),
        "L4_C2": ("Articles & Basic Grammar", "Core determiners and function words"),
        "L4_C3": ("Auxiliary Verb System", "Complete auxiliary verb system"),
        "L4_C4": ("Quotes & Discourse", "Quotation marks and discourse markers")
    },
    "layer_5": {
        "L5_C0": ("Grammar Foundation", "Core grammatical infrastructure"),
        "L5_C1": ("Abstract Concepts", "Abstract nouns and concepts"),
        "L5_C2": ("Discourse Markers", "Quotation and discourse elements"),
        "L5_C3": ("Auxiliary System", "Modal and auxiliary verbs"),
        "L5_C4": ("Function Word Core", "Essential function words")
    },
    "layer_6": {
        "L6_C0": ("Conceptual Nouns", "Abstract and conceptual nouns"),
        "L6_C1": ("Verb Auxiliaries", "Auxiliary and modal verbs"),
        "L6_C2": ("Discourse Elements", "Discourse and quotation markers"),
        "L6_C3": ("Grammar Core", "Core grammatical elements"),
        "L6_C4": ("Determiner System", "Articles and determiners")
    },
    "layer_7": {
        "L7_C0": ("Function Infrastructure", "Core function word system"),
        "L7_C1": ("Grammar Network", "Grammatical connectives"),
        "L7_C2": ("Auxiliary Network", "Auxiliary verb system"),
        "L7_C3": ("Discourse System", "Discourse markers"),
        "L7_C4": ("Noun Concepts", "Abstract noun system")
    },
    "layer_8": {
        "L8_C0": ("Grammar Core", "Central grammatical elements"),
        "L8_C1": ("Auxiliary Verbs", "Auxiliary verb network"),
        "L8_C2": ("Conceptual Nouns", "Abstract concepts"),
        "L8_C3": ("Discourse Markers", "Discourse elements"),
        "L8_C4": ("Function System", "Function word system")
    },
    "layer_9": {
        "L9_C0": ("Grammar Foundation", "Core grammar"),
        "L9_C1": ("Abstract Nouns", "Abstract concepts"),
        "L9_C2": ("Discourse Elements", "Discourse markers"),
        "L9_C3": ("Function Core", "Function words"),
        "L9_C4": ("Auxiliary System", "Auxiliary verbs")
    },
    "layer_10": {
        "L10_C0": ("Grammar Network", "Grammatical core"),
        "L10_C1": ("Conceptual System", "Abstract concepts"),
        "L10_C2": ("Discourse Markers", "Discourse elements"),
        "L10_C3": ("Function Foundation", "Function words"),
        "L10_C4": ("Auxiliary Network", "Auxiliary system")
    },
    "layer_11": {
        "L11_C0": ("Sentence Boundaries", "Punctuation and sentence delimiters"),
        "L11_C1": ("Grammar Elements", "Core grammatical words"),
        "L11_C2": ("Core Connectives", "Essential connectives"),
        "L11_C3": ("Short Forms", "Abbreviated and short forms"),
        "L11_C4": ("Morphological Elements", "Word endings and morphology")
    }
}

def create_differentiated_labels():
    """Create the new differentiated labels."""
    base_dir = Path(__file__).parent
    
    # Load existing label structure
    with open(base_dir / "llm_labels_k5" / "cluster_labels_k5.json", 'r') as f:
        existing = json.load(f)
    
    # Load full data for examples
    with open(base_dir / "llm_labels_k5" / "llm_labeling_data.json", 'r') as f:
        full_data = json.load(f)
    
    # Create new label structure
    new_labels = {
        "metadata": {
            "generated_at": "2025-05-29T00:30:00",
            "model": "claude",
            "k": 5,
            "total_clusters": 60,
            "method": "differentiated_analysis"
        },
        "labels": {}
    }
    
    # Apply differentiated labels
    for layer in range(12):
        layer_key = f"layer_{layer}"
        new_labels["labels"][layer_key] = {}
        
        for cluster_idx in range(5):
            cluster_key = f"L{layer}_C{cluster_idx}"
            
            # Get the differentiated label
            label, base_desc = DIFFERENTIATED_LABELS[layer_key][cluster_key]
            
            # Get examples from full data
            cluster_data = full_data["clusters"][cluster_key]
            examples = ", ".join(cluster_data["common_tokens"][:5])
            
            new_labels["labels"][layer_key][cluster_key] = {
                "label": label,
                "description": f"{base_desc}. Examples: {examples}",
                "size": cluster_data["size"],
                "percentage": cluster_data["percentage"]
            }
    
    # Save the differentiated labels
    output_path = base_dir / "llm_labels_k5" / "cluster_labels_k5.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_labels, f, indent=2)
    
    logging.info(f"Saved differentiated labels to {output_path}")
    
    # Create summary
    create_summary(new_labels, base_dir)


def create_summary(labels, base_dir):
    """Create a summary of differentiated labels."""
    lines = ["DIFFERENTIATED K=5 CLUSTER LABELS", "=" * 50, ""]
    
    for layer in range(12):
        layer_key = f"layer_{layer}"
        lines.append(f"\nLAYER {layer}:")
        lines.append("-" * 30)
        
        for cluster_idx in range(5):
            cluster_key = f"L{layer}_C{cluster_idx}"
            cluster_info = labels["labels"][layer_key][cluster_key]
            
            label = cluster_info["label"]
            pct = cluster_info["percentage"]
            
            lines.append(f"  C{cluster_idx}: {label} ({pct:.1f}%)")
    
    # Save summary
    output_path = base_dir / "llm_labels_k5" / "differentiated_labels_summary.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logging.info(f"Saved summary to {output_path}")


if __name__ == "__main__":
    create_differentiated_labels()
    print("\nDifferentiated labeling complete!")
    print("Labels have been updated to show unique characteristics of each cluster")