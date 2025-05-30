#!/usr/bin/env python3
"""
Create accurate semantic labels for k=5 clusters based on actual token content.
Clusters can have the same label across layers if they serve similar functions.
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def analyze_tokens_carefully(tokens, full_token_list=None):
    """Carefully analyze what linguistic role these tokens play."""
    # Clean tokens
    clean_tokens = [t.strip() for t in tokens[:20]]  # Look at more tokens
    
    # Key linguistic categories with examples
    categories = {
        'core_function': {
            'examples': ['the', 'a', 'of', 'to', 'in', 'and', 'that', 'is', 'was', 'for', 'as', 'with', 'be', 'at', 'by', 'on'],
            'label': 'Core Function Words',
            'desc': 'Articles, prepositions, conjunctions, and copulas'
        },
        'pronouns': {
            'examples': ['I', 'he', 'she', 'it', 'we', 'they', 'you', 'me', 'him', 'her', 'us', 'them', 'my', 'his', 'her', 'their', 'our', 'your'],
            'label': 'Pronouns',
            'desc': 'Personal, possessive, and object pronouns'
        },
        'auxiliaries': {
            'examples': ['have', 'has', 'had', 'will', 'would', 'can', 'could', 'may', 'might', 'shall', 'should', 'must', 'do', 'does', 'did', 'been', 'being'],
            'label': 'Auxiliary Verbs',
            'desc': 'Modal and auxiliary verbs'
        },
        'content_verbs': {
            'examples': ['said', 'get', 'make', 'go', 'know', 'take', 'see', 'come', 'think', 'look', 'want', 'give', 'use', 'find', 'tell', 'ask', 'work', 'say', 'made', 'went'],
            'label': 'Content Verbs',
            'desc': 'Main action and state verbs'
        },
        'content_nouns': {
            'examples': ['time', 'people', 'year', 'way', 'day', 'man', 'thing', 'woman', 'life', 'child', 'world', 'school', 'state', 'family', 'group', 'country', 'problem', 'fact', 'hand', 'part'],
            'label': 'Content Nouns',
            'desc': 'Common nouns referring to people, places, things, and concepts'
        },
        'punctuation': {
            'examples': ['.', ',', ';', ':', '!', '?', '-', '--', '(', ')', '"', "'", '``', "''", '...'],
            'label': 'Punctuation',
            'desc': 'Punctuation marks and quotation markers'
        },
        'morphology': {
            'examples': ['ing', 'ed', 'ly', 's', "'s", "n't", "'t", "'ll", "'ve", "'re", "'d", 'er', 'est', 'tion'],
            'label': 'Morphological Suffixes',
            'desc': 'Common suffixes and morphological endings'
        }
    }
    
    # Score each category
    scores = {}
    for cat_name, cat_info in categories.items():
        score = 0
        for token in clean_tokens:
            token_lower = token.lower()
            
            if cat_name == 'morphology':
                # Check if it's a suffix/morpheme
                if token_lower in cat_info['examples'] or (len(token_lower) <= 3 and token_lower.endswith(('ing', 'ed', 'ly', 's', 'er', 'est'))):
                    score += 2
            elif token_lower in cat_info['examples']:
                score += 2
            # Partial matches for content words
            elif cat_name in ['content_nouns', 'content_verbs']:
                # Check if it looks like a content word
                if len(token_lower) > 3 and token_lower.isalpha():
                    score += 0.5
        
        if score > 0:
            scores[cat_name] = score
    
    # Find best match
    if not scores:
        # Default fallback - look at token characteristics
        if any(len(t.strip()) <= 2 for t in clean_tokens[:5]):
            return 'morphology', categories['morphology']['label'], categories['morphology']['desc']
        else:
            return 'mixed', 'Mixed Lexical Items', 'Various word types and forms'
    
    best_category = max(scores.items(), key=lambda x: x[1])[0]
    
    # Check for mixed categories
    if len(scores) >= 3 and max(scores.values()) < 5:
        return 'mixed', 'Mixed Lexical Items', 'Various word types and forms'
    
    return best_category, categories[best_category]['label'], categories[best_category]['desc']


def create_accurate_labels():
    """Create accurate labels for each cluster based on actual content."""
    base_dir = Path(__file__).parent
    
    # Load data
    with open(base_dir / "llm_labels_k5" / "llm_labeling_data.json", 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # New labels structure
    new_labels = {
        "metadata": {
            "generated_at": "2025-05-29T00:00:00",
            "model": "claude",
            "k": 5,
            "total_clusters": 60,
            "method": "accurate_linguistic_analysis"
        },
        "labels": {}
    }
    
    # Analyze each cluster individually
    for layer in range(12):
        layer_key = f"layer_{layer}"
        new_labels["labels"][layer_key] = {}
        
        for cluster_idx in range(5):
            cluster_key = f"L{layer}_C{cluster_idx}"
            cluster_data = full_data["clusters"][cluster_key]
            
            # Get tokens
            tokens = cluster_data.get("common_tokens", [])
            
            # Analyze this specific cluster
            category, label, description = analyze_tokens_carefully(tokens)
            
            # Create label with examples
            examples = ", ".join(tokens[:5])
            
            new_labels["labels"][layer_key][cluster_key] = {
                "label": label,
                "description": f"{description}. Examples: {examples}",
                "category": category,
                "size": cluster_data["size"],
                "percentage": cluster_data["percentage"]
            }
            
            logging.info(f"{cluster_key}: {label} ({category}) - {tokens[:5]}")
    
    # Save the accurate labels
    output_path = base_dir / "llm_labels_k5" / "cluster_labels_k5_accurate.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_labels, f, indent=2)
    
    # Also overwrite the original file
    original_path = base_dir / "llm_labels_k5" / "cluster_labels_k5.json"
    with open(original_path, 'w', encoding='utf-8') as f:
        json.dump(new_labels, f, indent=2)
    
    logging.info(f"Saved accurate labels to {output_path}")
    logging.info(f"Updated original labels at {original_path}")
    
    # Create summary
    create_label_summary(new_labels, base_dir)


def create_label_summary(labels, base_dir):
    """Create a summary of the accurate labels."""
    lines = ["ACCURATE K=5 CLUSTER LABELS", "=" * 50, ""]
    
    # Count label occurrences
    label_counts = {}
    
    for layer in range(12):
        layer_key = f"layer_{layer}"
        lines.append(f"\nLAYER {layer}:")
        lines.append("-" * 30)
        
        for cluster_idx in range(5):
            cluster_key = f"L{layer}_C{cluster_idx}"
            cluster_info = labels["labels"][layer_key][cluster_key]
            
            label = cluster_info["label"]
            size_pct = cluster_info["percentage"]
            
            lines.append(f"  C{cluster_idx}: {label} ({size_pct:.1f}%)")
            
            # Track label frequency
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
    
    # Add summary statistics
    lines.append("\n\nLABEL FREQUENCY ACROSS LAYERS:")
    lines.append("-" * 30)
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {label}: appears in {count} clusters")
    
    # Save summary
    output_path = base_dir / "llm_labels_k5" / "accurate_labels_summary.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logging.info(f"Saved summary to {output_path}")


if __name__ == "__main__":
    create_accurate_labels()
    print("\nAccurate labeling complete!")
    print("Labels have been updated in cluster_labels_k5.json")