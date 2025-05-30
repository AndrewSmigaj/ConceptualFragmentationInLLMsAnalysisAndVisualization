#!/usr/bin/env python3
"""
Generate consistent labels for k=10 clusters using deterministic prompt engineering.
This replaces random labeling with systematic rules to ensure the same tokens
get the same labels across layers.
"""

import json
from pathlib import Path
import logging
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Deterministic labeling prompt template
LABELING_PROMPT_TEMPLATE = """
Analyze this cluster and provide a consistent label.

Cluster: {cluster_id}
Layer: {layer}
Size: {size} tokens
Top 30 tokens: {tokens}

CRITICAL: Use this EXACT process for consistency:

Step 1: Categorize using this STRICT priority order
(Check each rule in order. Use the FIRST one where >50% of top 10 tokens match):

1. If >50% are punctuation marks (. , ; : ! ? - '' `` " ') → "Punctuation"
2. If >50% are pronouns (I, you, he, she, it, we, they, me, him, her, us, them, my, your, his, her, its, our, their) → "Pronouns"
3. If >50% are articles/determiners (a, an, the, this, that, these, those, A, An, The) → "Determiners"
4. If >50% are prepositions (in, on, at, to, for, with, by, from, of, about, under, over, through, between, among) → "Prepositions"
5. If >50% are conjunctions (and, or, but, so, yet, nor, And, Or, But) → "Conjunctions"
6. If >50% are auxiliary/modal verbs (is, are, was, were, be, been, being, have, has, had, will, would, should, could, can, may, might, must, shall) → "Auxiliaries"
7. If >50% are content words AND mostly concrete objects → "Content"
8. If >50% are content words AND mostly abstract concepts → "Abstract"
9. If >50% are numbers or quantifiers → "Quantifiers"
10. If contains mix of prefixes/suffixes (-ing, -ed, -ly, -tion, un-, re-, pre-) → "Morphological"
11. If >50% are capitalized words → "Capitalized"
12. If none of above apply → "Mixed"

Step 2: Choose subcategory - use ONLY these terms:
- "Primary" (if >70% of tokens strongly match the category)
- "Secondary" (if 50-70% match the category)  
- "Combined" (if cluster combines exactly 2 categories equally)
- "Special" (if cluster has unique pattern not fitting above)

Step 3: Format
Return ONLY a JSON object with this exact structure:
{{
  "primary_category": "[chosen category from Step 1]",
  "subcategory": "[Primary/Secondary/Combined/Special]",
  "confidence": [0-100 score for how well tokens match the category],
  "reasoning": "[One sentence explaining the categorization]"
}}

Remember: Be consistent! If you see words like "to, in, is, for, with" together, they should ALWAYS be categorized the same way.
"""

def analyze_cluster_with_llm(cluster_id, layer, tokens, size):
    """
    Simulate LLM analysis using deterministic rules.
    In production, this would call the actual LLM API.
    """
    # For now, implement the deterministic rules directly
    # This ensures consistency while demonstrating the approach
    
    # Extract just the token text, removing leading spaces
    clean_tokens = [t.strip() for t in tokens[:10]]
    
    # Count matches for each category
    categories = {
        "Punctuation": 0,
        "Pronouns": 0,
        "Determiners": 0,
        "Prepositions": 0,
        "Conjunctions": 0,
        "Auxiliaries": 0,
        "Content": 0,
        "Abstract": 0,
        "Quantifiers": 0,
        "Morphological": 0,
        "Capitalized": 0
    }
    
    # Category word lists
    punctuation = {'.', ',', ';', ':', '!', '?', '-', "''", '``', '"', "'", '(', ')', '[', ']'}
    pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
                'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves'}
    determiners = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'A', 'An', 'The'}
    prepositions = {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 
                    'under', 'over', 'through', 'between', 'among', 'into', 'onto', 'upon'}
    conjunctions = {'and', 'or', 'but', 'so', 'yet', 'nor', 'And', 'Or', 'But'}
    auxiliaries = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                   'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall'}
    quantifiers = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                   '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'many', 'few', 'several', 'all', 'some'}
    
    # Count matches
    for token in clean_tokens:
        token_lower = token.lower()
        
        if token in punctuation or any(c in punctuation for c in token):
            categories["Punctuation"] += 1
        elif token_lower in pronouns:
            categories["Pronouns"] += 1
        elif token_lower in determiners:
            categories["Determiners"] += 1
        elif token_lower in prepositions:
            categories["Prepositions"] += 1
        elif token_lower in conjunctions:
            categories["Conjunctions"] += 1
        elif token_lower in auxiliaries:
            categories["Auxiliaries"] += 1
        elif token_lower in quantifiers or token.isdigit():
            categories["Quantifiers"] += 1
        elif any(token.endswith(suf) for suf in ['ing', 'ed', 'ly', 'tion', 'ment', 'ness']):
            categories["Morphological"] += 1
        elif token[0].isupper() if token else False:
            categories["Capitalized"] += 1
        else:
            # Default to content
            categories["Content"] += 1
    
    # Find primary category (first one with >50% match)
    primary_category = "Mixed"
    max_count = 0
    for cat, count in categories.items():
        if count > 5:  # >50% of top 10
            primary_category = cat
            max_count = count
            break
    
    # If no clear majority, pick the highest
    if primary_category == "Mixed":
        primary_category = max(categories.items(), key=lambda x: x[1])[0]
        max_count = categories[primary_category]
    
    # Determine subcategory
    confidence = (max_count / 10) * 100
    if confidence >= 70:
        subcategory = "Primary"
    elif confidence >= 50:
        subcategory = "Secondary"
    else:
        # Check if it's a combination of two categories
        top_two = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:2]
        if top_two[0][1] == top_two[1][1] and top_two[0][1] >= 3:
            subcategory = "Combined"
        else:
            subcategory = "Special"
    
    return {
        "primary_category": primary_category,
        "subcategory": subcategory,
        "confidence": confidence,
        "reasoning": f"Cluster contains {max_count}/10 {primary_category.lower()} tokens"
    }


def generate_consistent_labels():
    """Generate consistent labels for all k=10 clusters."""
    base_dir = Path(__file__).parent
    
    # Load cluster data
    with open(base_dir / "llm_labels_k10" / "llm_labeling_data.json", 'r') as f:
        cluster_data = json.load(f)
    
    # Track label assignments for consistency checking
    label_history = defaultdict(list)
    
    # Create new label structure
    new_labels = {
        "metadata": {
            "generated_at": "2025-05-29T10:00:00",
            "model": "deterministic_rules",
            "k": 10,
            "total_clusters": 120,
            "method": "consistent_labeling_v2",
            "description": "Labels generated using deterministic rules for consistency"
        },
        "labels": {}
    }
    
    # Process each layer
    for layer in range(12):
        layer_key = f"layer_{layer}"
        new_labels["labels"][layer_key] = {}
        
        logging.info(f"Processing layer {layer}...")
        
        for cluster_idx in range(10):
            cluster_key = f"L{layer}_C{cluster_idx}"
            
            if cluster_key in cluster_data["clusters"]:
                cluster_info = cluster_data["clusters"][cluster_key]
                tokens = cluster_info["common_tokens"]
                size = cluster_info["size"]
                
                # Analyze cluster
                analysis = analyze_cluster_with_llm(cluster_key, layer, tokens, size)
                
                # Create label
                if analysis["subcategory"] == "Primary":
                    label = f"{analysis['primary_category']}"
                else:
                    label = f"{analysis['primary_category']}: {analysis['subcategory']}"
                
                # Track similar token groups
                token_set = frozenset(tokens[:10])
                label_history[token_set].append({
                    "layer": layer,
                    "cluster": cluster_idx,
                    "label": label
                })
                
                # Create description with examples
                examples = ", ".join(tokens[:5])
                description = f"{analysis['reasoning']}. Examples: {examples}"
                
                new_labels["labels"][layer_key][cluster_key] = {
                    "label": label,
                    "description": description,
                    "size": size,
                    "percentage": cluster_info["percentage"],
                    "analysis": analysis
                }
    
    # Save the new consistent labels
    output_path = base_dir / "llm_labels_k10" / "cluster_labels_k10_consistent_v2.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_labels, f, indent=2)
    
    logging.info(f"Saved consistent labels to {output_path}")
    
    # Generate consistency report
    report = ["CONSISTENT LABELING REPORT", "=" * 60, ""]
    report.append("Label Distribution:")
    
    label_counts = Counter()
    for layer_data in new_labels["labels"].values():
        for cluster_data in layer_data.values():
            label_counts[cluster_data["label"]] += 1
    
    for label, count in label_counts.most_common():
        report.append(f"  {label}: {count} clusters")
    
    report.append("\n\nCross-layer consistency:")
    report.append("(Token groups that appear in multiple layers)")
    
    consistent_groups = 0
    inconsistent_groups = 0
    
    for token_set, occurrences in label_history.items():
        if len(occurrences) > 1:
            labels = [occ["label"] for occ in occurrences]
            if len(set(labels)) == 1:
                consistent_groups += 1
            else:
                inconsistent_groups += 1
                if inconsistent_groups <= 5:  # Show first 5 inconsistencies
                    report.append(f"\nInconsistent group:")
                    report.append(f"  Tokens: {', '.join(list(token_set)[:5])}")
                    for occ in occurrences:
                        report.append(f"    L{occ['layer']}_C{occ['cluster']}: {occ['label']}")
    
    report.append(f"\n\nTotal token groups appearing multiple times: {consistent_groups + inconsistent_groups}")
    report.append(f"Consistent labeling: {consistent_groups}")
    report.append(f"Inconsistent labeling: {inconsistent_groups}")
    
    consistency_rate = (consistent_groups / (consistent_groups + inconsistent_groups) * 100) if (consistent_groups + inconsistent_groups) > 0 else 0
    report.append(f"Consistency rate: {consistency_rate:.1f}%")
    
    # Save report
    report_path = base_dir / "llm_labels_k10" / "consistent_labeling_report_v2.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logging.info(f"Saved consistency report to {report_path}")
    
    # Update the main labels file
    import shutil
    shutil.copy(output_path, base_dir / "llm_labels_k10" / "cluster_labels_k10.json")
    logging.info("Updated main labels file")
    
    print("\nConsistent labeling complete!")
    print(f"Consistency rate: {consistency_rate:.1f}%")
    print("\nThe new labeling system uses deterministic rules to ensure")
    print("the same types of tokens always get the same label.")


if __name__ == "__main__":
    generate_consistent_labels()