#!/usr/bin/env python3
"""
Update existing cluster labels for consistency.
This script takes the current labels and applies consistent naming conventions.
"""

import json
from pathlib import Path
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define mapping from various label variations to consistent names
LABEL_CONSISTENCY_MAP = {
    # Function Words variations
    "Core Function Words": "Function Words: Core",
    "Function Core": "Function Words: Core",
    "Function Infrastructure": "Function Words: Infrastructure",
    "Function Network": "Function Words: Network",
    "Function Foundation": "Function Words: Foundation",
    "Function Word Core": "Function Words: Core",
    "Core Connectives": "Function Words: Connectives",
    "Grammar Network": "Function Words: Grammar Network",
    "Connective Network": "Function Words: Connectives",
    "Connective Infrastructure": "Function Words: Connectives",
    "Connective Core": "Function Words: Connectives",
    "Connective System": "Function Words: Connectives",
    
    # Grammar/Grammatical Markers variations
    "Core Grammar": "Grammar: Core",
    "Grammar Core": "Grammar: Core",
    "Grammar Infrastructure": "Grammar: Infrastructure",
    "Grammar Foundation": "Grammar: Foundation",
    "Mixed Grammar": "Grammar: Mixed Elements",
    "Grammar Support": "Grammar: Support Elements",
    "Grammatical Markers": "Grammar: Markers",
    
    # Auxiliary variations
    "Auxiliary System": "Auxiliaries: System",
    "Auxiliary Core": "Auxiliaries: Core",
    "Auxiliary Network": "Auxiliaries: Network", 
    "Auxiliary Infrastructure": "Auxiliaries: Infrastructure",
    "Auxiliary Verbs": "Auxiliaries: Primary",
    "Modal System": "Auxiliaries: Modal",
    "Modal & Prepositional": "Auxiliaries: Modal & Prepositional",
    "Be/have auxiliaries": "Auxiliaries: Be/Have",
    "Subject & Modal": "Auxiliaries: Subject & Modal",
    "Modal & Pronoun": "Auxiliaries: Modal & Pronoun",
    
    # Pronoun variations
    "Pronouns & References": "Pronouns: References",
    "Pronoun System": "Pronouns: System",
    "Pronoun Network": "Pronouns: Network",
    "Pronoun Core": "Pronouns: Core",
    "Subject Markers": "Pronouns: Subject",
    "Subject System": "Pronouns: Subject",
    "Subject Pronouns": "Pronouns: Subject",
    "Pronoun & Auxiliary Core": "Pronouns: With Auxiliaries",
    "Determiners & Possessives": "Pronouns: Possessive",
    
    # Content Words variations
    "Content Words": "Content: General",
    "Content Core": "Content: Core",
    "Content Base": "Content: Base",
    "Content Foundation": "Content: Foundation",
    "Content Vocabulary": "Content: Vocabulary",
    "General Content": "Content: General Mixed",
    "Mixed Content": "Content: Mixed",
    "Basic Content": "Content: Basic",
    "Abstract Concepts": "Content: Abstract",
    "Concrete Nouns": "Content: Concrete Nouns",
    "Human References": "Content: Human References",
    "Common Modifiers": "Content: Modifiers",
    "Spatial & Descriptive": "Content: Spatial & Descriptive",
    "Modifier Network": "Content: Modifier Network",
    
    # Punctuation variations
    "Punctuation System": "Punctuation: System",
    "Punctuation Core": "Punctuation: Core",
    "Punctuation Network": "Punctuation: Network",
    "Punctuation Infrastructure": "Punctuation: Infrastructure",
    "Punctuation & Symbols": "Punctuation: With Symbols",
    "Mixed Punctuation": "Punctuation: Mixed",
    "Punctuation & Morphology": "Punctuation: With Morphology",
    "Punctuation & Discourse": "Punctuation: With Discourse",
    "Quotes & Symbols": "Punctuation: Quotes & Symbols",
    
    # Discourse variations
    "Discourse Elements": "Discourse: Elements",
    "Discourse Markers": "Discourse: Markers",
    "Discourse System": "Discourse: System",
    "Discourse Beginnings": "Discourse: Beginnings",
    "Discourse Transitions": "Discourse: Transitions",
    "Sentence Starters": "Discourse: Sentence Starters",
    "Sentence Initiators": "Discourse: Sentence Starters",
    "Sentence Beginnings": "Discourse: Sentence Beginnings",
    
    # Morphological variations
    "Mixed Morphology": "Morphology: Mixed",
    "Morphological Patterns": "Morphology: Patterns",
    "Word Endings": "Morphology: Suffixes",
    "Articles & Morphology": "Morphology: With Articles",
    "Titles & Prefixes": "Morphology: Prefixes & Titles",
    "Prefix System": "Morphology: Prefix System",
    "Contractions & Markers": "Morphology: Contractions",
    "Negations & Contractions": "Morphology: Negations",
    "Negations": "Morphology: Negations",
    "Negation System": "Morphology: Negations",
    "Negation Forms": "Morphology: Negations",
    "Contracted Forms": "Morphology: Contractions",
    "Contracted Negations": "Morphology: Negative Contractions",
    
    # Prepositions
    "Prepositions & Auxiliaries": "Prepositions: With Auxiliaries",
    "Preposition Network": "Prepositions: Network",
    "Preposition Core": "Prepositions: Core",
    "Prepositional System": "Prepositions: System",
    
    # Comparative/Relational
    "Comparative & Relational": "Relational: Comparative",
    "Relative & Comparative": "Relational: Comparative",
    "Relational Words": "Relational: General",
    "Relational Terms": "Relational: Terms",
    "Relational System": "Relational: System",
    "Comparative Network": "Relational: Comparative Network",
    
    # Mixed categories
    "Core Determiners & Pronouns": "Mixed: Determiners & Pronouns",
    "Function Words & Connectives": "Function Words: With Connectives",
    "Pronouns & Morphology": "Mixed: Pronouns & Morphology",
    "Pronouns & Quotes": "Mixed: Pronouns & Quotes",
    "Pronouns & References": "Pronouns: With References",
    "Object & Relative": "Pronouns: Object & Relative",
    
    # Specific types that don't need change but should be in consistent format
    "Numbers & Quantifiers": "Content: Numbers & Quantifiers",
    "Proper Nouns": "Content: Proper Nouns",
    "Object Pronouns": "Pronouns: Object",
    "Personal Pronouns": "Pronouns: Personal"
}


def update_labels():
    """Update labels for consistency."""
    base_dir = Path(__file__).parent
    
    # Load current labels
    labels_path = base_dir / "llm_labels_k10" / "cluster_labels_k10.json"
    with open(labels_path, 'r') as f:
        label_data = json.load(f)
    
    # Track changes
    changes_made = defaultdict(list)
    consistency_groups = defaultdict(list)
    
    # Update labels
    for layer_key, layer_data in label_data['labels'].items():
        for cluster_key, cluster_info in layer_data.items():
            old_label = cluster_info['label']
            
            # Apply consistency mapping
            if old_label in LABEL_CONSISTENCY_MAP:
                new_label = LABEL_CONSISTENCY_MAP[old_label]
                cluster_info['label'] = new_label
                changes_made[old_label].append((cluster_key, new_label))
                
                # Track consistency groups
                primary = new_label.split(':')[0].strip()
                consistency_groups[primary].append(cluster_key)
            else:
                # If not in map, try to standardize format
                if ':' not in old_label:
                    # Try to categorize based on keywords
                    if any(word in old_label.lower() for word in ['function', 'preposition', 'conjunction']):
                        new_label = f"Function Words: {old_label}"
                    elif any(word in old_label.lower() for word in ['pronoun']):
                        new_label = f"Pronouns: {old_label}"
                    elif any(word in old_label.lower() for word in ['auxiliary', 'modal']):
                        new_label = f"Auxiliaries: {old_label}"
                    elif any(word in old_label.lower() for word in ['punctuation', 'quote', 'symbol']):
                        new_label = f"Punctuation: {old_label}"
                    elif any(word in old_label.lower() for word in ['discourse', 'sentence']):
                        new_label = f"Discourse: {old_label}"
                    elif any(word in old_label.lower() for word in ['morpho', 'prefix', 'suffix', 'contraction']):
                        new_label = f"Morphology: {old_label}"
                    elif any(word in old_label.lower() for word in ['content', 'noun', 'verb', 'modifier']):
                        new_label = f"Content: {old_label}"
                    elif any(word in old_label.lower() for word in ['grammar', 'grammatical']):
                        new_label = f"Grammar: {old_label}"
                    else:
                        new_label = f"Other: {old_label}"
                    
                    cluster_info['label'] = new_label
                    changes_made[old_label].append((cluster_key, new_label))
                    
                    primary = new_label.split(':')[0].strip()
                    consistency_groups[primary].append(cluster_key)
    
    # Add metadata about the update
    label_data['metadata']['last_updated'] = "2025-05-29T04:00:00"
    label_data['metadata']['consistency_version'] = "1.0"
    
    # Save updated labels
    output_path = base_dir / "llm_labels_k10" / "cluster_labels_k10_consistent.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(label_data, f, indent=2)
    
    logging.info(f"Saved consistent labels to {output_path}")
    
    # Also update the main file
    import shutil
    shutil.copy(output_path, labels_path)
    logging.info("Updated main labels file")
    
    # Generate change report
    report = ["LABEL CONSISTENCY UPDATE REPORT", "=" * 60, ""]
    report.append(f"Total changes made: {sum(len(v) for v in changes_made.values())}")
    report.append("")
    
    report.append("Changes by original label:")
    for old_label, changes in sorted(changes_made.items()):
        if changes:
            new_label = changes[0][1]  # All should have same new label
            report.append(f"\n'{old_label}' → '{new_label}'")
            report.append(f"  Affected clusters: {len(changes)}")
            report.append(f"  Clusters: {', '.join([c[0] for c in changes[:5]])}{'...' if len(changes) > 5 else ''}")
    
    report.append("\n\nConsistency groups:")
    for primary, clusters in sorted(consistency_groups.items(), key=lambda x: len(x[1]), reverse=True):
        report.append(f"\n{primary}: {len(clusters)} clusters")
        # Group by layer
        by_layer = defaultdict(list)
        for cluster in clusters:
            layer = int(cluster.split('_')[0][1:])
            by_layer[layer].append(cluster)
        
        for layer in sorted(by_layer.keys()):
            report.append(f"  Layer {layer}: {', '.join(by_layer[layer])}")
    
    # Save report
    report_path = base_dir / "llm_labels_k10" / "consistency_update_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logging.info(f"Saved consistency report to {report_path}")
    
    print("\nLabel consistency update complete!")
    print(f"Total changes: {sum(len(v) for v in changes_made.values())}")
    print(f"\nPrimary categories ({len(consistency_groups)}):")
    for primary, clusters in sorted(consistency_groups.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {primary}: {len(clusters)} clusters")


if __name__ == "__main__":
    update_labels()