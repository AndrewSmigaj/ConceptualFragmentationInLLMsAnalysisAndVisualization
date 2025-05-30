#!/usr/bin/env python3
"""
Create consistent semantic labels for k=5 clusters based on careful token analysis.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ConsistentLabeler:
    def __init__(self):
        # Define linguistic categories more carefully
        self.categories = {
            'determiners': ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'my', 'your', 'our'],
            'pronouns': ['I', 'he', 'she', 'it', 'we', 'they', 'you', 'me', 'him', 'her', 'us', 'them'],
            'auxiliaries': ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'will', 'would', 'can', 'could', 'may', 'might', 'shall', 'should', 'do', 'does', 'did'],
            'prepositions': ['of', 'to', 'in', 'for', 'with', 'on', 'at', 'by', 'from', 'up', 'about', 'into', 'through', 'after', 'over', 'between', 'out', 'against', 'during', 'without'],
            'conjunctions': ['and', 'or', 'but', 'that', 'as', 'if', 'when', 'than', 'because', 'while', 'where', 'so', 'though', 'since', 'whether'],
            'punctuation': ['.', ',', ';', ':', '!', '?', '-', '--', '(', ')', '[', ']', '{', '}', '"', "'", '``', "''", '...'],
            'morphemes': ['ing', 'ed', 'ly', 's', "'s", "n't", "'t", "'ve", "'re", "'ll", "'d", 'er', 'est', 'tion', 'ness', 'ment'],
            'content_nouns': ['time', 'people', 'year', 'way', 'day', 'man', 'thing', 'woman', 'life', 'child', 'world', 'school', 'state', 'family', 'student', 'group', 'country', 'problem'],
            'action_verbs': ['said', 'get', 'make', 'go', 'know', 'take', 'see', 'come', 'think', 'look', 'want', 'give', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call']
        }
        
    def analyze_cluster_comprehensive(self, tokens, type_distribution):
        """Comprehensively analyze a cluster's linguistic characteristics."""
        # Clean tokens (remove leading spaces)
        clean_tokens = [t.strip() for t in tokens]
        
        # Count matches for each category
        category_scores = defaultdict(int)
        
        for token in clean_tokens[:20]:  # Look at more tokens for better analysis
            token_lower = token.lower()
            
            for category, examples in self.categories.items():
                if token_lower in examples:
                    category_scores[category] += 1
                # Check for morpheme matches
                elif category == 'morphemes':
                    for morpheme in examples:
                        if token_lower.endswith(morpheme) or token_lower.startswith(morpheme):
                            category_scores[category] += 1
                            break
        
        # Also consider the token type distribution
        if 'punctuation' in type_distribution and type_distribution.get('punctuation', 0) > 5:
            category_scores['punctuation'] += 10
        
        # Determine primary category
        if not category_scores:
            # Default based on common patterns
            if any('ing' in t for t in clean_tokens):
                return 'morphemes', 'Verbal/Nominal Morphology'
            else:
                return 'content_words', 'Mixed Content'
        
        # Get the dominant category
        dominant_category = max(category_scores.items(), key=lambda x: x[1])[0]
        
        # Create more specific labels based on combinations
        if dominant_category == 'pronouns':
            if category_scores.get('auxiliaries', 0) > 0:
                return 'pronoun_verbal', 'Pronouns & Verbal Elements'
            else:
                return 'pronouns', 'Personal Pronouns'
        
        elif dominant_category == 'auxiliaries':
            if category_scores.get('action_verbs', 0) > 0:
                return 'verbal_mixed', 'Verbs & Auxiliaries'
            else:
                return 'auxiliaries', 'Auxiliary Verbs'
        
        elif dominant_category == 'determiners':
            if category_scores.get('prepositions', 0) > 2:
                return 'function_core', 'Core Function Words'
            else:
                return 'determiners', 'Determiners & Articles'
        
        elif dominant_category == 'prepositions':
            if category_scores.get('conjunctions', 0) > 1:
                return 'connectives', 'Grammatical Connectives'
            else:
                return 'prepositions', 'Prepositional Phrases'
        
        elif dominant_category == 'punctuation':
            return 'punctuation', 'Punctuation & Boundaries'
        
        elif dominant_category == 'morphemes':
            return 'morphology', 'Morphological Elements'
        
        elif dominant_category == 'content_nouns':
            return 'nouns', 'Common Nouns'
        
        elif dominant_category == 'action_verbs':
            return 'verbs', 'Action Verbs'
        
        else:
            # Check secondary categories
            if category_scores.get('content_nouns', 0) > 0 or category_scores.get('action_verbs', 0) > 0:
                return 'content_mixed', 'Mixed Content Words'
            else:
                return 'function_mixed', 'Mixed Function Elements'

    def create_consistent_labels(self):
        """Create consistent labels across all layers."""
        base_dir = Path(__file__).parent
        
        # Load both simplified and full data
        with open(base_dir / "llm_labels_k5" / "llm_labeling_simplified.json", 'r', encoding='utf-8') as f:
            simplified_data = json.load(f)
        
        with open(base_dir / "llm_labels_k5" / "llm_labeling_data.json", 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        
        # Track cluster characteristics across layers
        cluster_profiles = defaultdict(list)  # cluster_position -> [(layer, category, label)]
        
        # First pass: analyze each cluster
        for layer in range(12):
            layer_key = f"layer_{layer}"
            layer_clusters = simplified_data["layers"][layer_key]
            
            for cluster_info in layer_clusters:
                cluster_key = cluster_info['cluster']
                cluster_idx = int(cluster_key.split('_C')[1])
                
                # Get full token list from full data
                full_cluster = full_data["clusters"][cluster_key]
                all_tokens = full_cluster.get("common_tokens", cluster_info["top_10_tokens"])
                
                # Analyze cluster
                category, label = self.analyze_cluster_comprehensive(
                    all_tokens,
                    cluster_info.get("type_distribution", {})
                )
                
                cluster_profiles[cluster_idx].append((layer, category, label))
        
        # Second pass: ensure consistency
        consistent_labels = self._ensure_consistency(cluster_profiles)
        
        # Create final label structure
        new_labels = {
            "metadata": {
                "generated_at": "2025-05-28T23:55:00",
                "model": "claude",
                "k": 5,
                "total_clusters": 60,
                "method": "linguistically_informed_consistent_analysis"
            },
            "labels": {}
        }
        
        # Apply consistent labels
        for layer in range(12):
            layer_key = f"layer_{layer}"
            new_labels["labels"][layer_key] = {}
            
            for cluster_idx in range(5):
                cluster_key = f"L{layer}_C{cluster_idx}"
                label, description = consistent_labels[cluster_idx]
                
                # Get specific examples for this layer
                cluster_data = full_data["clusters"][cluster_key]
                examples = ", ".join(cluster_data["common_tokens"][:5])
                
                new_labels["labels"][layer_key][cluster_key] = {
                    "label": label,
                    "description": f"{description}. Examples in this layer: {examples}"
                }
        
        # Save results
        output_path = base_dir / "llm_labels_k5" / "cluster_labels_k5_consistent.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_labels, f, indent=2)
        
        logging.info(f"Saved consistent labels to {output_path}")
        
        # Create report
        self._create_consistency_report(cluster_profiles, consistent_labels, base_dir)
        
        return new_labels
    
    def _ensure_consistency(self, cluster_profiles):
        """Ensure labels are consistent across layers."""
        consistent_labels = {}
        
        # Define the canonical labels for each cluster position based on analysis
        label_definitions = {
            'function_core': ('Core Function Words', 'Articles, prepositions, and essential grammatical connectives'),
            'pronouns': ('Pronouns', 'Personal and possessive pronouns'),
            'verbal_mixed': ('Verbal System', 'Auxiliary verbs, main verbs, and verbal morphology'),
            'punctuation': ('Punctuation', 'Punctuation marks and sentence boundaries'),
            'content_mixed': ('Content Words', 'Nouns, adjectives, and other lexical content'),
            'connectives': ('Grammatical Connectives', 'Prepositions, conjunctions, and linking elements'),
            'morphology': ('Morphological Elements', 'Suffixes, prefixes, and word-forming elements'),
            'pronoun_verbal': ('Pronominal-Verbal', 'Pronouns with verbal associations'),
            'auxiliaries': ('Auxiliary System', 'Modal and auxiliary verbs'),
            'determiners': ('Determiners', 'Articles and demonstratives')
        }
        
        # For each cluster position, find the most consistent pattern
        for cluster_idx in range(5):
            profiles = cluster_profiles[cluster_idx]
            
            # Count category occurrences
            category_counts = Counter(p[1] for p in profiles)
            
            # Find dominant category (appears in most layers)
            dominant_category = category_counts.most_common(1)[0][0]
            
            # Log the decision
            logging.info(f"Cluster position {cluster_idx}: {dict(category_counts)}")
            logging.info(f"  -> Assigning '{dominant_category}'")
            
            # Get the label definition
            if dominant_category in label_definitions:
                consistent_labels[cluster_idx] = label_definitions[dominant_category]
            else:
                # Fallback
                consistent_labels[cluster_idx] = (
                    profiles[0][2],  # Use first label
                    f"Cluster type: {dominant_category}"
                )
        
        return consistent_labels
    
    def _create_consistency_report(self, cluster_profiles, consistent_labels, base_dir):
        """Create a detailed consistency report."""
        lines = ["K=5 CLUSTER LABEL CONSISTENCY REPORT", "=" * 50, ""]
        
        lines.append("FINAL CONSISTENT LABELS:")
        lines.append("-" * 30)
        for idx, (label, desc) in consistent_labels.items():
            lines.append(f"Cluster Position {idx}: {label}")
            lines.append(f"  Description: {desc}")
        lines.append("")
        
        lines.append("DETAILED ANALYSIS BY POSITION:")
        lines.append("-" * 30)
        
        for cluster_idx in range(5):
            lines.append(f"\nCluster Position {cluster_idx}:")
            profiles = cluster_profiles[cluster_idx]
            
            # Show how labels varied across layers
            for layer, category, label in profiles:
                lines.append(f"  Layer {layer:2d}: {category:20s} -> {label}")
            
            # Show summary
            category_counts = Counter(p[1] for p in profiles)
            lines.append(f"\n  Summary: {dict(category_counts)}")
            lines.append(f"  Final: {consistent_labels[cluster_idx][0]}")
        
        # Save report
        output_path = base_dir / "llm_labels_k5" / "label_consistency_report.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logging.info(f"Saved consistency report to {output_path}")


def main():
    labeler = ConsistentLabeler()
    labeler.create_consistent_labels()
    print("\nConsistent labeling complete!")
    print("Check llm_labels_k5/cluster_labels_k5_consistent.json for results")
    print("Check llm_labels_k5/label_consistency_report.txt for analysis")


if __name__ == "__main__":
    main()