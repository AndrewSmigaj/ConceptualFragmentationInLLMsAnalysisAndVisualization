#!/usr/bin/env python3
"""
Calculate semantic purity percentages for k=5 clusters.
Analyzes how well each cluster matches its assigned semantic label.
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SemanticPurityAnalyzer:
    """Analyze semantic purity of clusters based on linguistic categories."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        
        # Load cluster data
        with open(base_dir / "llm_labels_k5" / "llm_labeling_data.json", 'r') as f:
            self.cluster_data = json.load(f)
        
        # Load semantic labels
        with open(base_dir / "llm_labels_k5" / "cluster_labels_k5.json", 'r') as f:
            label_data = json.load(f)
            self.semantic_labels = label_data['labels']
        
        # Define linguistic categories for purity analysis
        self.linguistic_categories = {
            'function_words': {
                'the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 
                'with', 'from', 'up', 'out', 'down', 'off', 'over', 'under', 'as', 'that', 'this'
            },
            'pronouns': {
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'
            },
            'auxiliaries': {
                'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must'
            },
            'prepositions': {
                'of', 'to', 'in', 'for', 'on', 'with', 'by', 'from', 'up', 'out', 'down', 'off',
                'over', 'under', 'above', 'below', 'across', 'through', 'into', 'onto', 'upon'
            },
            'conjunctions': {
                'and', 'or', 'but', 'so', 'yet', 'for', 'nor', 'that', 'if', 'when', 'where',
                'while', 'since', 'because', 'although', 'though', 'unless', 'until', 'as'
            },
            'punctuation': {
                '.', ',', '?', '!', ';', ':', '-', '--', '(', ')', '[', ']', '{', '}',
                '"', "'", '``', "''", '/', '\\', '*', '&', '%', '$', '#', '@'
            },
            'copulas': {'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being'},
            'articles': {'the', 'a', 'an'},
            'discourse_markers': {'``', "''", '"', "'", 'well', 'now', 'then', 'so', 'oh', 'ah'},
            'morphological_endings': {'ing', 'ed', 'er', 'est', 'ly', 's', 'es', 'tion', 'sion', 'ness', 'ment'}
        }
    
    def analyze_token_categories(self, tokens):
        """Analyze what linguistic categories tokens belong to."""
        categories = defaultdict(int)
        total_tokens = len(tokens)
        
        for token in tokens:
            # Clean token (remove spaces and special chars for analysis)
            clean_token = token.strip().lower()
            if not clean_token:
                continue
                
            # Check each category
            for category, word_set in self.linguistic_categories.items():
                if clean_token in word_set:
                    categories[category] += 1
                # Special checks for morphological endings
                elif category == 'morphological_endings':
                    for ending in word_set:
                        if clean_token.endswith(ending) and len(clean_token) > len(ending):
                            categories[category] += 1
                            break
        
        # Convert to percentages
        category_percentages = {}
        for category, count in categories.items():
            category_percentages[category] = (count / total_tokens) * 100 if total_tokens > 0 else 0
        
        return category_percentages
    
    def calculate_semantic_purity(self, cluster_key, semantic_label):
        """Calculate how well a cluster matches its semantic label."""
        cluster_info = self.cluster_data['clusters'][cluster_key]
        tokens = cluster_info['common_tokens']
        
        # Analyze token categories
        categories = self.analyze_token_categories(tokens)
        
        # Map semantic labels to expected categories
        label_mappings = {
            'Function Core': ['function_words', 'prepositions', 'articles'],
            'Grammar Connectives': ['conjunctions', 'articles', 'function_words'],
            'Auxiliary System': ['auxiliaries', 'copulas'],
            'Abstract Nouns': [],  # Hard to categorize automatically
            'Discourse Markers': ['discourse_markers', 'punctuation'],
            'Copulas & Prepositions': ['copulas', 'prepositions'],
            'Articles & Conjunctions': ['articles', 'conjunctions'],
            'Pronouns & Short Words': ['pronouns'],
            'Punctuation & Capitals': ['punctuation'],
            'Quotes & Titles': ['discourse_markers'],
            'Core Prepositions': ['prepositions'],
            'Content Nouns': [],  # Hard to categorize automatically
            'Sentence Boundaries': ['punctuation'],
            'Short Forms': [],
            'Morphological Elements': ['morphological_endings']
        }
        
        expected_categories = label_mappings.get(semantic_label, [])
        
        if not expected_categories:
            # For labels we can't automatically verify, return a conservative estimate
            return 50.0, "estimated"
        
        # Calculate purity as the sum of percentages for expected categories
        purity = sum(categories.get(cat, 0) for cat in expected_categories)
        
        # Cap at 100%
        purity = min(purity, 100.0)
        
        return purity, "calculated"
    
    def analyze_all_clusters(self):
        """Analyze semantic purity for all clusters."""
        results = {}
        
        for layer in range(12):
            layer_key = f"layer_{layer}"
            results[layer_key] = {}
            
            for cluster_idx in range(5):
                cluster_key = f"L{layer}_C{cluster_idx}"
                
                if (layer_key in self.semantic_labels and 
                    cluster_key in self.semantic_labels[layer_key]):
                    
                    semantic_label = self.semantic_labels[layer_key][cluster_key]['label']
                    purity, method = self.calculate_semantic_purity(cluster_key, semantic_label)
                    
                    results[layer_key][cluster_key] = {
                        'label': semantic_label,
                        'purity': purity,
                        'method': method
                    }
                    
                    logging.info(f"{cluster_key}: {semantic_label} - {purity:.1f}% purity ({method})")
        
        return results
    
    def save_purity_analysis(self, results):
        """Save purity analysis results."""
        output_path = self.base_dir / "llm_labels_k5" / "semantic_purity_k5.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Saved purity analysis to {output_path}")
        
        # Create summary report
        summary_lines = ["SEMANTIC PURITY ANALYSIS", "=" * 50, ""]
        
        for layer in range(12):
            layer_key = f"layer_{layer}"
            if layer_key in results:
                summary_lines.append(f"LAYER {layer}:")
                summary_lines.append("-" * 30)
                
                for cluster_idx in range(5):
                    cluster_key = f"L{layer}_C{cluster_idx}"
                    if cluster_key in results[layer_key]:
                        info = results[layer_key][cluster_key]
                        purity = info['purity']
                        label = info['label']
                        method = info['method']
                        
                        summary_lines.append(f"  C{cluster_idx}: {label} ({purity:.1f}% {method})")
                
                summary_lines.append("")
        
        summary_path = self.base_dir / "llm_labels_k5" / "semantic_purity_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        logging.info(f"Saved purity summary to {summary_path}")


def main():
    base_dir = Path(__file__).parent
    analyzer = SemanticPurityAnalyzer(base_dir)
    
    # Analyze semantic purity
    results = analyzer.analyze_all_clusters()
    
    # Save results
    analyzer.save_purity_analysis(results)
    
    print("\nSemantic purity analysis complete!")
    print("Results saved to:")
    print("  - semantic_purity_k5.json")
    print("  - semantic_purity_summary.txt")


if __name__ == "__main__":
    main()