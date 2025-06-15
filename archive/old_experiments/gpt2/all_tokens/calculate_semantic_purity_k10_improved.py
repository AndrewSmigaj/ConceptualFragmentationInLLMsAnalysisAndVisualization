#!/usr/bin/env python3
"""
Calculate improved semantic purity percentages for k=10 clusters.
Uses data-driven labels and better category mappings.
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImprovedSemanticPurityAnalyzer:
    """Analyze semantic purity using data-driven approach."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        
        # Load cluster data
        with open(base_dir / "llm_labels_k10" / "llm_labeling_data.json", 'r') as f:
            self.cluster_data = json.load(f)
        
        # Load data-driven labels
        with open(base_dir / "llm_labels_k10" / "cluster_labels_k10_datadriven.json", 'r') as f:
            label_data = json.load(f)
            self.semantic_labels = label_data['labels']
        
        # Load purity mappings
        with open(base_dir / "llm_labels_k10" / "purity_mappings_k10.json", 'r') as f:
            self.purity_mappings = json.load(f)
        
        # Define linguistic categories (same as in labeler)
        self.linguistic_categories = {
            'punctuation': {
                '.', ',', '?', '!', ';', ':', '-', '--', '(', ')', '[', ']', 
                '{', '}', '"', "'", '``', "''", '...', '/', '*'
            },
            'articles': {'the', 'a', 'an'},
            'pronouns': {
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 
                'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
                'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves',
                'themselves', 'mine', 'yours', 'hers', 'ours', 'theirs'
            },
            'auxiliaries': {
                'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                'will', 'would', 'shall', 'should', 'can', 'could', 'may', 
                'might', 'must', 'ought', "'ll", "'d", "'ve"
            },
            'prepositions': {
                'of', 'to', 'in', 'for', 'on', 'with', 'by', 'from', 'up', 'out',
                'down', 'off', 'over', 'under', 'through', 'into', 'onto', 'upon',
                'about', 'above', 'across', 'after', 'against', 'along', 'among',
                'around', 'at', 'before', 'behind', 'below', 'beneath', 'beside',
                'between', 'beyond', 'during', 'except', 'inside', 'near', 'since',
                'toward', 'towards', 'underneath', 'until', 'within', 'without'
            },
            'conjunctions': {
                'and', 'or', 'but', 'so', 'yet', 'for', 'nor', 'if', 'when', 'where',
                'while', 'since', 'because', 'although', 'though', 'unless', 'until',
                'as', 'that', 'whether', 'whereas', 'wherever', 'whenever'
            },
            'negations': {'not', 'no', 'never', 'none', "n't", 'neither', 'nor'},
            'numbers': {
                'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 
                'nine', 'ten', 'hundred', 'thousand', 'million', 'billion',
                'first', 'second', 'third', 'fourth', 'fifth', 'few', 'many', 'several'
            },
            'quotes': {'``', "''", '"', "'"},
            'morphological': {'ing', 'ed', 'er', 'est', 'ly', 's', 'es', "'s", 'tion', 'sion', 'ness', 'ment'}
        }
    
    def analyze_token_categories(self, tokens):
        """Analyze what linguistic categories tokens belong to."""
        categories = defaultdict(int)
        total_tokens = len(tokens)
        
        for token in tokens:
            # Clean token
            clean_token = token.strip().lower()
            if not clean_token:
                continue
            
            # Check each category
            for category, word_set in self.linguistic_categories.items():
                if clean_token in word_set:
                    categories[category] += 1
                elif category == 'morphological':
                    # Check for morphological endings
                    for ending in word_set:
                        if clean_token.endswith(ending) and len(clean_token) > len(ending):
                            categories[category] += 1
                            break
            
            # Additional checks
            if clean_token.isdigit():
                categories['numeric'] += 1
            elif re.match(r'^[A-Z]', token.strip()):
                categories['capitalized'] += 1
        
        # Convert to percentages
        category_percentages = {}
        for category, count in categories.items():
            category_percentages[category] = (count / total_tokens) * 100 if total_tokens > 0 else 0
        
        return category_percentages
    
    def calculate_semantic_purity(self, cluster_key, semantic_label, primary_categories):
        """Calculate semantic purity based on data-driven categories."""
        cluster_info = self.cluster_data['clusters'][cluster_key]
        tokens = cluster_info['common_tokens']
        
        # Analyze token categories
        categories = self.analyze_token_categories(tokens)
        
        # Get expected categories from our mapping
        expected_categories = self.purity_mappings.get(semantic_label, [])
        
        if not expected_categories and primary_categories:
            # Use the primary categories from the label data
            expected_categories = list(primary_categories.keys())
        
        if not expected_categories:
            # For labels without clear categories, calculate based on label type
            if 'noun' in semantic_label.lower():
                # For noun clusters, check if tokens are mostly non-function words
                function_score = sum(categories.get(cat, 0) for cat in 
                                   ['articles', 'prepositions', 'conjunctions', 'auxiliaries'])
                purity = max(0, 100 - function_score)
                return purity, "content-based"
            elif 'mixed' in semantic_label.lower():
                # Mixed clusters have low purity by definition
                return 25.0, "mixed-content"
            else:
                # Default for uncategorizable
                return 40.0, "estimated"
        
        # Calculate purity as the sum of percentages for expected categories
        purity = sum(categories.get(cat, 0) for cat in expected_categories)
        
        # Cap at 100% but allow high scores for well-defined clusters
        purity = min(purity, 100.0)
        
        # Boost purity if it's a very cohesive cluster (few categories)
        if len([c for c in categories.values() if c > 5]) <= 2:
            purity = min(purity * 1.2, 100.0)
        
        return purity, "calculated"
    
    def analyze_all_clusters(self):
        """Analyze semantic purity for all clusters."""
        results = {}
        
        for layer in range(12):
            layer_key = f"layer_{layer}"
            results[layer_key] = {}
            
            for cluster_idx in range(10):
                cluster_key = f"L{layer}_C{cluster_idx}"
                
                if (layer_key in self.semantic_labels and 
                    cluster_key in self.semantic_labels[layer_key]):
                    
                    label_info = self.semantic_labels[layer_key][cluster_key]
                    semantic_label = label_info['label']
                    primary_categories = label_info.get('primary_categories', {})
                    
                    purity, method = self.calculate_semantic_purity(
                        cluster_key, semantic_label, primary_categories
                    )
                    
                    results[layer_key][cluster_key] = {
                        'label': semantic_label,
                        'purity': purity,
                        'method': method,
                        'primary_categories': primary_categories
                    }
                    
                    logging.info(f"{cluster_key}: {semantic_label} - {purity:.1f}% purity ({method})")
        
        return results
    
    def save_purity_analysis(self, results):
        """Save improved purity analysis results."""
        output_path = self.base_dir / "llm_labels_k10" / "semantic_purity_k10_improved.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Saved improved purity analysis to {output_path}")
        
        # Create summary report
        summary_lines = ["IMPROVED SEMANTIC PURITY ANALYSIS (K=10)", "=" * 50, ""]
        
        # Calculate statistics
        all_purities = []
        calculated_count = 0
        estimated_count = 0
        
        for layer in range(12):
            layer_key = f"layer_{layer}"
            if layer_key in results:
                summary_lines.append(f"LAYER {layer}:")
                summary_lines.append("-" * 30)
                
                layer_purities = []
                for cluster_idx in range(10):
                    cluster_key = f"L{layer}_C{cluster_idx}"
                    if cluster_key in results[layer_key]:
                        info = results[layer_key][cluster_key]
                        purity = info['purity']
                        label = info['label']
                        method = info['method']
                        
                        all_purities.append(purity)
                        layer_purities.append(purity)
                        
                        if method == 'calculated':
                            calculated_count += 1
                        else:
                            estimated_count += 1
                        
                        summary_lines.append(f"  C{cluster_idx}: {label} ({purity:.1f}% {method})")
                
                if layer_purities:
                    avg_purity = sum(layer_purities) / len(layer_purities)
                    summary_lines.append(f"  Layer average: {avg_purity:.1f}%")
                
                summary_lines.append("")
        
        # Overall statistics
        if all_purities:
            overall_avg = sum(all_purities) / len(all_purities)
            summary_lines.extend([
                "OVERALL STATISTICS:",
                "-" * 30,
                f"Average purity: {overall_avg:.1f}%",
                f"Calculated purities: {calculated_count}",
                f"Estimated purities: {estimated_count}",
                f"Highest purity: {max(all_purities):.1f}%",
                f"Lowest purity: {min(all_purities):.1f}%"
            ])
        
        summary_path = self.base_dir / "llm_labels_k10" / "semantic_purity_summary_improved.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        logging.info(f"Saved improved purity summary to {summary_path}")


def main():
    base_dir = Path(__file__).parent
    analyzer = ImprovedSemanticPurityAnalyzer(base_dir)
    
    # Analyze semantic purity
    results = analyzer.analyze_all_clusters()
    
    # Save results
    analyzer.save_purity_analysis(results)
    
    print("\nImproved semantic purity analysis complete!")
    print("Results saved to:")
    print("  - semantic_purity_k10_improved.json")
    print("  - semantic_purity_summary_improved.txt")


if __name__ == "__main__":
    main()