#!/usr/bin/env python3
"""
Prepare data for LLM-based cluster labeling for any k value.
Extracts representative tokens and patterns for each cluster to enable semantic labeling.
Usage: python prepare_llm_labeling_data.py --k 10
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LLMDataPreparer:
    def __init__(self, base_dir: Path, k: int):
        self.base_dir = base_dir
        self.k = k
        self.clustering_dir = base_dir / f"clustering_results_k{k}"
        self.llm_dir = base_dir / f"llm_labels_k{k}"
        self.llm_dir.mkdir(exist_ok=True)
        
        # Load token information
        logging.info("Loading token information...")
        with open(base_dir / "frequent_tokens_full.json", 'r', encoding='utf-8') as f:
            self.token_list = json.load(f)
        
        # Create token lookup
        self.token_lookup = {t['token_id']: t for t in self.token_list}
        
        # Load clustering results
        logging.info(f"Loading k={k} clustering results...")
        with open(self.clustering_dir / f"clustering_results_k{k}.json", 'r') as f:
            self.clustering_metadata = json.load(f)
        
        # Load actual labels
        with open(self.clustering_dir / f"all_labels_k{k}.json", 'r') as f:
            self.all_labels = json.load(f)
        
        logging.info(f"Loaded {len(self.token_list)} tokens with k={k} clustering")
    
    def analyze_cluster_characteristics(self, layer: int, cluster_id: int) -> Dict:
        """Analyze characteristics of tokens in a specific cluster."""
        labels = self.all_labels[str(layer)]
        
        # Find all token indices in this cluster
        token_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        
        if not token_indices:
            return {"error": "No tokens in cluster"}
        
        # Get token info for all tokens in cluster
        cluster_tokens = []
        for idx in token_indices:
            # Get token from list by index
            if idx < len(self.token_list):
                cluster_tokens.append(self.token_list[idx])
        
        # Analyze patterns
        token_types = Counter(t.get('token_type', 'unknown') for t in cluster_tokens)
        subword_patterns = Counter(t.get('subword_type', 'none') for t in cluster_tokens)
        
        pattern_counts = {
            'has_space': sum(1 for t in cluster_tokens if t.get('has_leading_space', False)),
            'is_alphabetic': sum(1 for t in cluster_tokens if t.get('is_alphabetic', False)),
            'is_punctuation': sum(1 for t in cluster_tokens if t.get('is_punctuation', False)),
            'is_numeric': sum(1 for t in cluster_tokens if t.get('is_numeric', False)),
            'is_uppercase': sum(1 for t in cluster_tokens if t.get('is_uppercase', False)),
            'is_lowercase': sum(1 for t in cluster_tokens if t.get('is_lowercase', False)),
            'is_mixed_case': sum(1 for t in cluster_tokens if t.get('is_mixed_case', False))
        }
        
        # Get most common tokens (clean display)
        common_tokens = [t['token_str'] for t in cluster_tokens[:30]]
        
        # Calculate percentages
        total = len(cluster_tokens)
        percentages = {k: (v/total)*100 for k, v in pattern_counts.items()}
        
        return {
            'size': total,
            'percentage': (total / len(self.token_list)) * 100,
            'common_tokens': common_tokens,
            'token_types': dict(token_types),
            'subword_patterns': dict(subword_patterns),
            'pattern_counts': pattern_counts,
            'pattern_percentages': percentages
        }
    
    def prepare_cluster_analysis(self):
        """Prepare comprehensive analysis for all clusters."""
        logging.info(f"Analyzing all {self.k} clusters across 12 layers...")
        
        cluster_data = {
            'metadata': {
                'k': self.k,
                'total_tokens': len(self.token_list),
                'total_layers': 12,
                'generated_at': datetime.now().isoformat()
            },
            'clusters': {}
        }
        
        for layer in range(12):
            logging.info(f"Analyzing layer {layer}...")
            
            for cluster_id in range(self.k):
                cluster_key = f"L{layer}_C{cluster_id}"
                cluster_info = self.analyze_cluster_characteristics(layer, cluster_id)
                cluster_data['clusters'][cluster_key] = cluster_info
        
        # Save cluster analysis
        output_path = self.llm_dir / "llm_labeling_data.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_data, f, indent=2)
        
        logging.info(f"Saved cluster analysis to {output_path}")
        return cluster_data
    
    def create_llm_prompt(self, cluster_data: Dict):
        """Create structured prompt for LLM analysis."""
        prompt_data = {
            "task": "Analyze GPT-2 token clusters and provide semantic labels",
            "instructions": f"""Please analyze these clusters from GPT-2 model layers and provide semantic labels.

IMPORTANT REQUIREMENTS:
1. Labels should be DIFFERENTIATED - show what makes each cluster unique
2. Labels should be CONSISTENT - same concepts get same names across layers
3. Focus on linguistic/semantic categories (grammar, syntax, semantics)
4. Be specific about what distinguishes each cluster
5. With k={self.k}, expect finer-grained distinctions than with smaller k

For each cluster, provide:
- A concise semantic label (2-4 words)
- A brief description of what tokens it contains

Example good labels:
- "Modal Auxiliaries" (not just "Auxiliary Verbs")
- "Sentence-Initial Capitals" (not just "Capitals")
- "Mathematical Operators" (not just "Symbols")

Analyze these {self.k} clusters per layer across 12 layers:""",
            "cluster_data": {}
        }
        
        # Add cluster data organized by layer
        for layer in range(12):
            layer_clusters = {}
            for cluster_id in range(self.k):
                cluster_key = f"L{layer}_C{cluster_id}"
                if cluster_key in cluster_data['clusters']:
                    cluster_info = cluster_data['clusters'][cluster_key]
                    
                    layer_clusters[f"C{cluster_id}"] = {
                        "size": cluster_info['size'],
                        "percentage": f"{cluster_info['percentage']:.1f}%",
                        "common_tokens": cluster_info['common_tokens'][:15],
                        "token_types": cluster_info.get('token_types', {}),
                        "linguistic_features": {
                            "has_space": f"{cluster_info['pattern_percentages']['has_space']:.1f}%",
                            "alphabetic": f"{cluster_info['pattern_percentages']['is_alphabetic']:.1f}%",
                            "punctuation": f"{cluster_info['pattern_percentages']['is_punctuation']:.1f}%",
                            "numeric": f"{cluster_info['pattern_percentages']['is_numeric']:.1f}%",
                            "uppercase": f"{cluster_info['pattern_percentages']['is_uppercase']:.1f}%"
                        }
                    }
            
            prompt_data["cluster_data"][f"layer_{layer}"] = layer_clusters
        
        # Save JSON prompt
        json_path = self.llm_dir / f"llm_analysis_prompt_k{self.k}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(prompt_data, f, indent=2)
        
        # Create human-readable version
        readable_lines = [
            prompt_data["instructions"],
            "",
            "CLUSTER DATA BY LAYER:",
            "=" * 50
        ]
        
        for layer in range(12):
            readable_lines.append(f"\nLAYER {layer}:")
            readable_lines.append("-" * 30)
            
            layer_data = prompt_data["cluster_data"][f"layer_{layer}"]
            for cluster_id in sorted(layer_data.keys()):
                info = layer_data[cluster_id]
                readable_lines.extend([
                    f"\n{cluster_id}: {info['size']} tokens ({info['percentage']})",
                    f"Common tokens: {', '.join(info['common_tokens'][:10])}",
                    f"Features: {info['linguistic_features']}"
                ])
        
        readable_path = self.llm_dir / "llm_analysis_prompt_readable.txt"
        with open(readable_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(readable_lines))
        
        logging.info(f"Saved LLM prompts to {json_path} and {readable_path}")


def main():
    """Prepare LLM labeling data for specified k value."""
    parser = argparse.ArgumentParser(description='Prepare LLM labeling data for GPT-2 clusters')
    parser.add_argument('--k', type=int, required=True, help='Number of clusters')
    parser.add_argument('--base-dir', type=str, default='.', help='Base directory')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    k = args.k
    
    print(f"\n{'='*60}")
    print(f"PREPARING LLM LABELING DATA FOR K={k}")
    print(f"{'='*60}")
    
    # Check if clustering results exist
    clustering_dir = base_dir / f"clustering_results_k{k}"
    if not clustering_dir.exists():
        print(f"\nERROR: Clustering results not found at {clustering_dir}")
        print(f"Please run clustering with k={k} first.")
        return
    
    # Initialize preparer
    preparer = LLMDataPreparer(base_dir, k=k)
    
    # Prepare cluster analysis
    cluster_data = preparer.prepare_cluster_analysis()
    
    # Create LLM prompt
    preparer.create_llm_prompt(cluster_data)
    
    print(f"\n{'='*60}")
    print(f"LLM LABELING DATA PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nData saved to: {preparer.llm_dir}")
    print(f"Next step: Analyze the prompt and provide semantic labels")


if __name__ == "__main__":
    main()