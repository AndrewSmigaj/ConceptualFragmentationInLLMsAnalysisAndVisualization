#!/usr/bin/env python3
"""
Prepare data for LLM-based cluster labeling for k=5 clustering results.
Extracts representative tokens and patterns for each cluster to enable semantic labeling.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class K5LLMLabelPreparer:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.clustering_dir = base_dir / "clustering_results_k5"
        self.llm_dir = base_dir / "llm_labels_k5"
        self.llm_dir.mkdir(exist_ok=True)
        
        # Load token information
        logging.info("Loading token information...")
        with open(base_dir / "frequent_tokens_full.json", 'r', encoding='utf-8') as f:
            self.token_list = json.load(f)
        
        # Create token lookup
        self.token_lookup = {t['token_id']: t for t in self.token_list}
        
        # Load clustering results
        logging.info("Loading k=5 clustering results...")
        with open(self.clustering_dir / "clustering_results_k5.json", 'r') as f:
            self.clustering_results = json.load(f)
        
        # Load actual labels
        with open(self.clustering_dir / "all_labels_k5.json", 'r') as f:
            self.all_labels = json.load(f)
    
    def extract_cluster_characteristics(self, layer: int, cluster: int) -> Dict:
        """Extract characteristics for a specific cluster."""
        cluster_key = f"L{layer}_C{cluster}"
        
        # Get tokens in this cluster
        cluster_tokens = []
        layer_labels = self.all_labels[str(layer)]
        
        for token_idx, label in enumerate(layer_labels):
            if label == cluster:
                cluster_tokens.append(self.token_list[token_idx])
        
        # Analyze token types
        token_types = Counter(t['token_type'] for t in cluster_tokens)
        
        # Analyze subword patterns
        subword_patterns = Counter(t.get('subword_type', 'none') for t in cluster_tokens)
        
        # Get top 20 most common tokens (by position in frequent list)
        common_tokens = cluster_tokens[:20]  # Already sorted by frequency
        
        # Sample diverse tokens (different types)
        diverse_tokens = []
        seen_types = set()
        for token in cluster_tokens:
            if token['token_type'] not in seen_types and len(diverse_tokens) < 10:
                diverse_tokens.append(token)
                seen_types.add(token['token_type'])
        
        return {
            "cluster_id": cluster_key,
            "layer": layer,
            "cluster": cluster,
            "size": len(cluster_tokens),
            "percentage": (len(cluster_tokens) / len(self.token_list)) * 100,
            "token_type_distribution": dict(token_types.most_common()),
            "subword_patterns": dict(subword_patterns.most_common()),
            "common_tokens": [t['token_str'] for t in common_tokens],
            "diverse_tokens": [t['token_str'] for t in diverse_tokens],
            "dominant_type": token_types.most_common(1)[0][0] if token_types else "unknown",
            "dominant_pattern": subword_patterns.most_common(1)[0][0] if subword_patterns else "no_pattern"
        }
    
    def prepare_llm_labeling_data(self):
        """Prepare comprehensive data for LLM labeling."""
        logging.info("Preparing LLM labeling data...")
        
        labeling_data = {
            "metadata": {
                "total_tokens": len(self.token_list),
                "num_layers": 12,
                "num_clusters": 5,
                "token_source": "10k most frequent GPT-2 tokens from Brown corpus",
                "timestamp": datetime.now().isoformat()
            },
            "clusters": {},
            "layer_summaries": {}
        }
        
        # Process each layer
        for layer in range(12):
            layer_clusters = []
            
            for cluster in range(5):
                cluster_data = self.extract_cluster_characteristics(layer, cluster)
                cluster_key = cluster_data["cluster_id"]
                labeling_data["clusters"][cluster_key] = cluster_data
                layer_clusters.append(cluster_data)
            
            # Create layer summary
            labeling_data["layer_summaries"][f"layer_{layer}"] = {
                "layer": layer,
                "clusters": [c["cluster_id"] for c in layer_clusters],
                "size_distribution": {c["cluster_id"]: c["size"] for c in layer_clusters},
                "dominant_types": {c["cluster_id"]: c["dominant_type"] for c in layer_clusters},
                "dominant_subword_patterns": {c["cluster_id"]: c["dominant_pattern"] for c in layer_clusters}
            }
        
        # Save the data
        output_path = self.llm_dir / "llm_labeling_data.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(labeling_data, f, indent=2)
        
        logging.info(f"Saved LLM labeling data to {output_path}")
        
        # Also create a simplified version for quick labeling
        self.create_simplified_labeling_data(labeling_data)
        
        return labeling_data
    
    def create_simplified_labeling_data(self, full_data: Dict):
        """Create a simplified version focusing on key information for labeling."""
        simplified = {
            "instructions": "Please provide semantic labels for these clusters based on their token composition.",
            "layers": {}
        }
        
        for layer in range(12):
            layer_data = []
            for cluster in range(5):
                cluster_key = f"L{layer}_C{cluster}"
                cluster_info = full_data["clusters"][cluster_key]
                
                layer_data.append({
                    "cluster": cluster_key,
                    "size": f"{cluster_info['size']} tokens ({cluster_info['percentage']:.1f}%)",
                    "dominant_type": cluster_info["dominant_type"],
                    "dominant_pattern": cluster_info["dominant_pattern"],
                    "top_10_tokens": cluster_info["common_tokens"][:10],
                    "type_distribution": cluster_info["token_type_distribution"]
                })
            
            simplified["layers"][f"layer_{layer}"] = layer_data
        
        # Save simplified version
        output_path = self.llm_dir / "llm_labeling_simplified.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified, f, indent=2)
        
        logging.info(f"Saved simplified labeling data to {output_path}")


def main():
    base_dir = Path(__file__).parent
    preparer = K5LLMLabelPreparer(base_dir)
    preparer.prepare_llm_labeling_data()
    logging.info("LLM labeling data preparation complete!")


if __name__ == "__main__":
    main()