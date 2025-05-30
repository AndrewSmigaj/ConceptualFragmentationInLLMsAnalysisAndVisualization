#!/usr/bin/env python3
"""
Generate LLM-based semantic labels for k=5 clusters using DirectInterpreter.
"""

import json
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from concept_fragmentation.llm.analysis import ClusterAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class K5ClusterLabeler:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.llm_dir = base_dir / "llm_labels_k5"
        
        # Load simplified labeling data
        with open(self.llm_dir / "llm_labeling_simplified.json", 'r', encoding='utf-8') as f:
            self.labeling_data = json.load(f)
        
        # Initialize LLM
        self.analyzer = ClusterAnalysis(provider='claude', model='claude-3-sonnet-20240229')
        logging.info("Initialized Claude for labeling")
    
    def generate_cluster_labels(self):
        """Generate semantic labels for all clusters."""
        labels = {}
        
        # Process each layer
        for layer in range(12):
            layer_key = f"layer_{layer}"
            layer_clusters = self.labeling_data["layers"][layer_key]
            
            # Prepare prompt for this layer
            prompt = self._create_layer_prompt(layer, layer_clusters)
            
            # Get LLM response
            logging.info(f"Generating labels for layer {layer}...")
            try:
                response = self.interpreter.analyze(prompt)
                layer_labels = self._parse_response(response, layer)
                labels[layer_key] = layer_labels
            except Exception as e:
                logging.error(f"Error generating labels for layer {layer}: {e}")
                labels[layer_key] = self._create_fallback_labels(layer_clusters)
        
        # Save labels
        self._save_labels(labels)
        return labels
    
    def _create_layer_prompt(self, layer: int, clusters) -> str:
        """Create prompt for labeling clusters in a layer."""
        prompt = f"""Please analyze these 5 clusters from layer {layer} of a GPT-2 model and provide semantic labels.

These clusters contain the 10,000 most frequent tokens from the Brown corpus, clustered based on their activations.

For each cluster, provide:
1. A short semantic label (2-4 words)
2. A brief description of what types of tokens dominate this cluster

Layer {layer} Clusters:
"""
        
        for cluster_info in clusters:
            prompt += f"\n{cluster_info['cluster']}:"
            prompt += f"\n  Size: {cluster_info['size']}"
            prompt += f"\n  Dominant type: {cluster_info['dominant_type']}"
            prompt += f"\n  Top 10 tokens: {', '.join(cluster_info['top_10_tokens'])}"
            prompt += f"\n  Type distribution: {cluster_info['type_distribution']}"
            prompt += "\n"
        
        prompt += """
Please format your response as JSON:
{
  "L0_C0": {"label": "...", "description": "..."},
  "L0_C1": {"label": "...", "description": "..."},
  ...
}

Focus on linguistic/grammatical patterns rather than just token types."""
        
        return prompt
    
    def _parse_response(self, response: str, layer: int) -> dict:
        """Parse LLM response to extract labels."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                labels_dict = json.loads(json_match.group())
                # Ensure all clusters have labels
                layer_labels = {}
                for cluster in range(5):
                    cluster_key = f"L{layer}_C{cluster}"
                    if cluster_key in labels_dict:
                        layer_labels[cluster_key] = labels_dict[cluster_key]
                    else:
                        layer_labels[cluster_key] = {
                            "label": f"Cluster {cluster}",
                            "description": "No description available"
                        }
                return layer_labels
        except:
            pass
        
        # Fallback parsing
        return self._create_fallback_labels(self.labeling_data["layers"][f"layer_{layer}"])
    
    def _create_fallback_labels(self, clusters) -> dict:
        """Create fallback labels based on dominant types."""
        labels = {}
        for cluster_info in clusters:
            cluster_key = cluster_info['cluster']
            dominant_type = cluster_info['dominant_type']
            
            # Simple rule-based labeling
            if 'word' in dominant_type:
                label = "Word tokens"
            elif 'punct' in dominant_type:
                label = "Punctuation"
            elif 'mixed' in dominant_type:
                label = "Mixed tokens"
            elif 'numeric' in dominant_type:
                label = "Numeric tokens"
            else:
                label = f"{dominant_type.replace('_', ' ').title()}"
            
            labels[cluster_key] = {
                "label": label,
                "description": f"Cluster dominated by {dominant_type} tokens"
            }
        
        return labels
    
    def _save_labels(self, labels):
        """Save generated labels."""
        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model": "claude",
                "k": 5,
                "total_clusters": 60  # 12 layers * 5 clusters
            },
            "labels": labels
        }
        
        output_path = self.llm_dir / "cluster_labels_k5.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        logging.info(f"Saved cluster labels to {output_path}")
        
        # Also create a summary
        self._create_label_summary(labels)
    
    def _create_label_summary(self, labels):
        """Create a human-readable summary of labels."""
        summary = ["K=5 CLUSTER LABELS SUMMARY", "=" * 50, ""]
        
        for layer in range(12):
            summary.append(f"LAYER {layer}:")
            layer_labels = labels.get(f"layer_{layer}", {})
            
            for cluster in range(5):
                cluster_key = f"L{layer}_C{cluster}"
                if cluster_key in layer_labels:
                    label_info = layer_labels[cluster_key]
                    summary.append(f"  {cluster_key}: {label_info['label']}")
                    summary.append(f"    {label_info['description']}")
            
            summary.append("")
        
        output_path = self.llm_dir / "cluster_labels_summary.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary))
        
        logging.info(f"Saved label summary to {output_path}")


def main():
    base_dir = Path(__file__).parent
    labeler = K5ClusterLabeler(base_dir)
    
    # Generate labels
    labels = labeler.generate_cluster_labels()
    
    logging.info("Label generation complete!")


if __name__ == "__main__":
    main()