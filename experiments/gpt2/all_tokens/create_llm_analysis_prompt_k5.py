#!/usr/bin/env python3
"""
Create a comprehensive prompt for Claude to analyze k=5 clusters.
This generates a JSON file that can be directly analyzed to produce semantic labels.
"""

import json
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_analysis_prompt():
    """Create a comprehensive prompt for LLM analysis."""
    base_dir = Path(__file__).parent
    llm_dir = base_dir / "llm_labels_k5"
    
    # Load the simplified labeling data
    with open(llm_dir / "llm_labeling_simplified.json", 'r', encoding='utf-8') as f:
        labeling_data = json.load(f)
    
    # Create prompt structure
    prompt_data = {
        "task": "Analyze GPT-2 token clusters and provide semantic labels",
        "instructions": """Please analyze these clusters from GPT-2 model layers and provide:
1. A short semantic label (2-4 words) for each cluster
2. A brief description of what linguistic/grammatical patterns characterize each cluster

These clusters contain the 10,000 most frequent tokens from the Brown corpus, clustered based on their activations at each layer.

Focus on identifying linguistic patterns rather than just token types. Look for:
- Grammatical categories (nouns, verbs, adjectives, etc.)
- Syntactic roles (subjects, objects, modifiers)
- Semantic groupings (abstract vs concrete, animate vs inanimate)
- Morphological patterns (prefixes, suffixes, compound parts)
- Functional categories (determiners, prepositions, conjunctions)

Please format your response as a JSON object with the following structure:
{
  "layer_0": {
    "L0_C0": {"label": "...", "description": "..."},
    "L0_C1": {"label": "...", "description": "..."},
    ...
  },
  "layer_1": {
    ...
  },
  ...
}
""",
        "cluster_data": {}
    }
    
    # Add cluster data for each layer
    for layer in range(12):
        layer_key = f"layer_{layer}"
        layer_clusters = labeling_data["layers"][layer_key]
        
        layer_data = {}
        for cluster_info in layer_clusters:
            cluster_key = cluster_info['cluster']
            
            # Format cluster data
            cluster_data = {
                "size": cluster_info['size'],
                "dominant_type": cluster_info['dominant_type'],
                "top_tokens": cluster_info['top_10_tokens'],
                "type_distribution": cluster_info['type_distribution']
            }
            
            # Add dominant pattern if available
            if 'dominant_pattern' in cluster_info:
                cluster_data['dominant_pattern'] = cluster_info['dominant_pattern']
            
            layer_data[cluster_key] = cluster_data
        
        prompt_data["cluster_data"][layer_key] = layer_data
    
    # Save the prompt data
    output_path = llm_dir / "llm_analysis_prompt_k5.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_data, f, indent=2)
    
    logging.info(f"Created analysis prompt at {output_path}")
    
    # Also create a human-readable version
    create_readable_prompt(prompt_data, llm_dir)
    
    return prompt_data


def create_readable_prompt(prompt_data, output_dir):
    """Create a human-readable version of the prompt."""
    lines = []
    lines.append("GPT-2 K=5 CLUSTER ANALYSIS REQUEST")
    lines.append("=" * 60)
    lines.append("")
    lines.append(prompt_data["instructions"])
    lines.append("")
    lines.append("CLUSTER DATA BY LAYER:")
    lines.append("-" * 40)
    
    for layer in range(12):
        layer_key = f"layer_{layer}"
        lines.append(f"\nLAYER {layer}:")
        
        layer_data = prompt_data["cluster_data"][layer_key]
        for cluster_key, cluster_info in layer_data.items():
            lines.append(f"\n  {cluster_key}:")
            lines.append(f"    Size: {cluster_info['size']}")
            lines.append(f"    Dominant type: {cluster_info['dominant_type']}")
            lines.append(f"    Top tokens: {', '.join(cluster_info['top_tokens'])}")
            
            # Show type distribution
            type_dist = cluster_info['type_distribution']
            if type_dist:
                lines.append("    Type distribution:")
                for token_type, count in sorted(type_dist.items(), key=lambda x: x[1], reverse=True)[:3]:
                    lines.append(f"      - {token_type}: {count}")
    
    output_path = output_dir / "llm_analysis_prompt_readable.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logging.info(f"Created readable prompt at {output_path}")


def main():
    """Create the analysis prompt."""
    prompt_data = create_analysis_prompt()
    
    print("\n" + "="*60)
    print("ANALYSIS PROMPT CREATED")
    print("="*60)
    print("\nThe analysis prompt has been saved to:")
    print("  - llm_labels_k5/llm_analysis_prompt_k5.json (for direct analysis)")
    print("  - llm_labels_k5/llm_analysis_prompt_readable.txt (human-readable)")
    print("\nYou can now analyze these clusters to generate semantic labels.")
    print("The JSON file contains all the necessary data for comprehensive analysis.")


if __name__ == "__main__":
    main()