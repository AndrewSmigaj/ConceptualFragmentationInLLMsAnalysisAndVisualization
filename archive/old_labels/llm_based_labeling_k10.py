#!/usr/bin/env python3
"""
LLM-based labeling for k=10 clusters.
Uses actual LLM analysis to understand cluster patterns and maintain consistency.
"""

import json
from pathlib import Path
import logging
from collections import defaultdict
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from concept_fragmentation.llm.client import LLMClient
from concept_fragmentation.llm.factory import LLMFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Prompts for LLM analysis
INITIAL_ANALYSIS_PROMPT = """Analyze this cluster of tokens from GPT-2 layer {layer}:

Top 30 tokens: {tokens}
Cluster size: {size} tokens

What unifies these tokens? Consider:
- Grammatical function (articles, prepositions, auxiliaries, etc.)
- Semantic category (objects, actions, concepts, etc.)
- Morphological patterns (prefixes, suffixes, word forms)
- Usage patterns (how these words typically function in sentences)

Provide a JSON response with:
{{
  "label": "concise 2-3 word label",
  "linguistic_type": "what type of linguistic organization this represents",
  "function_or_meaning": "function" or "meaning" or "mixed",
  "reasoning": "brief explanation of what unifies these tokens"
}}"""

CONSISTENCY_CHECK_PROMPT = """You previously analyzed similar clusters. Here's the context:

Previous similar clusters:
{previous_clusters}

Now analyze this new cluster:
Layer {layer}: {tokens}
Size: {size} tokens

Is this essentially the same type of cluster as any of the previous ones? 
If yes, use the EXACT same label for consistency.
If it has evolved or is different, explain how.

Provide a JSON response with:
{{
  "matches_previous": true/false,
  "matched_label": "exact label if matching" or null,
  "new_label": "new label if different" or null,
  "reasoning": "explanation of your decision"
}}"""

EVOLUTION_ANALYSIS_PROMPT = """Analyze how this concept evolves through GPT-2's layers:

{cluster_progression}

What transformation is happening as this group of tokens moves through the network?
How does GPT-2's representation of these tokens change from early to late layers?

Provide insights about the evolution pattern."""


class LLMLabeler:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.label_cache = {}
        self.similar_clusters = defaultdict(list)
        
    def analyze_cluster(self, cluster_key, layer, tokens, size):
        """Analyze a single cluster using LLM."""
        prompt = INITIAL_ANALYSIS_PROMPT.format(
            layer=layer,
            tokens=', '.join(tokens[:30]),
            size=size
        )
        
        try:
            response = self.llm_client.generate(prompt)
            # Parse JSON response
            analysis = json.loads(response)
            return analysis
        except Exception as e:
            logging.error(f"Error analyzing {cluster_key}: {e}")
            # Fallback
            return {
                "label": f"Cluster {cluster_key}",
                "linguistic_type": "unknown",
                "function_or_meaning": "unknown",
                "reasoning": "Analysis failed"
            }
    
    def check_consistency(self, cluster_key, layer, tokens, size, similar_clusters):
        """Check if this cluster matches previous similar ones."""
        if not similar_clusters:
            return None
            
        # Format previous clusters for prompt
        previous_info = []
        for prev_key, prev_data in similar_clusters:
            previous_info.append(
                f"Layer {prev_data['layer']}: {', '.join(prev_data['tokens'][:10])}\n"
                f"Label: {prev_data['label']}\n"
                f"Type: {prev_data['linguistic_type']}"
            )
        
        prompt = CONSISTENCY_CHECK_PROMPT.format(
            previous_clusters='\n\n'.join(previous_info),
            layer=layer,
            tokens=', '.join(tokens[:30]),
            size=size
        )
        
        try:
            response = self.llm_client.generate(prompt)
            result = json.loads(response)
            return result
        except Exception as e:
            logging.error(f"Error checking consistency for {cluster_key}: {e}")
            return None
    
    def find_similar_clusters(self, tokens, current_key):
        """Find clusters with similar token sets."""
        current_set = set(tokens[:10])
        similar = []
        
        for key, data in self.label_cache.items():
            if key != current_key:
                other_set = set(data['tokens'][:10])
                overlap = len(current_set & other_set)
                if overlap >= 6:  # 60% overlap
                    similar.append((key, data))
        
        return similar


def main():
    """Run LLM-based labeling for k=10 clusters."""
    base_dir = Path(__file__).parent
    
    # Load cluster data
    with open(base_dir / "llm_labels_k10" / "llm_labeling_data.json", 'r') as f:
        cluster_data = json.load(f)
    
    # Initialize LLM client
    try:
        llm_client = LLMFactory.create_client('claude')
        logging.info("Using Claude for analysis")
    except:
        logging.warning("Claude not available, using OpenAI")
        llm_client = LLMFactory.create_client('openai')
    
    labeler = LLMLabeler(llm_client)
    
    # Results structure
    results = {
        "metadata": {
            "generated_at": "2025-05-29T12:00:00",
            "model": "llm_analysis",
            "k": 10,
            "total_clusters": 120,
            "method": "llm_based_labeling",
            "description": "Labels generated by LLM analysis of actual cluster contents"
        },
        "labels": {}
    }
    
    # Phase 1: Initial analysis of all clusters
    logging.info("Phase 1: Initial cluster analysis...")
    
    for layer in range(12):
        layer_key = f"layer_{layer}"
        results["labels"][layer_key] = {}
        
        for cluster_idx in range(10):
            cluster_key = f"L{layer}_C{cluster_idx}"
            
            if cluster_key in cluster_data["clusters"]:
                cluster_info = cluster_data["clusters"][cluster_key]
                tokens = cluster_info["common_tokens"]
                size = cluster_info["size"]
                
                logging.info(f"Analyzing {cluster_key}...")
                
                # Get initial analysis
                analysis = labeler.analyze_cluster(cluster_key, layer, tokens, size)
                
                # Cache for consistency checking
                labeler.label_cache[cluster_key] = {
                    'layer': layer,
                    'tokens': tokens,
                    'label': analysis['label'],
                    'linguistic_type': analysis['linguistic_type'],
                    'analysis': analysis
                }
                
                # Store result
                results["labels"][layer_key][cluster_key] = {
                    "label": analysis['label'],
                    "description": f"{analysis['reasoning']}. Examples: {', '.join(tokens[:5])}",
                    "size": size,
                    "percentage": cluster_info["percentage"],
                    "linguistic_type": analysis['linguistic_type'],
                    "function_or_meaning": analysis['function_or_meaning'],
                    "analysis": analysis
                }
    
    # Phase 2: Consistency refinement
    logging.info("\nPhase 2: Checking consistency across similar clusters...")
    
    consistency_updates = []
    
    for layer in range(12):
        layer_key = f"layer_{layer}"
        
        for cluster_idx in range(10):
            cluster_key = f"L{layer}_C{cluster_idx}"
            
            if cluster_key in labeler.label_cache:
                data = labeler.label_cache[cluster_key]
                tokens = data['tokens']
                
                # Find similar clusters
                similar = labeler.find_similar_clusters(tokens, cluster_key)
                
                if similar:
                    logging.info(f"Checking consistency for {cluster_key} ({len(similar)} similar clusters)")
                    
                    consistency = labeler.check_consistency(
                        cluster_key, layer, tokens, 
                        results["labels"][layer_key][cluster_key]["size"],
                        similar
                    )
                    
                    if consistency and consistency.get('matches_previous'):
                        # Update to consistent label
                        old_label = results["labels"][layer_key][cluster_key]["label"]
                        new_label = consistency['matched_label']
                        
                        if old_label != new_label:
                            consistency_updates.append({
                                'cluster': cluster_key,
                                'old': old_label,
                                'new': new_label,
                                'reason': consistency['reasoning']
                            })
                            
                            results["labels"][layer_key][cluster_key]["label"] = new_label
                            results["labels"][layer_key][cluster_key]["consistency_note"] = consistency['reasoning']
    
    # Save results
    output_path = base_dir / "llm_labels_k10" / "cluster_labels_k10_llm_analyzed.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Saved LLM-analyzed labels to {output_path}")
    
    # Generate report
    report = ["LLM-BASED LABELING REPORT", "=" * 60, ""]
    
    # Label distribution
    label_counts = defaultdict(int)
    type_counts = defaultdict(int)
    function_meaning_counts = defaultdict(int)
    
    for layer_data in results["labels"].values():
        for cluster_data in layer_data.values():
            label_counts[cluster_data["label"]] += 1
            type_counts[cluster_data["linguistic_type"]] += 1
            function_meaning_counts[cluster_data["function_or_meaning"]] += 1
    
    report.append("Label Distribution (top 20):")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        report.append(f"  {label}: {count} clusters")
    
    report.append(f"\n\nLinguistic Types:")
    for ltype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        report.append(f"  {ltype}: {count} clusters")
    
    report.append(f"\n\nFunction vs Meaning:")
    for fm, count in function_meaning_counts.items():
        report.append(f"  {fm}: {count} clusters ({count/120*100:.1f}%)")
    
    report.append(f"\n\nConsistency Updates Made: {len(consistency_updates)}")
    for update in consistency_updates[:10]:
        report.append(f"  {update['cluster']}: '{update['old']}' → '{update['new']}'")
    
    # Save report
    report_path = base_dir / "llm_labels_k10" / "llm_labeling_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    # Update main labels file
    import shutil
    shutil.copy(output_path, base_dir / "llm_labels_k10" / "cluster_labels_k10.json")
    
    print("\nLLM-based labeling complete!")
    print(f"Analyzed {len(results['labels']) * 10} clusters")
    print(f"Made {len(consistency_updates)} consistency updates")
    print("\nThe LLM has identified patterns in how GPT-2 organizes tokens.")
    print("Check the report for insights about linguistic types and function/meaning distribution.")


if __name__ == "__main__":
    main()