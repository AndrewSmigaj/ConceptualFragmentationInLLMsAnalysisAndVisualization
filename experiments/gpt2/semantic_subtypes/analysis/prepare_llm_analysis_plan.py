#!/usr/bin/env python3
"""
Plan and prepare data for LLM analysis of GPT-2 semantic subtype clustering.
Creates structured data files that capture key patterns for analysis.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple

class LLMAnalysisPreparation:
    def __init__(self, result_dir: str = "semantic_subtypes_optimal_experiment_20250523_182344"):
        self.result_dir = Path(result_dir)
        self.output_dir = Path("llm_analysis_data")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_all_data(self) -> Tuple[Dict, Dict, Dict]:
        """Load all necessary data for analysis."""
        with open(self.result_dir / "semantic_subtypes_kmeans_optimal.pkl", 'rb') as f:
            kmeans_results = pickle.load(f)
        
        with open(self.result_dir / "semantic_subtypes_ets_optimal.pkl", 'rb') as f:
            ets_results = pickle.load(f)
        
        with open("data/gpt2_semantic_subtypes_curated.json", 'r') as f:
            curated_data = json.load(f)
        
        return kmeans_results, ets_results, curated_data
    
    def prepare_analysis_structure(self):
        """Create the overall structure for LLM analysis."""
        
        analysis_plan = {
            "analysis_components": {
                "1_cluster_interpretation": {
                    "goal": "Understand what semantic properties unite words in each cluster",
                    "key_questions": [
                        "What common semantic features do words in each cluster share?",
                        "How do these features differ from our predefined semantic subtypes?",
                        "Are there emergent semantic categories that GPT-2 has learned?"
                    ],
                    "data_needed": [
                        "cluster_contents_by_layer.json",
                        "cluster_semantic_profiles.json",
                        "unexpected_groupings.json"
                    ]
                },
                
                "2_layer_evolution": {
                    "goal": "Track how semantic organization evolves across layers",
                    "key_questions": [
                        "How does semantic granularity change from early to late layers?",
                        "Which layers show the most dramatic reorganization?",
                        "What semantic distinctions emerge or disappear across layers?"
                    ],
                    "data_needed": [
                        "semantic_trajectory_analysis.json",
                        "layer_transition_patterns.json",
                        "semantic_convergence_divergence.json"
                    ]
                },
                
                "3_method_comparison": {
                    "goal": "Compare K-means vs ETS clustering interpretations",
                    "key_questions": [
                        "Do K-means and ETS reveal different semantic organizations?",
                        "Which method better captures linguistic intuitions?",
                        "What do the differences tell us about geometric vs threshold-based clustering?"
                    ],
                    "data_needed": [
                        "method_agreement_analysis.json",
                        "method_specific_patterns.json"
                    ]
                },
                
                "4_outlier_analysis": {
                    "goal": "Understand words that don't fit expected patterns",
                    "key_questions": [
                        "Why is 'depend' consistently an outlier?",
                        "What makes certain words cluster unexpectedly?",
                        "Do outliers reveal special properties of GPT-2's representations?"
                    ],
                    "data_needed": [
                        "outlier_words_analysis.json",
                        "singleton_clusters.json"
                    ]
                },
                
                "5_semantic_hypothesis": {
                    "goal": "Form hypotheses about GPT-2's semantic organization",
                    "key_questions": [
                        "What alternative semantic taxonomy does GPT-2 use?",
                        "How does distributional learning create different categories than human intuitions?",
                        "What implications does this have for understanding LLM representations?"
                    ],
                    "data_needed": [
                        "emergent_categories.json",
                        "cross_subtype_patterns.json"
                    ]
                }
            },
            
            "analysis_workflow": [
                "1. Start with cluster interpretation at key layers (0, 5, 10)",
                "2. Identify emergent semantic patterns",
                "3. Track pattern evolution across layers",
                "4. Compare K-means and ETS interpretations",
                "5. Analyze outliers and unexpected groupings",
                "6. Synthesize findings into semantic organization hypothesis"
            ]
        }
        
        # Save analysis plan
        with open(self.output_dir / "llm_analysis_plan.json", 'w') as f:
            json.dump(analysis_plan, f, indent=2)
        
        print(f"Analysis plan saved to: {self.output_dir / 'llm_analysis_plan.json'}")
        
        return analysis_plan
    
    def prepare_cluster_contents_data(self, kmeans_results, ets_results, curated_data):
        """Prepare detailed cluster contents for each layer."""
        
        print("\nPreparing cluster contents data...")
        
        # Create word to subtype mapping
        word_to_subtype = {}
        for subtype, words in curated_data['curated_words'].items():
            for word in words:
                word_to_subtype[word] = subtype
        
        cluster_contents = {
            "description": "Detailed contents of each cluster at each layer",
            "total_words": len(kmeans_results['sentences']),
            "semantic_subtypes": list(curated_data['curated_words'].keys()),
            "layers": {}
        }
        
        # Key layers for detailed analysis
        key_layers = [0, 1, 2, 5, 7, 10, 11, 12]
        
        for layer_idx in key_layers:
            layer_key = f"layer_{layer_idx}"
            
            cluster_contents["layers"][layer_key] = {
                "layer_description": self._get_layer_description(layer_idx),
                "kmeans": self._extract_cluster_details(
                    kmeans_results, layer_idx, word_to_subtype
                ),
                "ets": self._extract_cluster_details(
                    ets_results, layer_idx, word_to_subtype
                )
            }
        
        # Save cluster contents
        with open(self.output_dir / "cluster_contents_by_layer.json", 'w') as f:
            json.dump(cluster_contents, f, indent=2)
        
        print(f"Cluster contents saved to: {self.output_dir / 'cluster_contents_by_layer.json'}")
        
        return cluster_contents
    
    def _get_layer_description(self, layer_idx: int) -> str:
        """Get semantic description of what each layer typically represents."""
        descriptions = {
            0: "Input embedding layer - captures surface-level lexical features",
            1: "Early layer - basic syntactic and morphological patterns",
            2: "Early layer - initial semantic groupings",
            5: "Middle layer - semantic role and category emergence",
            7: "Middle layer - refined semantic distinctions",
            10: "Late layer - abstract semantic representations",
            11: "Late layer - task-oriented semantic organization",
            12: "Output layer - prediction-focused representations"
        }
        return descriptions.get(layer_idx, f"Layer {layer_idx}")
    
    def _extract_cluster_details(self, clustering_results, layer_idx, word_to_subtype):
        """Extract detailed information about clusters at a specific layer."""
        
        layer_key = f"layer_{layer_idx}"
        if layer_key not in clustering_results['layer_results']:
            return {}
        
        layer_data = clustering_results['layer_results'][layer_key]
        sentences = clustering_results['sentences']
        
        # Group words by cluster
        clusters = defaultdict(lambda: {
            'words': [],
            'subtype_distribution': defaultdict(int),
            'words_by_subtype': defaultdict(list)
        })
        
        for sent_idx, cluster_labels in layer_data['cluster_labels'].items():
            if 0 in cluster_labels:  # Token position 0
                cluster_id = cluster_labels[0]
                word = sentences[sent_idx]
                subtype = word_to_subtype.get(word, 'unknown')
                
                clusters[cluster_id]['words'].append(word)
                clusters[cluster_id]['subtype_distribution'][subtype] += 1
                clusters[cluster_id]['words_by_subtype'][subtype].append(word)
        
        # Format cluster information
        formatted_clusters = {}
        
        for cluster_id, cluster_data in sorted(clusters.items()):
            # Calculate cluster statistics
            total_words = len(cluster_data['words'])
            dominant_subtype = max(
                cluster_data['subtype_distribution'].items(),
                key=lambda x: x[1]
            )[0] if cluster_data['subtype_distribution'] else 'unknown'
            
            purity = (cluster_data['subtype_distribution'][dominant_subtype] / total_words 
                     if total_words > 0 else 0)
            
            # Get top words by frequency in cluster
            word_counts = Counter(cluster_data['words'])
            
            formatted_clusters[f"cluster_{cluster_id}"] = {
                "size": total_words,
                "purity": round(purity, 3),
                "dominant_subtype": dominant_subtype,
                "subtype_distribution": dict(cluster_data['subtype_distribution']),
                "sample_words": {
                    subtype: sorted(words)[:10]
                    for subtype, words in cluster_data['words_by_subtype'].items()
                },
                "all_words": sorted(cluster_data['words'])
            }
        
        return formatted_clusters
    
    def prepare_semantic_profiles(self, kmeans_results, curated_data):
        """Create semantic profiles showing how each subtype distributes across clusters."""
        
        print("\nPreparing semantic profiles...")
        
        word_to_subtype = {}
        for subtype, words in curated_data['curated_words'].items():
            for word in words:
                word_to_subtype[word] = subtype
        
        semantic_profiles = {
            "description": "How each semantic subtype distributes across clusters at each layer",
            "subtypes": {}
        }
        
        for subtype in curated_data['curated_words'].keys():
            subtype_profile = {
                "total_words": len(curated_data['curated_words'][subtype]),
                "example_words": sorted(curated_data['curated_words'][subtype])[:10],
                "layer_distributions": {}
            }
            
            # Track distribution across layers
            for layer_idx in range(13):
                layer_key = f"layer_{layer_idx}"
                layer_data = kmeans_results['layer_results'][layer_key]
                
                # Count how subtype distributes across clusters
                cluster_distribution = defaultdict(int)
                
                for sent_idx, word in enumerate(kmeans_results['sentences']):
                    if word_to_subtype.get(word) == subtype:
                        if sent_idx in layer_data['cluster_labels'] and 0 in layer_data['cluster_labels'][sent_idx]:
                            cluster_id = layer_data['cluster_labels'][sent_idx][0]
                            cluster_distribution[cluster_id] += 1
                
                subtype_profile["layer_distributions"][layer_key] = dict(cluster_distribution)
            
            semantic_profiles["subtypes"][subtype] = subtype_profile
        
        # Save semantic profiles
        with open(self.output_dir / "cluster_semantic_profiles.json", 'w') as f:
            json.dump(semantic_profiles, f, indent=2)
        
        print(f"Semantic profiles saved to: {self.output_dir / 'cluster_semantic_profiles.json'}")
        
        return semantic_profiles
    
    def identify_unexpected_groupings(self, kmeans_results, curated_data):
        """Identify words that cluster together unexpectedly."""
        
        print("\nIdentifying unexpected groupings...")
        
        word_to_subtype = {}
        for subtype, words in curated_data['curated_words'].items():
            for word in words:
                word_to_subtype[word] = subtype
        
        unexpected_groupings = {
            "description": "Words from different semantic subtypes that consistently cluster together",
            "consistent_pairs": [],
            "layer_specific_surprises": {}
        }
        
        # Find word pairs that cluster together across multiple layers
        word_pairs = defaultdict(int)
        
        for layer_idx in range(13):
            layer_key = f"layer_{layer_idx}"
            layer_data = kmeans_results['layer_results'][layer_key]
            
            # Build cluster membership
            cluster_members = defaultdict(list)
            for sent_idx, clusters in layer_data['cluster_labels'].items():
                if 0 in clusters:
                    word = kmeans_results['sentences'][sent_idx]
                    subtype = word_to_subtype.get(word, 'unknown')
                    cluster_members[clusters[0]].append((word, subtype))
            
            # Find surprising pairs in each cluster
            layer_surprises = []
            
            for cluster_id, members in cluster_members.items():
                # Look for words from different subtypes
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        word1, subtype1 = members[i]
                        word2, subtype2 = members[j]
                        
                        if subtype1 != subtype2:
                            pair = tuple(sorted([word1, word2]))
                            word_pairs[pair] += 1
                            
                            if layer_idx in [0, 5, 10]:  # Key layers
                                layer_surprises.append({
                                    "cluster": cluster_id,
                                    "word1": word1,
                                    "subtype1": subtype1,
                                    "word2": word2,
                                    "subtype2": subtype2
                                })
            
            if layer_surprises and layer_idx in [0, 5, 10]:
                unexpected_groupings["layer_specific_surprises"][layer_key] = layer_surprises[:10]
        
        # Find consistent unexpected pairs
        consistent_threshold = 8  # Appear together in at least 8 layers
        
        for (word1, word2), count in word_pairs.items():
            if count >= consistent_threshold:
                subtype1 = word_to_subtype.get(word1, 'unknown')
                subtype2 = word_to_subtype.get(word2, 'unknown')
                
                unexpected_groupings["consistent_pairs"].append({
                    "words": [word1, word2],
                    "subtypes": [subtype1, subtype2],
                    "layers_together": count,
                    "similarity_hypothesis": "TODO: Analyze why these cluster together"
                })
        
        # Sort by frequency
        unexpected_groupings["consistent_pairs"].sort(
            key=lambda x: x["layers_together"], reverse=True
        )
        
        # Save unexpected groupings
        with open(self.output_dir / "unexpected_groupings.json", 'w') as f:
            json.dump(unexpected_groupings, f, indent=2)
        
        print(f"Unexpected groupings saved to: {self.output_dir / 'unexpected_groupings.json'}")
        
        return unexpected_groupings
    
    def analyze_outliers(self, kmeans_results, ets_results, curated_data):
        """Analyze outlier words and singleton clusters."""
        
        print("\nAnalyzing outliers...")
        
        word_to_subtype = {}
        for subtype, words in curated_data['curated_words'].items():
            for word in words:
                word_to_subtype[word] = subtype
        
        outlier_analysis = {
            "description": "Analysis of outlier words and singleton clusters",
            "singleton_clusters": {
                "kmeans": {},
                "ets": {}
            },
            "consistent_outliers": []
        }
        
        # Track which words are often in small clusters
        outlier_counts = defaultdict(int)
        
        for method_name, results in [("kmeans", kmeans_results), ("ets", ets_results)]:
            for layer_idx in range(13):
                layer_key = f"layer_{layer_idx}"
                layer_data = results['layer_results'][layer_key]
                
                # Count cluster sizes
                cluster_sizes = defaultdict(int)
                cluster_words = defaultdict(list)
                
                for sent_idx, clusters in layer_data['cluster_labels'].items():
                    if 0 in clusters:
                        cluster_id = clusters[0]
                        cluster_sizes[cluster_id] += 1
                        word = results['sentences'][sent_idx]
                        cluster_words[cluster_id].append(word)
                
                # Find singleton and small clusters
                for cluster_id, size in cluster_sizes.items():
                    if size <= 3:  # Small cluster
                        words_in_cluster = cluster_words[cluster_id]
                        
                        outlier_analysis["singleton_clusters"][method_name][f"{layer_key}_cluster_{cluster_id}"] = {
                            "size": size,
                            "words": words_in_cluster,
                            "subtypes": [word_to_subtype.get(w, 'unknown') for w in words_in_cluster]
                        }
                        
                        # Track outlier frequency
                        for word in words_in_cluster:
                            outlier_counts[word] += 1
        
        # Find consistent outliers
        for word, count in sorted(outlier_counts.items(), key=lambda x: -x[1]):
            if count >= 5:  # Outlier in at least 5 layer/method combinations
                outlier_analysis["consistent_outliers"].append({
                    "word": word,
                    "subtype": word_to_subtype.get(word, 'unknown'),
                    "outlier_frequency": count,
                    "analysis_needed": f"Why is '{word}' consistently an outlier?"
                })
        
        # Save outlier analysis
        with open(self.output_dir / "outlier_words_analysis.json", 'w') as f:
            json.dump(outlier_analysis, f, indent=2)
        
        print(f"Outlier analysis saved to: {self.output_dir / 'outlier_words_analysis.json'}")
        
        return outlier_analysis
    
    def prepare_layer_transition_analysis(self, kmeans_results, curated_data):
        """Analyze how words transition between clusters across layers."""
        
        print("\nPreparing layer transition analysis...")
        
        word_to_subtype = {}
        for subtype, words in curated_data['curated_words'].items():
            for word in words:
                word_to_subtype[word] = subtype
        
        transition_analysis = {
            "description": "Analysis of how words move between clusters across layers",
            "transition_patterns": [],
            "stability_by_subtype": {}
        }
        
        # Track transitions for each word
        word_paths = {}
        sentences = kmeans_results['sentences']
        
        for sent_idx, word in enumerate(sentences):
            path = []
            for layer_idx in range(13):
                layer_key = f"layer_{layer_idx}"
                if sent_idx in kmeans_results['layer_results'][layer_key]['cluster_labels']:
                    if 0 in kmeans_results['layer_results'][layer_key]['cluster_labels'][sent_idx]:
                        cluster_id = kmeans_results['layer_results'][layer_key]['cluster_labels'][sent_idx][0]
                        path.append(cluster_id)
            
            if len(path) == 13:
                word_paths[word] = path
        
        # Analyze transition patterns
        for window_start in [0, 4, 8]:
            window_end = min(window_start + 4, 12)
            window_name = f"layers_{window_start}-{window_end}"
            
            transitions = defaultdict(int)
            
            for word, path in word_paths.items():
                # Count transitions in this window
                for i in range(window_start, window_end):
                    transition = f"C{path[i]}->C{path[i+1]}"
                    transitions[transition] += 1
            
            # Find most common transitions
            common_transitions = sorted(transitions.items(), key=lambda x: -x[1])[:10]
            
            transition_analysis["transition_patterns"].append({
                "window": window_name,
                "common_transitions": [
                    {"transition": t, "count": c} for t, c in common_transitions
                ]
            })
        
        # Calculate stability by subtype
        for subtype in curated_data['curated_words'].keys():
            subtype_words = curated_data['curated_words'][subtype]
            
            stability_scores = []
            for word in subtype_words:
                if word in word_paths:
                    path = word_paths[word]
                    # Count how many times cluster changes
                    changes = sum(1 for i in range(len(path)-1) if path[i] != path[i+1])
                    stability = 1 - (changes / (len(path) - 1))
                    stability_scores.append(stability)
            
            if stability_scores:
                transition_analysis["stability_by_subtype"][subtype] = {
                    "avg_stability": round(np.mean(stability_scores), 3),
                    "std_stability": round(np.std(stability_scores), 3)
                }
        
        # Save transition analysis
        with open(self.output_dir / "layer_transition_patterns.json", 'w') as f:
            json.dump(transition_analysis, f, indent=2)
        
        print(f"Transition analysis saved to: {self.output_dir / 'layer_transition_patterns.json'}")
        
        return transition_analysis
    
    def generate_llm_prompts(self):
        """Generate specific prompts for LLM analysis."""
        
        print("\nGenerating LLM analysis prompts...")
        
        prompts = {
            "cluster_interpretation_prompts": [
                {
                    "task": "semantic_feature_discovery",
                    "prompt": """Looking at the words in each cluster at layer {layer}, identify the common semantic features that unite them. 
                    Consider features beyond our predefined categories like:
                    - Abstractness/concreteness
                    - Animacy
                    - Agency
                    - Temporal properties
                    - Emotional valence
                    - Syntactic flexibility
                    - Frequency of use
                    - Morphological complexity
                    
                    For each cluster, provide:
                    1. Primary semantic feature(s)
                    2. Secondary features
                    3. Why these features might be computationally relevant for GPT-2"""
                },
                
                {
                    "task": "emergent_categories",
                    "prompt": """Based on the clustering patterns across layers, what emergent semantic categories has GPT-2 learned that differ from our linguistic intuitions?
                    
                    Consider:
                    1. Categories that span traditional part-of-speech boundaries
                    2. Distributional patterns that create unexpected groupings
                    3. Functional categories based on typical contexts
                    4. How these might optimize for next-token prediction"""
                }
            ],
            
            "outlier_analysis_prompts": [
                {
                    "task": "outlier_explanation",
                    "prompt": """Analyze why '{word}' is consistently an outlier across layers. Consider:
                    1. Unique distributional properties
                    2. Multiple meanings or uses
                    3. Syntactic flexibility
                    4. Frequency effects
                    5. Morphological uniqueness"""
                }
            ],
            
            "synthesis_prompts": [
                {
                    "task": "semantic_organization_hypothesis",
                    "prompt": """Based on all the clustering patterns, propose a hypothesis for how GPT-2 organizes semantic information across its layers.
                    
                    Address:
                    1. What principles govern the organization at early vs. late layers?
                    2. How does this differ from human semantic intuitions?
                    3. What does this reveal about how transformers learn meaning?
                    4. Implications for understanding LLM representations"""
                }
            ]
        }
        
        # Save prompts
        with open(self.output_dir / "llm_analysis_prompts.json", 'w') as f:
            json.dump(prompts, f, indent=2)
        
        print(f"LLM prompts saved to: {self.output_dir / 'llm_analysis_prompts.json'}")
        
        return prompts
    
    def create_analysis_summary(self):
        """Create a summary document for the LLM analysis."""
        
        print("\nCreating analysis summary...")
        
        summary = f"""# GPT-2 Semantic Subtypes Clustering Analysis Summary

## Overview
This analysis examines how GPT-2 organizes 774 single-token words from 8 semantic subtypes across its 13 layers using optimal clustering configurations determined by elbow method.

## Data Prepared for Analysis

### 1. Cluster Contents (`cluster_contents_by_layer.json`)
- Detailed word lists for each cluster at key layers (0, 1, 2, 5, 7, 10, 11, 12)
- Subtype distribution within each cluster
- Purity scores and dominant subtypes

### 2. Semantic Profiles (`cluster_semantic_profiles.json`)
- How each semantic subtype distributes across clusters
- Layer-by-layer evolution of subtype clustering

### 3. Unexpected Groupings (`unexpected_groupings.json`)
- Word pairs from different subtypes that consistently cluster together
- Layer-specific surprising combinations

### 4. Outlier Analysis (`outlier_words_analysis.json`)
- Singleton clusters and their contents
- Consistently outlying words like 'depend'

### 5. Transition Patterns (`layer_transition_patterns.json`)
- How words move between clusters across layers
- Stability analysis by semantic subtype

## Key Questions for LLM Analysis

1. **Cluster Interpretation**: What semantic features unite words in each cluster that transcend our predefined categories?

2. **Emergent Organization**: What alternative semantic taxonomy has GPT-2 learned through distributional patterns?

3. **Layer Evolution**: How does semantic granularity and organization change across layers?

4. **Outlier Insights**: What makes certain words (especially 'depend') consistently outliers?

5. **Theoretical Implications**: What does this reveal about how transformers learn and organize meaning?

## Analysis Approach

1. Start with cluster interpretation at key layers
2. Identify emergent semantic patterns
3. Track evolution across layers
4. Analyze outliers and unexpected groupings
5. Synthesize findings into a theory of GPT-2's semantic organization

## Files Generated
- `llm_analysis_plan.json`: Structured analysis plan
- `cluster_contents_by_layer.json`: Main data file with cluster contents
- `cluster_semantic_profiles.json`: Subtype distribution analysis
- `unexpected_groupings.json`: Surprising word combinations
- `outlier_words_analysis.json`: Outlier and singleton analysis
- `layer_transition_patterns.json`: Cluster transition patterns
- `llm_analysis_prompts.json`: Specific prompts for analysis
"""
        
        # Save summary
        with open(self.output_dir / "analysis_summary.md", 'w') as f:
            f.write(summary)
        
        print(f"Analysis summary saved to: {self.output_dir / 'analysis_summary.md'}")
        
        return summary

def main():
    """Run the complete LLM analysis preparation."""
    
    print("="*60)
    print("Preparing LLM Analysis Data for GPT-2 Semantic Subtypes")
    print("="*60)
    
    prep = LLMAnalysisPreparation()
    
    # Load data
    print("\nLoading clustering results...")
    kmeans_results, ets_results, curated_data = prep.load_all_data()
    
    # 1. Create analysis structure
    analysis_plan = prep.prepare_analysis_structure()
    
    # 2. Prepare cluster contents
    cluster_contents = prep.prepare_cluster_contents_data(kmeans_results, ets_results, curated_data)
    
    # 3. Create semantic profiles
    semantic_profiles = prep.prepare_semantic_profiles(kmeans_results, curated_data)
    
    # 4. Identify unexpected groupings
    unexpected_groupings = prep.identify_unexpected_groupings(kmeans_results, curated_data)
    
    # 5. Analyze outliers
    outlier_analysis = prep.analyze_outliers(kmeans_results, ets_results, curated_data)
    
    # 6. Prepare transition analysis
    transition_analysis = prep.prepare_layer_transition_analysis(kmeans_results, curated_data)
    
    # 7. Generate LLM prompts
    prompts = prep.generate_llm_prompts()
    
    # 8. Create summary
    summary = prep.create_analysis_summary()
    
    print("\n" + "="*60)
    print("LLM Analysis Preparation Complete!")
    print("="*60)
    print(f"\nAll files saved to: llm_analysis_data/")
    print("\nNext steps:")
    print("1. Review the analysis_summary.md")
    print("2. Start with cluster_contents_by_layer.json for initial interpretation")
    print("3. Use the prompts in llm_analysis_prompts.json to guide analysis")
    print("4. Look for emergent patterns that explain GPT-2's semantic organization")

if __name__ == "__main__":
    main()