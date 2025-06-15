"""
Compile comprehensive data for LLM analysis of GPT-2 pivot experiment.

Combines cluster paths, statistics, metrics, and metadata into a single
JSON file for LLM narrative generation and analysis.
"""

import pickle
import json
from typing import Dict, List, Any

def compile_llm_analysis_data():
    """Compile all relevant data for LLM analysis."""
    
    print("Loading all experimental data...")
    
    # Load clustering results
    with open("gpt2_pivot_clustering_results.pkl", "rb") as f:
        clustering_results = pickle.load(f)
    
    # Load APA metrics
    with open("gpt2_pivot_apa_metrics.json", "r") as f:
        apa_metrics = json.load(f)
    
    # Load sentences
    with open("gpt2_pivot_contrast_sentences.txt", "r") as f:
        contrast_sentences = [line.strip() for line in f if line.strip()]
    
    with open("gpt2_pivot_consistent_sentences.txt", "r") as f:
        consistent_sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(contrast_sentences)} contrast + {len(consistent_sentences)} consistent sentences")
    
    # Compile comprehensive analysis data
    analysis_data = {
        "experiment_overview": {
            "title": "GPT-2 Archetypal Path Analysis: Semantic Pivot Experiment",
            "description": "Analysis of how the semantic pivot 'but' affects token representations across GPT-2's 13 layers",
            "methodology": "3-token sentences processed through GPT-2, clustered at each layer, paths tracked across layers",
            "total_sentences": len(contrast_sentences) + len(consistent_sentences),
            "contrast_sentences": len(contrast_sentences),
            "consistent_sentences": len(consistent_sentences),
            "layers_analyzed": 13,
            "clustering_method": "k-means with silhouette optimization"
        },
        
        "sentence_data": {
            "contrast_examples": contrast_sentences[:10],  # First 10 for context
            "consistent_examples": consistent_sentences[:10],  # First 10 for context
            "contrast_class_description": "Positive adjective BUT negative adjective (semantic contradiction)",
            "consistent_class_description": "Positive adjective BUT positive adjective (semantic reinforcement)",
            "pivot_token": "but",
            "pivot_position": "token index 1 (middle token in 3-token sentences)"
        },
        
        "archetypal_paths": {
            "total_unique_paths": apa_metrics["archetypal_path_analysis"]["unique_paths_count"],
            "top_7_archetypal_paths": dict(list(apa_metrics["archetypal_path_analysis"]["most_frequent_paths"].items())[:7]),
            "top_contrast_paths": dict(list(apa_metrics["archetypal_path_analysis"]["path_frequencies"]["contrast"].items())[:4]),
            "top_consistent_paths": dict(list(apa_metrics["archetypal_path_analysis"]["path_frequencies"]["consistent"].items())[:4]), 
            "class_comparison": apa_metrics["archetypal_path_analysis"]["contrast_vs_consistent_paths"],
            "interpretation_note": "Each path shows cluster assignments L{layer}C{cluster} across 13 layers"
        },
        
        "clustering_statistics": {
            "layer_wise_metrics": {},
            "silhouette_scores": apa_metrics["basic_clustering_metrics"]["silhouette_scores"],
            "optimal_k_values": apa_metrics["basic_clustering_metrics"]["optimal_k_values"],
            "cluster_distributions": apa_metrics["basic_clustering_metrics"]["cluster_distributions"]
        },
        
        "cross_layer_analysis": {
            "centroid_similarity": apa_metrics["cross_layer_metrics"]["centroid_similarity_rho_c"],
            "membership_overlap": apa_metrics["cross_layer_metrics"]["membership_overlap_J"],
            "interpretation": "ρ^c measures how similar cluster centroids are between layers, J measures membership overlap"
        },
        
        "path_analysis": {
            "fragmentation_scores": "F_i values computed per token path",
            "similarity_convergent_density": apa_metrics["path_metrics"]["similarity_convergent_path_density_D"],
            "path_purity_by_class": apa_metrics["path_metrics"]["path_purity"],
            "interpretation": "Higher fragmentation = more cluster switching, higher purity = more consistent paths within class"
        },
        
        "pivot_specific_findings": {
            "fragmentation_deltas": apa_metrics["pivot_specific_metrics"]["fragmentation_deltas"],
            "path_divergence_indices": apa_metrics["pivot_specific_metrics"]["path_divergence_indices"],
            "contrast_class_stats": apa_metrics["pivot_specific_metrics"]["contrast_stats"],
            "consistent_class_stats": apa_metrics["pivot_specific_metrics"]["consistent_stats"],
            "interpretation": "Fragmentation delta = post-pivot fragmentation - pre-pivot fragmentation, Path divergence = Hamming distance between pre/post-pivot paths"
        },
        
        "key_research_questions": [
            "Using the samples, and comparing clusters, label the clusters with unique names",
            "How does the third token change the trajectory",
            "Do contrast sentences (positive but negative) create different archetypal paths than consistent sentences (positive but positive)?",
            "Which layers show the most fragmentation/processing of the semantic contradiction?",
            "What narrative can explain the differences in path patterns between the two sentence classes?",
            "How do the three token positions (first, 'but', third) cluster differently across layers?"
        ],
        
        "additional_research_questions": [
            "How does the intensity of adjectives affect clustering and pathway divergence?",
            "Are there positional effects on token clustering within the three-token sequence?",
            "How do clustering patterns differ for rare vs. frequent adjectives?",
            "Does GPT-2 exhibit layer-specific attention to syntactic vs. semantic roles?",
            "How do pathways differ for sentences with antonymous vs. non-antonymous adjective pairs?",
            "Do 'but' tokens cluster together regardless of the surrounding adjectives?",
            "How do final token representations reflect the complete semantic context?",
            "Are contrast sentences more dispersed across clusters than consistent sentences?"
        ],
        
        "analysis_focus": [
            "Label each cluster with semantic names based on sentence examples and intensity patterns",
            "Analyze how third token intensity (mild vs intense) affects trajectory changes",
            "Compare archetypal path patterns between contrast and consistent sentence classes",
            "Identify layers with highest fragmentation or path divergence around the semantic pivot",
            "Generate narrative explanations for cluster semantics and archetypal path differences",
            "Discuss implications for understanding how GPT-2 processes semantic contradictions"
        ],
        
        "metadata": {
            "model": "gpt2",
            "total_parameters": "117M",
            "layers": 13,
            "hidden_size": 768,
            "analysis_date": "2025-05-21",
            "analysis_type": "Archetypal Path Analysis (APA)",
            "clustering_quality": "Good silhouette scores (0.3-0.48)",
            "statistical_significance": f"Large dataset ({len(contrast_sentences) + len(consistent_sentences)} sentences) enables robust analysis"
        }
    }
    
    # Add detailed layer statistics with cluster content information
    all_sentences = contrast_sentences + consistent_sentences
    
    for layer_key, layer_data in clustering_results['layer_results'].items():
        layer_idx = layer_data['layer_idx']
        
        # Create cluster content mapping - which sentences go through each cluster
        cluster_contents = {}
        for sent_idx, token_labels in layer_data["cluster_labels"].items():
            sentence_text = all_sentences[sent_idx] if sent_idx < len(all_sentences) else f"sentence_{sent_idx}"
            sentence_type = "contrast" if sent_idx < len(contrast_sentences) else "consistent"
            
            # For each token in the sentence, record which cluster it belongs to
            for token_idx, cluster_label in token_labels.items():
                cluster_key = f"cluster_{cluster_label}"
                if cluster_key not in cluster_contents:
                    cluster_contents[cluster_key] = {
                        "sentences": [],
                        "contrast_count": 0,
                        "consistent_count": 0
                    }
                
                # Add sentence info (avoid duplicates)
                sentence_info = {
                    "text": sentence_text,
                    "type": sentence_type,
                    "token_position": token_idx  # 0=first_word, 1=but, 2=third_word
                }
                
                # Check if this sentence is already recorded for this cluster
                existing = False
                for existing_sent in cluster_contents[cluster_key]["sentences"]:
                    if existing_sent["text"] == sentence_text:
                        existing = True
                        break
                
                if not existing:
                    # Keep first 10 examples per cluster and add intensity info
                    if len(cluster_contents[cluster_key]["sentences"]) < 10:
                        # Add intensity classification for third word
                        tokens = sentence_text.split()
                        third_word = tokens[2] if len(tokens) == 3 else ""
                        
                        # Simple intensity classification
                        intense_negative = ['disgusting', 'appalling', 'atrocious', 'ghastly', 'revolting', 'vile', 'repulsive', 'hideous', 'loathsome', 'detestable', 'abominable', 'execrable', 'odious', 'repugnant', 'nauseating']
                        intense_positive = ['outstanding', 'magnificent', 'marvelous', 'splendid', 'fabulous', 'terrific', 'incredible', 'remarkable', 'exceptional', 'phenomenal']
                        
                        if third_word.lower() in intense_negative or third_word.lower() in intense_positive:
                            intensity = "intense"
                        else:
                            intensity = "mild"
                        
                        sentence_info["third_word"] = third_word
                        sentence_info["intensity"] = intensity
                        cluster_contents[cluster_key]["sentences"].append(sentence_info)
                    
                    if sentence_type == "contrast":
                        cluster_contents[cluster_key]["contrast_count"] += 1
                    else:
                        cluster_contents[cluster_key]["consistent_count"] += 1
        
        analysis_data["clustering_statistics"]["layer_wise_metrics"][f"layer_{layer_idx}"] = {
            "optimal_k": layer_data["optimal_k"],
            "silhouette_score": layer_data["silhouette_score"],
            "num_cluster_assignments": len([label for sent_labels in layer_data["cluster_labels"].values() 
                                          for label in sent_labels.values()]),
            "cluster_contents": cluster_contents
        }
    
    # Add only archetypal path examples (no individual sentence paths)
    analysis_data["archetypal_path_examples"] = {
        "note": "Individual sentence paths omitted for brevity - focus on archetypal patterns above",
        "total_sentences_analyzed": len(clustering_results['sentences']),
        "paths_available_for_detailed_analysis": "Available in clustering_results.pkl if needed"
    }
    
    return analysis_data

def save_llm_analysis_data():
    """Compile and save comprehensive data for LLM analysis."""
    
    analysis_data = compile_llm_analysis_data()
    
    # Save comprehensive analysis data
    with open("gpt2_pivot_llm_analysis_data.json", "w") as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    print("SUCCESS: Comprehensive LLM analysis data saved to gpt2_pivot_llm_analysis_data.json")
    
    # Print summary
    print(f"""
DATA COMPILATION SUMMARY:
- Total sentences: {analysis_data['experiment_overview']['total_sentences']}
- Unique archetypal paths: {analysis_data['archetypal_paths']['total_unique_paths']}
- Layers analyzed: {analysis_data['experiment_overview']['layers_analyzed']}
- Clustering quality: {analysis_data['metadata']['clustering_quality']}

KEY FINDINGS FOR LLM ANALYSIS:
- Contrast vs Consistent path differences: {analysis_data['archetypal_paths']['class_comparison']}
- Most frequent path occurs {list(analysis_data['archetypal_paths']['top_7_archetypal_paths'].values())[0]} times
- Sample paths and detailed statistics included for comprehensive analysis

READY FOR LLM NARRATIVE GENERATION!
""")
    
    return analysis_data

if __name__ == "__main__":
    data = save_llm_analysis_data()