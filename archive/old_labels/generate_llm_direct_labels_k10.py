#!/usr/bin/env python3
"""
Generate labels for k=10 clusters using direct LLM analysis.
This follows the established pattern of using the ClusterAnalysis API for reproducible LLM-based labeling.
"""

import json
from pathlib import Path
import logging
import sys
from collections import defaultdict
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from concept_fragmentation.llm.analysis import ClusterAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_token_cluster_profile(cluster_key, tokens, size, percentage, layer):
    """
    Build a profile string for a token cluster that works with the ClusterAnalysis API.
    
    The API expects "demographic/statistical profiles" but we have token clusters.
    We'll adapt by presenting token information as linguistic characteristics.
    """
    # Analyze token characteristics
    token_types = analyze_token_types(tokens[:100])  # Analyze more tokens for better profile
    
    profile_lines = [
        f"Token Cluster Analysis (Layer {layer}/11):",
        f"Total tokens: {size} ({percentage:.1f}% of vocabulary)",
        "",
        "Token Category Distribution:",
    ]
    
    # Add type distribution
    for token_type, count in sorted(token_types.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = (count / sum(token_types.values())) * 100
            profile_lines.append(f"- {token_type}: {count} tokens ({pct:.1f}%)")
    
    profile_lines.extend([
        "",
        "Representative Token Sample:",
        f"{', '.join(repr(t) for t in tokens[:30])}",
        "",
        "Linguistic Characteristics:"
    ])
    
    # Analyze linguistic patterns
    patterns = analyze_linguistic_patterns(tokens[:50])
    for pattern, description in patterns.items():
        if description:
            profile_lines.append(f"- {pattern}: {description}")
    
    return '\n'.join(profile_lines)


def analyze_token_types(tokens):
    """Categorize tokens into linguistic types."""
    types = {
        'function_words': 0,
        'content_words': 0,
        'punctuation': 0,
        'special_tokens': 0,
        'word_parts': 0,
        'numbers': 0,
        'proper_nouns': 0
    }
    
    # Common function words for reference
    function_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
        'me', 'when', 'make', 'can', 'like', 'no', 'just', 'him', 'know', 'take',
        'into', 'your', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
        'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back',
        'after', 'use', 'two', 'how', 'our', 'well', 'way', 'even', 'new', 'want',
        'because', 'any', 'these', 'us', 'is', 'was', 'are', 'been', 'has', 'had',
        'were', 'said', 'did', 'get', 'may', 'am', 'de', 'en', 'un', ''
    }
    
    for token in tokens:
        # Clean token
        t = token.strip().lower()
        
        # Check categories
        if t in function_words:
            types['function_words'] += 1
        elif len(t) == 1 and not t.isalnum():
            types['punctuation'] += 1
        elif t.startswith('Ġ') or t.startswith('Ċ'):  # GPT-2 special tokens
            types['special_tokens'] += 1
        elif t.startswith('##') or (len(t) > 1 and not t[0].isalpha()):  # Word parts
            types['word_parts'] += 1
        elif t.isdigit() or any(c.isdigit() for c in t):
            types['numbers'] += 1
        elif t and t[0].isupper():
            types['proper_nouns'] += 1
        else:
            types['content_words'] += 1
    
    return types


def analyze_linguistic_patterns(tokens):
    """Analyze linguistic patterns in the token set."""
    patterns = {
        'grammatical_role': '',
        'morphological_pattern': '',
        'semantic_field': '',
        'position_tendency': ''
    }
    
    # Simplified analysis based on token characteristics
    clean_tokens = [t.strip().lower() for t in tokens]
    
    # Grammatical role analysis
    if sum(1 for t in clean_tokens if t in {'the', 'a', 'an', 'this', 'that', 'these', 'those'}) > 5:
        patterns['grammatical_role'] = 'Determiners and articles'
    elif sum(1 for t in clean_tokens if t in {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am'}) > 5:
        patterns['grammatical_role'] = 'Auxiliary verbs'
    elif sum(1 for t in clean_tokens if t in {'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of'}) > 5:
        patterns['grammatical_role'] = 'Prepositions'
    elif sum(1 for t in clean_tokens if t in {'I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her'}) > 5:
        patterns['grammatical_role'] = 'Personal pronouns'
    
    # Morphological patterns
    if sum(1 for t in tokens if t.endswith('ing')) > 5:
        patterns['morphological_pattern'] = 'Present participles (-ing forms)'
    elif sum(1 for t in tokens if t.endswith('ed')) > 5:
        patterns['morphological_pattern'] = 'Past tense/participles (-ed forms)'
    elif sum(1 for t in tokens if t.endswith('ly')) > 5:
        patterns['morphological_pattern'] = 'Adverbs (-ly forms)'
    elif sum(1 for t in tokens if t.endswith('s') or t.endswith('es')) > 10:
        patterns['morphological_pattern'] = 'Plural/3rd person forms'
    
    # Semantic field (very basic)
    if any(t in clean_tokens for t in ['time', 'day', 'year', 'hour', 'minute', 'week', 'month']):
        patterns['semantic_field'] = 'Temporal concepts'
    elif any(t in clean_tokens for t in ['place', 'location', 'area', 'region', 'city', 'country']):
        patterns['semantic_field'] = 'Spatial concepts'
    elif any(t in clean_tokens for t in ['people', 'person', 'man', 'woman', 'child', 'human']):
        patterns['semantic_field'] = 'Human entities'
    
    # Position tendency
    sentence_starters = {'The', 'A', 'In', 'On', 'He', 'She', 'It', 'They', 'We', 'I'}
    if sum(1 for t in tokens if t in sentence_starters) > 5:
        patterns['position_tendency'] = 'Often sentence-initial'
    
    return patterns


def find_similar_clusters(current_tokens, all_clusters, current_key, threshold=0.5):
    """
    Find clusters with high token overlap using Jaccard similarity.
    """
    current_set = set(current_tokens[:30])  # Use top 30 for comparison
    similar = []
    
    for other_key, other_data in all_clusters.items():
        if other_key != current_key:
            other_tokens = other_data['tokens']
            other_set = set(other_tokens[:30])
            
            # Calculate Jaccard similarity
            intersection = len(current_set & other_set)
            union = len(current_set | other_set)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > threshold:
                similar.append({
                    'key': other_key,
                    'similarity': similarity,
                    'tokens': other_tokens,
                    'layer': other_data['layer']
                })
    
    return sorted(similar, key=lambda x: x['similarity'], reverse=True)


def main():
    """Run direct LLM-based labeling for k=10 clusters."""
    base_dir = Path(__file__).parent
    
    # Load cluster data
    input_path = base_dir / "llm_labels_k10" / "llm_labeling_data.json"
    with open(input_path, 'r') as f:
        cluster_data = json.load(f)
    
    # Initialize LLM analysis with Claude
    logging.info("Initializing ClusterAnalysis with Claude...")
    try:
        analyzer = ClusterAnalysis(
            provider="claude",
            model="claude-3-opus-20240229",
            use_cache=True,
            cache_dir=str(base_dir / "cache" / "llm"),
            optimize_prompts=False,
            debug=False
        )
        logging.info(f"Successfully initialized {analyzer.provider} with model {analyzer.model}")
    except Exception as e:
        logging.error(f"Failed to initialize Claude: {e}")
        logging.info("Falling back to OpenAI...")
        analyzer = ClusterAnalysis(
            provider="openai",
            model="gpt-4",
            use_cache=True,
            cache_dir=str(base_dir / "cache" / "llm"),
            optimize_prompts=False,
            debug=False
        )
        logging.info(f"Successfully initialized {analyzer.provider} with model {analyzer.model}")
    
    # Results structure
    results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": analyzer.model,
            "provider": analyzer.provider,
            "k": 10,
            "total_clusters": 120,
            "method": "direct_llm_analysis",
            "description": "Labels generated by direct LLM analysis using established ClusterAnalysis API"
        },
        "labels": {}
    }
    
    # Build profiles for all clusters
    logging.info("Building linguistic profiles for all clusters...")
    cluster_profiles = {}
    all_clusters_data = {}
    
    for layer in range(12):
        layer_key = f"layer_{layer}"
        results["labels"][layer_key] = {}
        
        for cluster_idx in range(10):
            cluster_key = f"L{layer}_C{cluster_idx}"
            
            if cluster_key in cluster_data["clusters"]:
                cluster_info = cluster_data["clusters"][cluster_key]
                tokens = cluster_info["common_tokens"]
                size = cluster_info["size"]
                percentage = cluster_info["percentage"]
                
                # Build linguistic profile
                profile_str = build_token_cluster_profile(cluster_key, tokens, size, percentage, layer)
                cluster_profiles[cluster_key] = profile_str
                
                # Store for similarity checking
                all_clusters_data[cluster_key] = {
                    'tokens': tokens,
                    'size': size,
                    'percentage': percentage,
                    'layer': layer
                }
    
    # Generate labels using the ClusterAnalysis API
    logging.info(f"Generating labels for {len(cluster_profiles)} clusters...")
    labels = analyzer.label_clusters_sync(cluster_profiles, max_concurrency=5)
    
    # Store initial results
    for cluster_key, label in labels.items():
        layer = all_clusters_data[cluster_key]['layer']
        layer_key = f"layer_{layer}"
        
        results["labels"][layer_key][cluster_key] = {
            "label": label,
            "description": f"Cluster containing tokens like: {', '.join(repr(t) for t in all_clusters_data[cluster_key]['tokens'][:5])}",
            "size": all_clusters_data[cluster_key]['size'],
            "percentage": all_clusters_data[cluster_key]['percentage']
        }
    
    # Check for consistency issues
    logging.info("Checking for consistency across similar clusters...")
    consistency_report = []
    consistency_fixes = []
    
    for cluster_key in cluster_profiles.keys():
        similar_clusters = find_similar_clusters(
            all_clusters_data[cluster_key]['tokens'],
            all_clusters_data,
            cluster_key,
            threshold=0.6
        )
        
        for similar in similar_clusters:
            if similar['key'] in labels:
                label1 = labels[cluster_key]
                label2 = labels[similar['key']]
                
                if label1 != label2:
                    consistency_report.append({
                        'cluster1': cluster_key,
                        'label1': label1,
                        'cluster2': similar['key'],
                        'label2': label2,
                        'similarity': similar['similarity']
                    })
    
    # If significant consistency issues, run targeted re-labeling
    if len(consistency_report) > 10:
        logging.info(f"Found {len(consistency_report)} consistency issues. Running targeted re-labeling...")
        
        # Group highly similar clusters
        cluster_groups = defaultdict(set)
        for issue in consistency_report:
            if issue['similarity'] > 0.7:  # High similarity threshold
                cluster_groups[issue['cluster1']].add(issue['cluster2'])
                cluster_groups[issue['cluster2']].add(issue['cluster1'])
        
        # Re-label groups with consistency focus
        for primary_cluster, related_clusters in cluster_groups.items():
            if len(related_clusters) > 1:  # Multiple similar clusters
                group = {primary_cluster} | related_clusters
                
                # Build a combined profile emphasizing similarity
                combined_tokens = []
                for c in group:
                    combined_tokens.extend(all_clusters_data[c]['tokens'][:20])
                
                # Get unique tokens that appear in multiple clusters
                token_counts = defaultdict(int)
                for token in combined_tokens:
                    token_counts[token] += 1
                
                shared_tokens = [t for t, count in token_counts.items() if count >= 2][:30]
                
                # Create focused profile
                profile = f"This is a group of {len(group)} similar clusters across layers that share these core tokens:\n"
                profile += f"{', '.join(repr(t) for t in shared_tokens)}\n\n"
                profile += "The clusters appear in these layers: "
                profile += ', '.join(f"Layer {all_clusters_data[c]['layer']}" for c in sorted(group))
                profile += "\n\nProvide a single consistent label that captures what unifies these tokens across all layers."
                
                # Get single label for the group
                consistent_label = analyzer.generate_with_cache(
                    prompt=f"Analyze this group of similar neural network clusters and provide a single, concise semantic label (3-5 words):\n\n{profile}\n\nSemantic Label:",
                    temperature=0.2,
                    max_tokens=20
                ).text.strip().strip('"\'').replace("Label:", "").strip()
                
                # Apply to all clusters in group
                for cluster in group:
                    old_label = labels[cluster]
                    if old_label != consistent_label:
                        consistency_fixes.append({
                            'cluster': cluster,
                            'old': old_label,
                            'new': consistent_label
                        })
                        labels[cluster] = consistent_label
                        
                        # Update results
                        layer = all_clusters_data[cluster]['layer']
                        layer_key = f"layer_{layer}"
                        results["labels"][layer_key][cluster]["label"] = consistent_label
                        results["labels"][layer_key][cluster]["consistency_note"] = "Updated for cross-layer consistency"
    
    # Save results
    output_path = base_dir / "llm_labels_k10" / "cluster_labels_k10.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Saved labels to {output_path}")
    
    # Generate detailed report
    report = ["DIRECT LLM ANALYSIS REPORT", "=" * 60, ""]
    
    # Model info
    report.extend([
        f"Model: {analyzer.provider} - {analyzer.model}",
        f"Generated at: {results['metadata']['generated_at']}",
        f"Total clusters: {len(cluster_profiles)}",
        ""
    ])
    
    # Label distribution
    label_counts = defaultdict(int)
    for layer_data in results["labels"].values():
        for cluster_data in layer_data.values():
            label_counts[cluster_data["label"]] += 1
    
    report.append("Label Distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        report.append(f"  {label}: {count} clusters ({count/120*100:.1f}%)")
    
    report.extend([
        "",
        f"Total unique labels: {len(label_counts)}",
        f"Average clusters per label: {120/len(label_counts):.1f}",
        ""
    ])
    
    # Consistency analysis
    if consistency_fixes:
        report.extend([
            f"Consistency Fixes Applied: {len(consistency_fixes)}",
            "Examples:"
        ])
        for fix in consistency_fixes[:10]:
            report.append(f"  {fix['cluster']}: '{fix['old']}' → '{fix['new']}'")
        report.append("")
    
    # Final consistency check
    final_issues = 0
    for issue in consistency_report:
        if labels[issue['cluster1']] != labels[issue['cluster2']] and issue['similarity'] > 0.7:
            final_issues += 1
    
    report.extend([
        "Final Consistency Analysis:",
        f"  High-similarity clusters (>70%) with different labels: {final_issues}",
        f"  Consistency rate: {(1 - final_issues/len(consistency_report) if consistency_report else 1)*100:.1f}%",
        ""
    ])
    
    # Cache statistics
    cache_stats = analyzer.get_cache_stats()
    report.extend([
        "Cache Statistics:",
        f"  Cache enabled: {cache_stats.get('enabled', False)}",
        f"  Total items: {cache_stats.get('total_items', 0)}",
        f"  Hit rate: {cache_stats.get('hit_rate', 0):.1f}%" if 'hit_rate' in cache_stats else "  Hit rate: N/A",
        ""
    ])
    
    # Layer-by-layer summary
    report.append("Layer-by-Layer Label Summary:")
    for layer in range(12):
        layer_key = f"layer_{layer}"
        layer_labels = results["labels"][layer_key]
        
        report.append(f"\nLayer {layer}:")
        for cluster_idx in range(10):
            cluster_key = f"L{layer}_C{cluster_idx}"
            if cluster_key in layer_labels:
                label_info = layer_labels[cluster_key]
                report.append(f"  C{cluster_idx}: {label_info['label']} ({label_info['size']} tokens)")
    
    # Save report
    report_path = base_dir / "llm_labels_k10" / "direct_llm_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("\nDirect LLM analysis complete!")
    print(f"Generated {len(label_counts)} unique labels for {sum(label_counts.values())} clusters")
    print(f"Applied {len(consistency_fixes)} consistency fixes")
    print(f"Results saved to: {output_path}")
    print(f"Report saved to: {report_path}")
    
    # Close the analyzer to save cache
    analyzer.close()


if __name__ == "__main__":
    main()