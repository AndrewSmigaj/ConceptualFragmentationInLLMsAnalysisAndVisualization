#!/usr/bin/env python3
"""
Create data-driven semantic labels for k=10 clusters.
Analyzes actual token contents to create meaningful, consistent labels.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import re
import os

# Comprehensive linguistic categories
DETERMINERS = {' the', ' a', ' an', ' this', ' that', ' these', ' those', ' my', ' your', ' his', ' her', ' its', ' our', ' their', ' all', ' some', ' any', ' each', ' every', ' no'}
PREPOSITIONS = {' in', ' on', ' at', ' to', ' for', ' of', ' with', ' by', ' from', ' about', ' into', ' through', ' over', ' under', ' between', ' among', ' during', ' before', ' after', ' above', ' below', ' up', ' down', ' out'}
CONJUNCTIONS = {' and', ' or', ' but', ' nor', ' yet', ' so', ' for', ' as', ' if', ' when', ' while', ' because', ' since', ' although', ' though', ' unless', ' until', ' whether'}
PRONOUNS = {' he', ' she', ' it', ' they', ' we', ' you', ' I', ' me', ' him', ' her', ' them', ' us', ' myself', ' yourself', ' himself', ' herself', ' itself', ' ourselves', ' themselves'}
AUXILIARIES = {' is', ' are', ' was', ' were', ' have', ' has', ' had', ' do', ' does', ' did', ' will', ' would', ' could', ' should', ' may', ' might', ' must', ' shall', ' can', ' been', ' be', ' being'}
MODALS = {' can', ' could', ' may', ' might', ' must', ' shall', ' should', ' will', ' would', ' ought'}

# Common verbs and their forms
COMMON_VERBS = {' say', ' get', ' make', ' go', ' know', ' take', ' see', ' come', ' think', ' look', ' want', ' give', ' use', ' find', ' tell', ' ask', ' work', ' call', ' try', ' need', ' feel', ' become', ' leave', ' put', ' mean', ' keep', ' let', ' begin', ' seem', ' help', ' show'}
MOTION_VERBS = {' go', ' come', ' walk', ' run', ' move', ' travel', ' arrive', ' leave', ' enter', ' exit', ' approach', ' depart', ' return'}
COMMUNICATION_VERBS = {' say', ' tell', ' ask', ' speak', ' talk', ' write', ' read', ' call', ' answer', ' reply', ' explain', ' describe', ' mention', ' announce'}
MENTAL_VERBS = {' think', ' know', ' believe', ' understand', ' remember', ' forget', ' learn', ' realize', ' recognize', ' suppose', ' imagine', ' wonder', ' doubt'}

# Semantic categories
TIME_WORDS = {' time', ' day', ' year', ' week', ' month', ' hour', ' minute', ' second', ' today', ' tomorrow', ' yesterday', ' now', ' then', ' when', ' soon', ' later', ' early', ' late', ' morning', ' afternoon', ' evening', ' night', ' always', ' never', ' sometimes', ' often'}
SPATIAL_WORDS = {' place', ' here', ' there', ' where', ' everywhere', ' nowhere', ' somewhere', ' home', ' house', ' room', ' street', ' city', ' country', ' world', ' area', ' region', ' location', ' building', ' office', ' school', ' up', ' down', ' left', ' right', ' front', ' back', ' side'}
QUANTITY_WORDS = {' all', ' some', ' many', ' few', ' much', ' little', ' more', ' less', ' most', ' least', ' several', ' various', ' enough', ' plenty', ' number', ' amount'}

def load_full_cluster_data(base_dir: str) -> Dict[str, List[str]]:
    """Load full token lists for each cluster."""
    # Try to load from clustering results
    cluster_tokens = {}
    
    # Check different possible locations
    paths_to_try = [
        os.path.join(base_dir, "clustering_results_k10", "cluster_contents_k10.json"),
        os.path.join(base_dir, "llm_labels_k10", "cluster_tokens_k10.json"),
        os.path.join(base_dir, "k10_analysis_results", "cluster_tokens_k10.json")
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'clusters' in data:
                    for cluster_id, cluster_data in data['clusters'].items():
                        if 'tokens' in cluster_data:
                            cluster_tokens[cluster_id] = cluster_data['tokens']
                return cluster_tokens
    
    return cluster_tokens

def analyze_token_list(tokens: List[str]) -> Dict[str, float]:
    """Analyze a full list of tokens for linguistic patterns."""
    total = len(tokens) if tokens else 1
    
    analysis = {
        # Function word categories
        'determiners': sum(1 for t in tokens if t in DETERMINERS) / total,
        'prepositions': sum(1 for t in tokens if t in PREPOSITIONS) / total,
        'conjunctions': sum(1 for t in tokens if t in CONJUNCTIONS) / total,
        'pronouns': sum(1 for t in tokens if t in PRONOUNS) / total,
        'auxiliaries': sum(1 for t in tokens if t in AUXILIARIES) / total,
        'modals': sum(1 for t in tokens if t in MODALS) / total,
        
        # Verb categories
        'common_verbs': sum(1 for t in tokens if t in COMMON_VERBS) / total,
        'motion_verbs': sum(1 for t in tokens if t in MOTION_VERBS) / total,
        'communication_verbs': sum(1 for t in tokens if t in COMMUNICATION_VERBS) / total,
        'mental_verbs': sum(1 for t in tokens if t in MENTAL_VERBS) / total,
        
        # Semantic categories
        'time_words': sum(1 for t in tokens if t in TIME_WORDS) / total,
        'spatial_words': sum(1 for t in tokens if t in SPATIAL_WORDS) / total,
        'quantity_words': sum(1 for t in tokens if t in QUANTITY_WORDS) / total,
        
        # Morphological patterns
        'punctuation': sum(1 for t in tokens if re.match(r'^[^\w\s]+$', t.strip())) / total,
        'numeric': sum(1 for t in tokens if re.match(r'^\s*\d+\s*$', t)) / total,
        'capitalized': sum(1 for t in tokens if t.strip() and t.strip()[0].isupper()) / total,
        'contractions': sum(1 for t in tokens if "'" in t and ("'t" in t or "'ll" in t or "'ve" in t or "'re" in t or "'d" in t or "'m" in t)) / total,
        'possessives': sum(1 for t in tokens if t.endswith("'s")) / total,
        
        # Suffix patterns
        'ing_forms': sum(1 for t in tokens if t.strip().endswith('ing')) / total,
        'ed_forms': sum(1 for t in tokens if t.strip().endswith('ed')) / total,
        'ly_forms': sum(1 for t in tokens if t.strip().endswith('ly')) / total,
        'plural_s': sum(1 for t in tokens if t.strip().endswith('s') and not t.endswith("'s")) / total,
        'er_forms': sum(1 for t in tokens if t.strip().endswith('er') and len(t.strip()) > 3) / total,
        
        # Special patterns
        'space_prefix': sum(1 for t in tokens if t.startswith(' ')) / total,
        'no_space': sum(1 for t in tokens if not t.startswith(' ')) / total,
        'short_tokens': sum(1 for t in tokens if len(t.strip()) <= 3) / total,
        'long_tokens': sum(1 for t in tokens if len(t.strip()) >= 8) / total
    }
    
    # Calculate composite scores
    analysis['function_words_total'] = (analysis['determiners'] + analysis['prepositions'] + 
                                       analysis['conjunctions'] + analysis['auxiliaries'])
    analysis['content_words_total'] = 1.0 - analysis['function_words_total'] - analysis['punctuation']
    analysis['verb_forms_total'] = (analysis['common_verbs'] + analysis['motion_verbs'] + 
                                   analysis['communication_verbs'] + analysis['mental_verbs'])
    analysis['morphological_total'] = (analysis['ing_forms'] + analysis['ed_forms'] + 
                                      analysis['ly_forms'] + analysis['plural_s'])
    
    return analysis

def create_data_driven_label(analysis: Dict[str, float], tokens: List[str]) -> str:
    """Create a label based on data-driven analysis."""
    
    # Priority 1: Punctuation
    if analysis['punctuation'] > 0.7:
        return "Punctuation Marks"
    
    # Priority 2: Numeric
    if analysis['numeric'] > 0.5:
        return "Numeric Tokens"
    
    # Priority 3: Strong function word categories
    if analysis['function_words_total'] > 0.6:
        # Determine dominant type
        max_func = max(
            ('determiners', analysis['determiners']),
            ('prepositions', analysis['prepositions']),
            ('conjunctions', analysis['conjunctions']),
            ('auxiliaries', analysis['auxiliaries']),
            ('pronouns', analysis['pronouns'])
        )
        
        if max_func[0] == 'determiners' and max_func[1] > 0.3:
            return "Determiners & Articles"
        elif max_func[0] == 'prepositions' and max_func[1] > 0.25:
            return "Prepositions & Relational Terms"
        elif max_func[0] == 'conjunctions' and max_func[1] > 0.2:
            return "Conjunctions & Connectives"
        elif max_func[0] == 'auxiliaries' and max_func[1] > 0.2:
            if analysis['modals'] > 0.1:
                return "Modal & Auxiliary Verbs"
            else:
                return "Auxiliary & Helping Verbs"
        elif max_func[0] == 'pronouns' and max_func[1] > 0.25:
            return "Personal Pronouns"
        else:
            return "Core Function Words"
    
    # Priority 4: Verb categories
    if analysis['verb_forms_total'] > 0.3:
        max_verb = max(
            ('motion', analysis['motion_verbs']),
            ('communication', analysis['communication_verbs']),
            ('mental', analysis['mental_verbs']),
            ('common', analysis['common_verbs'])
        )
        
        if max_verb[0] == 'motion' and max_verb[1] > 0.1:
            return "Motion & Movement Verbs"
        elif max_verb[0] == 'communication' and max_verb[1] > 0.1:
            return "Communication Verbs"
        elif max_verb[0] == 'mental' and max_verb[1] > 0.1:
            return "Mental & Cognitive Verbs"
        else:
            return "Common Action Verbs"
    
    # Priority 5: Morphological patterns
    if analysis['morphological_total'] > 0.4:
        max_morph = max(
            ('ing', analysis['ing_forms']),
            ('ed', analysis['ed_forms']),
            ('ly', analysis['ly_forms']),
            ('plural', analysis['plural_s'])
        )
        
        if max_morph[0] == 'ing' and max_morph[1] > 0.25:
            return "Progressive Forms (-ing)"
        elif max_morph[0] == 'ed' and max_morph[1] > 0.25:
            return "Past Tense Forms (-ed)"
        elif max_morph[0] == 'ly' and max_morph[1] > 0.2:
            return "Adverbial Forms (-ly)"
        elif max_morph[0] == 'plural' and max_morph[1] > 0.3:
            return "Plural Nouns"
        else:
            return "Inflected Forms"
    
    # Priority 6: Semantic categories
    if analysis['time_words'] > 0.15:
        return "Temporal Expressions"
    elif analysis['spatial_words'] > 0.15:
        return "Spatial & Location Terms"
    elif analysis['quantity_words'] > 0.1:
        return "Quantity & Amount Terms"
    
    # Priority 7: Special patterns
    if analysis['contractions'] > 0.15:
        return "Contracted Forms"
    elif analysis['possessives'] > 0.1:
        return "Possessive Forms"
    elif analysis['capitalized'] > 0.4:
        # Sample tokens to determine type
        cap_tokens = [t for t in tokens[:50] if t.strip() and t.strip()[0].isupper()]
        if any(t in [' Mr', ' Mrs', ' Dr', ' President'] for t in cap_tokens):
            return "Titles & Proper Names"
        else:
            return "Capitalized Content Words"
    
    # Priority 8: Token length patterns
    if analysis['short_tokens'] > 0.5 and analysis['no_space'] > 0.4:
        return "Subword Units"
    elif analysis['long_tokens'] > 0.3:
        return "Complex/Long Words"
    
    # Default based on content vs function
    if analysis['content_words_total'] > 0.7:
        # Try to be more specific by sampling
        sample = tokens[:30]
        
        # Check for common noun patterns
        if sum(1 for t in sample if t.endswith('tion') or t.endswith('ment') or t.endswith('ness')) > 5:
            return "Abstract Nouns"
        elif sum(1 for t in sample if t in [' man', ' woman', ' person', ' people', ' child']) > 3:
            return "Human/Person Terms"
        elif sum(1 for t in sample if any(part in t for part in ['body', 'hand', 'head', 'eye', 'face'])) > 3:
            return "Body & Physical Terms"
        else:
            return "General Content Words"
    else:
        return "Mixed Lexical Items"

def calculate_cluster_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Calculate similarity between clusters based on shared tokens."""
    if not tokens1 or not tokens2:
        return 0.0
    
    set1, set2 = set(tokens1), set(tokens2)
    intersection = len(set1 & set2)
    smaller_set = min(len(set1), len(set2))
    
    return intersection / smaller_set if smaller_set > 0 else 0.0

def group_similar_clusters(cluster_tokens: Dict[str, List[str]], threshold: float = 0.5) -> List[List[str]]:
    """Group clusters with high token overlap."""
    cluster_ids = list(cluster_tokens.keys())
    groups = []
    used = set()
    
    # Sort by layer to process in order
    cluster_ids_sorted = sorted(cluster_ids, key=lambda x: (int(x.split('_')[0][1:]), int(x.split('_')[1][1:])))
    
    for id1 in cluster_ids_sorted:
        if id1 in used:
            continue
        
        group = [id1]
        used.add(id1)
        
        # Find similar clusters
        for id2 in cluster_ids_sorted:
            if id2 not in used and id2 != id1:
                similarity = calculate_cluster_similarity(
                    cluster_tokens[id1], 
                    cluster_tokens[id2]
                )
                if similarity >= threshold:
                    group.append(id2)
                    used.add(id2)
        
        groups.append(group)
    
    return groups

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load cluster data
    print("Loading cluster data...")
    data_path = os.path.join(base_dir, "llm_labels_k10", "llm_labeling_data.json")
    with open(data_path, 'r') as f:
        cluster_metadata = json.load(f)
    
    # Try to load full token lists
    print("Loading full token lists...")
    cluster_tokens = load_full_cluster_data(base_dir)
    
    # If we don't have full token lists, use common_tokens
    if not cluster_tokens:
        print("Using common_tokens as fallback...")
        cluster_tokens = {}
        for cluster_id, cluster_data in cluster_metadata['clusters'].items():
            cluster_tokens[cluster_id] = cluster_data.get('common_tokens', [])
    
    # Group similar clusters
    print("Grouping similar clusters...")
    groups = group_similar_clusters(cluster_tokens, threshold=0.5)
    print(f"Found {len(groups)} cluster groups")
    
    # Analyze and label each group
    print("Creating data-driven labels...")
    labels = {}
    label_counts = Counter()
    
    for group in groups:
        # Combine tokens from all clusters in group
        combined_tokens = []
        for cluster_id in group:
            combined_tokens.extend(cluster_tokens.get(cluster_id, []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in combined_tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        
        # Analyze combined tokens
        analysis = analyze_token_list(unique_tokens)
        
        # Create label
        label = create_data_driven_label(analysis, unique_tokens)
        
        # Make unique if needed
        if label_counts[label] > 0:
            # Add distinguishing info
            layers = [int(cid.split('_')[0][1:]) for cid in group]
            min_layer, max_layer = min(layers), max(layers)
            if min_layer != max_layer:
                label = f"{label} (L{min_layer}-{max_layer})"
            else:
                label = f"{label} (L{min_layer})"
        
        label_counts[label] += 1
        
        # Assign to all clusters in group
        for cluster_id in group:
            labels[cluster_id] = label
    
    # Format output
    formatted_labels = {}
    for cluster_id, label in labels.items():
        layer = int(cluster_id.split('_')[0][1:])
        cluster_num = int(cluster_id.split('_')[1][1:])
        formatted_labels[cluster_id] = {
            "label": label,
            "layer": layer,
            "cluster": cluster_num,
            "full_label": f"L{layer}: {label}"
        }
    
    # Save results
    output_path = os.path.join(base_dir, "llm_labels_k10", "cluster_labels_k10_datadriven.json")
    with open(output_path, 'w') as f:
        json.dump(formatted_labels, f, indent=2)
    
    print(f"\nLabels saved to {output_path}")
    
    # Print label distribution
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count} groups")
    
    # Create analysis report
    report_path = os.path.join(base_dir, "llm_labels_k10", "datadriven_analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write("Data-Driven Label Analysis Report\n")
        f.write("=================================\n\n")
        
        # Group clusters by label
        label_to_clusters = defaultdict(list)
        for cluster_id, label_data in formatted_labels.items():
            label_to_clusters[label_data['label']].append(cluster_id)
        
        for label, cluster_ids in sorted(label_to_clusters.items()):
            f.write(f"\nLabel: {label}\n")
            f.write(f"Clusters ({len(cluster_ids)}): {', '.join(sorted(cluster_ids))}\n")
            
            # Show sample tokens
            first_cluster = cluster_ids[0]
            sample_tokens = cluster_tokens.get(first_cluster, [])[:15]
            f.write(f"Sample tokens: {sample_tokens}\n")
            
            # Show cluster sizes
            sizes = []
            for cid in cluster_ids:
                if cid in cluster_metadata['clusters']:
                    sizes.append(cluster_metadata['clusters'][cid]['size'])
            f.write(f"Cluster sizes: {sizes}\n")
    
    print(f"Analysis report saved to {report_path}")

if __name__ == "__main__":
    main()