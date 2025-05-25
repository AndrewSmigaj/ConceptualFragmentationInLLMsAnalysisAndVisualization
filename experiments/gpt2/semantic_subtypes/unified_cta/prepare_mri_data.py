"""
Prepare data for Concept MRI visualization.

This script:
1. Loads the path analysis data with unique cluster labels
2. Adds semantic/grammatical categories from curated word lists
3. Enhances Sankey data with category information
4. Outputs comprehensive visualization data
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def load_semantic_categories():
    """Load the curated word lists with semantic categories."""
    curated_path = Path(__file__).parent.parent / "data" / "gpt2_semantic_subtypes_curated.json"
    
    with open(curated_path, 'r') as f:
        data = json.load(f)
    
    # Create word to category mapping
    word_to_category = {}
    category_colors = {
        'concrete_nouns': '#1f77b4',      # Blue
        'abstract_nouns': '#17becf',      # Light blue
        'physical_adjectives': '#ff7f0e',  # Orange
        'emotive_adjectives': '#d62728',   # Red
        'manner_adverbs': '#2ca02c',       # Green
        'degree_adverbs': '#bcbd22',       # Yellow-green
        'action_verbs': '#9467bd',         # Purple
        'stative_verbs': '#e377c2'         # Pink
    }
    
    # Simplified grammatical categories
    grammatical_mapping = {
        'concrete_nouns': 'noun',
        'abstract_nouns': 'noun',
        'physical_adjectives': 'adjective',
        'emotive_adjectives': 'adjective',
        'manner_adverbs': 'adverb',
        'degree_adverbs': 'adverb',
        'action_verbs': 'verb',
        'stative_verbs': 'verb'
    }
    
    for category, words in data['curated_words'].items():
        for word in words:
            word_to_category[word] = {
                'semantic_category': category,
                'grammatical_category': grammatical_mapping[category],
                'color': category_colors[category]
            }
    
    return word_to_category, category_colors


def enhance_path_data(results_dir):
    """Enhance path data with semantic categories."""
    # Load path analysis data
    llm_data_path = Path(results_dir) / "llm_analysis_data" / "all_paths_unique_labels.json"
    
    with open(llm_data_path, 'r') as f:
        path_data = json.load(f)
    
    # Load semantic categories
    word_to_category, category_colors = load_semantic_categories()
    
    # Enhance each path with category information
    for window in ['early', 'middle', 'late']:
        if window not in path_data['windowed_paths_unique']:
            continue
            
        window_data = path_data['windowed_paths_unique'][window]
        
        for path_info in window_data['all_paths']:
            # Analyze word categories in this path
            category_counts = defaultdict(int)
            grammatical_counts = defaultdict(int)
            
            # Check all words in the path
            all_words = path_info.get('all_words', [])
            if not all_words and path_info['frequency'] > 5:
                # Use example words if we don't have all words
                all_words = path_info['example_words']
            
            for word in all_words:
                if word in word_to_category:
                    cat_info = word_to_category[word]
                    category_counts[cat_info['semantic_category']] += 1
                    grammatical_counts[cat_info['grammatical_category']] += 1
                else:
                    grammatical_counts['unknown'] += 1
            
            # Determine dominant category
            if grammatical_counts:
                dominant_gram = max(grammatical_counts.items(), key=lambda x: x[1])[0]
            else:
                dominant_gram = 'unknown'
            
            # Add category info to path
            path_info['category_analysis'] = {
                'semantic_breakdown': dict(category_counts),
                'grammatical_breakdown': dict(grammatical_counts),
                'dominant_grammatical': dominant_gram,
                'dominant_color': {
                    'noun': '#1f77b4',
                    'adjective': '#ff7f0e',
                    'adverb': '#2ca02c',
                    'verb': '#9467bd',
                    'unknown': '#7f7f7f'
                }.get(dominant_gram, '#7f7f7f')
            }
    
    return path_data, word_to_category


def enhance_sankey_data(results_dir, word_to_category):
    """Enhance existing Sankey data with category colors."""
    sankey_dir = Path(results_dir) / "windowed_analysis"
    
    enhanced_sankey = {}
    
    for window in ['early', 'middle', 'late']:
        sankey_path = sankey_dir / f"sankey_{window}.json"
        
        if not sankey_path.exists():
            print(f"Warning: {sankey_path} not found")
            continue
            
        with open(sankey_path, 'r') as f:
            sankey_data = json.load(f)
        
        # The existing Sankey data already has the structure we need
        # Just add metadata about the window
        sankey_data['metadata'] = {
            'window': window,
            'title': f"{window.capitalize()} Window (Layers {sankey_data['layers'][0]}-{sankey_data['layers'][-1]})"
        }
        
        enhanced_sankey[window] = sankey_data
    
    return enhanced_sankey


def create_cluster_contents(results_dir, word_to_category):
    """Create a mapping of cluster contents for inspection."""
    # This would require the full trajectory data
    # For now, we'll extract from path information
    
    cluster_contents = defaultdict(list)
    
    # Load the enhanced path data
    llm_data_path = Path(results_dir) / "llm_analysis_data" / "all_paths_unique_labels.json"
    with open(llm_data_path, 'r') as f:
        path_data = json.load(f)
    
    # Extract cluster membership from paths
    for window in ['early', 'middle', 'late']:
        window_data = path_data['windowed_paths_unique'][window]
        
        for path_info in window_data['all_paths']:
            # Get words in this path
            words = path_info.get('all_words', path_info['example_words'][:10])
            
            # Add words to each cluster in the path
            for cluster_id in path_info['cluster_sequence']:
                for word in words:
                    word_entry = {
                        'word': word,
                        'category': word_to_category.get(word, {}).get('semantic_category', 'unknown'),
                        'grammatical': word_to_category.get(word, {}).get('grammatical_category', 'unknown')
                    }
                    # Check if word already in cluster
                    word_exists = any(w['word'] == word for w in cluster_contents[cluster_id])
                    if not word_exists:
                        cluster_contents[cluster_id].append(word_entry)
    
    # Convert to regular dict
    return {cluster: list(words) for cluster, words in cluster_contents.items()}


def main(results_dir=None):
    """Main function to prepare all MRI visualization data."""
    
    if results_dir is None:
        # Use the most recent results
        results_dir = Path("results/unified_cta_config/unified_cta_20250524_073316")
    else:
        results_dir = Path(results_dir)
    
    print("Loading semantic categories...")
    word_to_category, category_colors = load_semantic_categories()
    
    print("Enhancing path data...")
    enhanced_paths, word_to_category = enhance_path_data(results_dir)
    
    print("Enhancing Sankey data...")
    enhanced_sankey = enhance_sankey_data(results_dir, word_to_category)
    
    print("Creating cluster contents...")
    cluster_contents = create_cluster_contents(results_dir, word_to_category)
    
    # Compile all data for visualization
    mri_data = {
        'metadata': {
            'n_words': enhanced_paths['experiment_info']['n_words'],
            'n_layers': enhanced_paths['experiment_info']['n_layers'],
            'semantic_categories': list(category_colors.keys()),
            'category_colors': category_colors
        },
        'paths': enhanced_paths['windowed_paths_unique'],
        'sankey': enhanced_sankey,
        'cluster_contents': cluster_contents,
        'word_categories': word_to_category,
        'transitions': enhanced_paths['transition_analysis'],
        'key_patterns': enhanced_paths['key_patterns']
    }
    
    # Save the compiled data
    output_path = results_dir / "concept_mri_data.json"
    with open(output_path, 'w') as f:
        json.dump(mri_data, f, indent=2)
    
    print(f"\nMRI data saved to: {output_path}")
    
    # Print summary statistics
    print("\n=== Data Summary ===")
    print(f"Total words: {mri_data['metadata']['n_words']}")
    print(f"Semantic categories: {len(mri_data['metadata']['semantic_categories'])}")
    
    for window in ['early', 'middle', 'late']:
        if window in mri_data['paths']:
            n_paths = len(mri_data['paths'][window]['all_paths'])
            print(f"{window.capitalize()} window: {n_paths} paths")
    
    return mri_data


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = None
    
    main(results_dir)