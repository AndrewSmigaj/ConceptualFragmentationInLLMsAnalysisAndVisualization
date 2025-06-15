"""
Run GPT-2 semantic subtypes experiment with expanded word list.

This script:
1. Validates the expanded word list against GPT-2 tokenizer
2. Extracts activations for all validated words
3. Runs the unified CTA analysis
4. Generates results and visualizations
"""

import os
import sys
import json
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from gpt2_semantic_subtypes_wordlists_expanded import ALL_WORD_LISTS
from gpt2_semantic_subtypes_curator import GPT2SemanticSubtypesCurator
from gpt2_semantic_subtypes_experiment import GPT2SemanticSubtypesExperiment

def validate_expanded_words():
    """Validate expanded word list and create curated dataset."""
    print("=== Validating Expanded Word List ===")
    
    # Initialize curator
    curator = GPT2SemanticSubtypesCurator(ALL_WORD_LISTS)
    
    # Validate all words
    curator.validate_all_words()
    
    # Save curated dataset
    output_path = "data/gpt2_semantic_subtypes_curated_expanded.json"
    curator.save_curated_dataset(output_path)
    
    # Print statistics
    stats = curator.get_curation_statistics()
    print(f"\nValidation complete:")
    print(f"Total validated words: {stats['overall_statistics']['total_curated_words']}")
    print(f"Overall efficiency: {stats['overall_statistics']['overall_efficiency']:.1%}")
    
    # Print grammatical breakdown
    words = stats['curated_words']
    nouns = len(words.get('concrete_nouns', [])) + len(words.get('abstract_nouns', []))
    adj = len(words.get('physical_adjectives', [])) + len(words.get('emotive_adjectives', []))
    adv = len(words.get('manner_adverbs', [])) + len(words.get('degree_adverbs', []))
    verbs = len(words.get('action_verbs', [])) + len(words.get('stative_verbs', []))
    total = nouns + adj + adv + verbs
    
    print(f"\nGrammatical breakdown of validated words:")
    print(f"  Nouns: {nouns} ({nouns/total*100:.1f}%)")
    print(f"  Adjectives: {adj} ({adj/total*100:.1f}%)")
    print(f"  Adverbs: {adv} ({adv/total*100:.1f}%)")
    print(f"  Verbs: {verbs} ({verbs/total*100:.1f}%)")
    
    return output_path

def run_expanded_experiment(curated_data_path):
    """Run the experiment with expanded word list."""
    print("\n=== Running Expanded GPT-2 Experiment ===")
    
    # Configure experiment
    config = {
        "model_name": "gpt2",
        "batch_size": 32,
        "device": "cuda",
        "clustering_method": "unified_cta",
        "gap_statistic": True,
        "max_k": 10,
        "n_refs": 10,
        "output_dir": "results/expanded",
        "cache_dir": "data/cache/expanded"
    }
    
    # Initialize experiment
    experiment = GPT2SemanticSubtypesExperiment(
        curated_data_path=curated_data_path,
        config=config
    )
    
    # Run full pipeline
    print("\n1. Extracting activations...")
    experiment.extract_activations()
    
    print("\n2. Running clustering analysis...")
    experiment.run_clustering()
    
    print("\n3. Analyzing trajectories...")
    experiment.analyze_trajectories()
    
    print("\n4. Generating visualizations...")
    experiment.generate_visualizations()
    
    print("\n5. Running LLM analysis...")
    experiment.run_llm_analysis()
    
    # Save results
    results_path = Path(config["output_dir"]) / "experiment_results.json"
    experiment.save_results(results_path)
    
    print(f"\nExperiment complete! Results saved to: {results_path}")
    
    # Print key findings
    print("\n=== Key Findings ===")
    results = experiment.get_results_summary()
    print(f"Entity superhighway convergence: {results['entity_convergence']:.1%}")
    print(f"Path reduction: {results['path_reduction']}")
    print(f"Phase transition window: {results['phase_transition']}")
    
    return results

def compare_with_original():
    """Compare expanded results with original experiment."""
    print("\n=== Comparing with Original Results ===")
    
    # Load original results
    original_path = "results/experiment_results.json"
    if os.path.exists(original_path):
        with open(original_path, 'r') as f:
            original = json.load(f)
        
        # Load expanded results
        expanded_path = "results/expanded/experiment_results.json"
        with open(expanded_path, 'r') as f:
            expanded = json.load(f)
        
        print("\nComparison:")
        print(f"Original dataset: {original['total_words']} words")
        print(f"Expanded dataset: {expanded['total_words']} words")
        print(f"\nOriginal convergence: {original['entity_convergence']:.1%}")
        print(f"Expanded convergence: {expanded['entity_convergence']:.1%}")
        print(f"\nOriginal verb representation: {original['verb_percentage']:.1%}")
        print(f"Expanded verb representation: {expanded['verb_percentage']:.1%}")
    else:
        print("Original results not found for comparison")

if __name__ == "__main__":
    # Step 1: Validate expanded words
    curated_path = validate_expanded_words()
    
    # Step 2: Run experiment
    results = run_expanded_experiment(curated_path)
    
    # Step 3: Compare results
    compare_with_original()
    
    print("\n=== Expanded Experiment Complete ===")