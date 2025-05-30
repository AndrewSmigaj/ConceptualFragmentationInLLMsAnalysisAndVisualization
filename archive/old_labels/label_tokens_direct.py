#!/usr/bin/env python3
"""
Direct token labeling for GPT-2 tokens.
Generates a script that can be run to create labels for analysis.
"""

import json
from pathlib import Path
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def prepare_token_analysis(tokens_path: Path, output_dir: Path, sample_size: int = None):
    """Prepare tokens for analysis and generate labeling instructions."""
    
    # Load tokens
    logging.info(f"Loading tokens from {tokens_path}")
    with open(tokens_path, 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    
    if sample_size:
        tokens = tokens[:sample_size]
        logging.info(f"Using sample of {sample_size} tokens")
    
    # Group tokens by type for more efficient analysis
    token_groups = defaultdict(list)
    for token_data in tokens:
        token_type = token_data.get('token_type', 'unknown')
        is_subword = token_data.get('is_subword', False)
        
        if token_data.get('is_punctuation', False):
            group = 'punctuation'
        elif token_data.get('is_numeric', False):
            group = 'numeric'
        elif is_subword:
            group = 'subword'
        elif token_type == 'word':
            group = 'complete_word'
        elif token_type == 'word_with_space':
            group = 'word_with_space'
        else:
            group = 'other'
        
        token_groups[group].append(token_data)
    
    # Save grouped tokens for analysis
    output_dir.mkdir(exist_ok=True)
    
    for group_name, group_tokens in token_groups.items():
        output_path = output_dir / f"tokens_{group_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(group_tokens, f, indent=2)
        logging.info(f"Saved {len(group_tokens)} {group_name} tokens to {output_path}")
    
    # Create analysis instructions
    instructions = []
    instructions.append("# Token Analysis Instructions\n")
    instructions.append(f"Total tokens to analyze: {len(tokens)}\n")
    instructions.append("\n## Token Groups:\n")
    
    for group_name, group_tokens in sorted(token_groups.items()):
        instructions.append(f"\n### {group_name.replace('_', ' ').title()} ({len(group_tokens)} tokens)\n")
        
        # Show examples
        examples = group_tokens[:10]
        instructions.append("Examples:\n")
        for ex in examples:
            token_str = ex['token_str']
            token_id = ex['token_id']
            instructions.append(f"- ID {token_id}: '{token_str}'\n")
        
        # Analysis prompts for each group
        if group_name == 'punctuation':
            instructions.append("\nAnalysis focus: syntactic function, typical usage context\n")
        elif group_name == 'numeric':
            instructions.append("\nAnalysis focus: numeric type, typical usage (ordinal, cardinal, year, etc.)\n")
        elif group_name == 'subword':
            instructions.append("\nAnalysis focus: morphological function (prefix/suffix/root), grammatical role\n")
        elif group_name in ['complete_word', 'word_with_space']:
            instructions.append("\nAnalysis focus: POS tag, semantic category, conceptual hierarchy\n")
    
    # Save instructions
    instructions_path = output_dir / "analysis_instructions.md"
    with open(instructions_path, 'w') as f:
        f.writelines(instructions)
    
    logging.info(f"Analysis instructions saved to {instructions_path}")
    
    # Generate summary statistics
    print("\n=== Token Distribution Summary ===")
    total = len(tokens)
    for group_name, group_tokens in sorted(token_groups.items()):
        count = len(group_tokens)
        percentage = count / total * 100
        print(f"{group_name.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    return token_groups


def create_sample_labels():
    """Create example labels to show the expected format."""
    
    sample_labels = {
        # Punctuation example
        0: {
            "token": "!",
            "grammatical": {"pos": "PUNCT", "subtype": "exclamation"},
            "semantic": {"category": "punctuation", "function": "emphasis"},
            "morphological": {"role": "punctuation_mark"},
            "conceptual_hierarchy": [],
            "usage_context": "Sentence-ending exclamation mark for emphasis or surprise"
        },
        # Complete word example  
        262: {
            "token": "the",
            "grammatical": {"pos": "DET", "subtype": "definite_article"},
            "semantic": {"category": "function_word", "subcategory": "determiner"},
            "morphological": {"role": "root_word"},
            "conceptual_hierarchy": ["determiner", "function_word", "linguistic_unit"],
            "usage_context": "Definite article, most common word in English"
        },
        # Subword example
        278: {
            "token": "ing",
            "grammatical": {"pos": "SUFFIX", "subtype": "verbal_inflection"},
            "semantic": {"category": "morpheme", "subcategory": "inflectional_suffix"},
            "morphological": {"role": "suffix", "function": "progressive_aspect"},
            "conceptual_hierarchy": ["verbal_suffix", "inflectional_morpheme", "morphological_unit"],
            "usage_context": "Progressive/continuous aspect marker for verbs"
        }
    }
    
    return sample_labels


def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "token_labels_prepared"
    
    # Prepare tokens for analysis
    tokens_path = base_dir / "top_10k_tokens_full.json"
    if not tokens_path.exists():
        logging.error(f"Token file not found: {tokens_path}")
        return
    
    # Check if we should use a sample
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=None, 
                       help='Process only a sample of tokens')
    args = parser.parse_args()
    
    token_groups = prepare_token_analysis(tokens_path, output_dir, args.sample)
    
    # Create sample labels file
    sample_labels = create_sample_labels()
    sample_path = output_dir / "sample_labels.json"
    with open(sample_path, 'w') as f:
        json.dump(sample_labels, f, indent=2)
    
    print(f"\nPreparation complete!")
    print(f"Token groups saved to: {output_dir}")
    print(f"Sample labels format: {sample_path}")
    print("\nNext step: Analyze each token group and create comprehensive labels")


if __name__ == "__main__":
    main()