#!/usr/bin/env python3
"""
LLM-based token labeling for GPT-2 tokens.
Uses Claude to provide rich semantic, grammatical, and contextual labels
for tokens, including subwords and fragments that WordNet can't handle.
"""

import json
from pathlib import Path
import logging
from tqdm import tqdm
from collections import defaultdict
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from concept_fragmentation.llm.factory import LLMClientFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LLMTokenLabeler:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / "token_labels_llm"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize LLM client
        self.llm = LLMClientFactory.get_client('anthropic')
        
        # Batch processing parameters
        self.batch_size = 50  # Process 50 tokens at a time
        self.max_retries = 3
        
    def label_tokens(self, tokens_path: Path):
        """Label tokens using LLM analysis."""
        # Load tokens
        logging.info(f"Loading tokens from {tokens_path}")
        with open(tokens_path, 'r', encoding='utf-8') as f:
            tokens = json.load(f)
        
        logging.info(f"Processing {len(tokens)} tokens with LLM labeling...")
        
        labeled_tokens = {}
        
        # Process in batches
        for i in tqdm(range(0, len(tokens), self.batch_size), desc="Processing batches"):
            batch = tokens[i:i + self.batch_size]
            batch_labels = self._process_batch(batch)
            labeled_tokens.update(batch_labels)
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        # Save results
        output_path = self.results_dir / "llm_token_labels.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(labeled_tokens, f, indent=2)
        
        logging.info(f"Saved labeled tokens to {output_path}")
        
        # Generate and save statistics
        self._generate_statistics(labeled_tokens)
        
        return labeled_tokens
    
    def _process_batch(self, batch):
        """Process a batch of tokens with LLM."""
        # Prepare token list for LLM
        token_list = []
        for token_data in batch:
            token_list.append({
                'id': token_data['token_id'],
                'token': token_data['token_str'],
                'type': token_data['token_type'],
                'is_subword': token_data.get('is_subword', False),
                'has_space': token_data.get('has_leading_space', False)
            })
        
        prompt = f"""Analyze these GPT-2 tokens and provide detailed linguistic labels. For each token, determine:

1. **Grammatical category** (POS tag): NOUN, VERB, ADJ, ADV, PREP, CONJ, DET, PRON, PUNCT, NUM, PARTICLE, or FRAGMENT
2. **Semantic category**: concrete_object, abstract_concept, action, property, relation, function_word, subword_prefix, subword_suffix, subword_middle, punctuation, number, etc.
3. **Morphological role**: root_word, prefix, suffix, inflection, derivation, compound_part, or fragment
4. **Conceptual hierarchy**: For meaningful words, provide 2-3 levels of conceptual categories from specific to general (e.g., "dog" -> animal -> living_thing -> entity)

Tokens to analyze:
{json.dumps(token_list, indent=2)}

Provide output as a JSON object where each key is the token ID and value contains:
{{
  "grammatical": {{"pos": "...", "subtype": "..."}},
  "semantic": {{"category": "...", "subcategory": "..."}},
  "morphological": {{"role": "...", "function": "..."}},
  "conceptual_hierarchy": ["specific", "general", "most_general"],
  "usage_context": "brief description of typical usage"
}}

Important: 
- For subwords like "ing", "tion", analyze their grammatical function (e.g., progressive marker, nominalization)
- For fragments, identify if they're common prefixes/suffixes or partial words
- Be consistent in categorization across similar tokens
"""

        for attempt in range(self.max_retries):
            try:
                response = self.llm.complete(prompt, max_tokens=2000)
                
                # Parse response
                result_text = response.content
                
                # Extract JSON from response
                if '```json' in result_text:
                    json_start = result_text.find('```json') + 7
                    json_end = result_text.find('```', json_start)
                    result_text = result_text[json_start:json_end]
                elif '```' in result_text:
                    json_start = result_text.find('```') + 3
                    json_end = result_text.find('```', json_start)
                    result_text = result_text[json_start:json_end]
                
                batch_results = json.loads(result_text.strip())
                
                # Convert string IDs back to integers
                final_results = {}
                for token_id_str, labels in batch_results.items():
                    final_results[int(token_id_str)] = labels
                
                return final_results
                
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    # Return basic labels on final failure
                    return self._get_fallback_labels(batch)
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return {}
    
    def _get_fallback_labels(self, batch):
        """Provide basic fallback labels if LLM fails."""
        results = {}
        for token_data in batch:
            token_id = token_data['token_id']
            token_str = token_data['token_str'].strip()
            
            # Basic categorization
            if token_data.get('is_punctuation', False):
                pos = 'PUNCT'
                semantic = 'punctuation'
            elif token_data.get('is_numeric', False):
                pos = 'NUM'
                semantic = 'number'
            elif token_data.get('is_subword', False):
                pos = 'FRAGMENT'
                semantic = 'subword_fragment'
            else:
                pos = 'UNKNOWN'
                semantic = 'unknown'
            
            results[token_id] = {
                'grammatical': {'pos': pos},
                'semantic': {'category': semantic},
                'morphological': {'role': 'unknown'},
                'conceptual_hierarchy': [],
                'usage_context': 'No analysis available'
            }
        
        return results
    
    def _generate_statistics(self, labeled_tokens):
        """Generate and save labeling statistics."""
        stats = defaultdict(lambda: defaultdict(int))
        
        for token_id, labels in labeled_tokens.items():
            # Count POS tags
            pos = labels.get('grammatical', {}).get('pos', 'UNKNOWN')
            stats['pos_distribution'][pos] += 1
            
            # Count semantic categories
            semantic = labels.get('semantic', {}).get('category', 'unknown')
            stats['semantic_categories'][semantic] += 1
            
            # Count morphological roles
            morph = labels.get('morphological', {}).get('role', 'unknown')
            stats['morphological_roles'][morph] += 1
            
            # Count tokens with hierarchy
            if labels.get('conceptual_hierarchy'):
                stats['hierarchy_coverage']['has_hierarchy'] += 1
            else:
                stats['hierarchy_coverage']['no_hierarchy'] += 1
        
        # Convert to regular dict and calculate percentages
        total_tokens = len(labeled_tokens)
        final_stats = {
            'total_tokens': total_tokens,
            'pos_distribution': {k: {'count': v, 'percentage': v/total_tokens*100} 
                               for k, v in sorted(stats['pos_distribution'].items(), 
                                                key=lambda x: x[1], reverse=True)},
            'semantic_categories': {k: {'count': v, 'percentage': v/total_tokens*100} 
                                  for k, v in sorted(stats['semantic_categories'].items(), 
                                                   key=lambda x: x[1], reverse=True)[:20]},
            'morphological_roles': {k: {'count': v, 'percentage': v/total_tokens*100} 
                                  for k, v in sorted(stats['morphological_roles'].items(), 
                                                   key=lambda x: x[1], reverse=True)},
            'hierarchy_coverage': dict(stats['hierarchy_coverage'])
        }
        
        # Save statistics
        stats_path = self.results_dir / "llm_labeling_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        # Print summary
        print("\n=== LLM Token Labeling Statistics ===")
        print(f"Total tokens labeled: {total_tokens}")
        
        print("\nTop POS tags:")
        for pos, data in list(final_stats['pos_distribution'].items())[:10]:
            print(f"  {pos}: {data['count']} ({data['percentage']:.1f}%)")
        
        print("\nTop semantic categories:")
        for cat, data in list(final_stats['semantic_categories'].items())[:10]:
            print(f"  {cat}: {data['count']} ({data['percentage']:.1f}%)")
        
        print("\nMorphological roles:")
        for role, data in final_stats['morphological_roles'].items():
            print(f"  {role}: {data['count']} ({data['percentage']:.1f}%)")
        
        hierarchy_data = final_stats['hierarchy_coverage']
        if 'has_hierarchy' in hierarchy_data:
            hierarchy_pct = hierarchy_data['has_hierarchy'] / total_tokens * 100
            print(f"\nTokens with conceptual hierarchy: {hierarchy_data['has_hierarchy']} ({hierarchy_pct:.1f}%)")


def main():
    base_dir = Path(__file__).parent
    labeler = LLMTokenLabeler(base_dir)
    
    # Label the top 10k tokens
    tokens_path = base_dir / "top_10k_tokens_full.json"
    if not tokens_path.exists():
        logging.error(f"Token file not found: {tokens_path}")
        return
    
    # Check if we should use a smaller sample for testing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=None, 
                       help='Process only a sample of tokens for testing')
    args = parser.parse_args()
    
    if args.sample:
        # Load and sample tokens
        with open(tokens_path, 'r', encoding='utf-8') as f:
            all_tokens = json.load(f)
        
        # Take a diverse sample
        sample_tokens = all_tokens[:args.sample]
        
        # Save temporary sample file
        sample_path = base_dir / f"sample_{args.sample}_tokens.json"
        with open(sample_path, 'w') as f:
            json.dump(sample_tokens, f)
        
        labeled_tokens = labeler.label_tokens(sample_path)
        sample_path.unlink()  # Clean up
    else:
        labeled_tokens = labeler.label_tokens(tokens_path)
    
    print(f"\nLabeling complete! Results saved in {labeler.results_dir}")


if __name__ == "__main__":
    main()