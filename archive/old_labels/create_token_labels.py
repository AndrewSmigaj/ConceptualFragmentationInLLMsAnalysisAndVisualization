#!/usr/bin/env python3
"""
Create linguistic labels for GPT-2 tokens.
This script analyzes tokens and creates comprehensive labels for trajectory analysis.
"""

import json
from pathlib import Path
import logging
from collections import defaultdict
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TokenLabeler:
    def __init__(self):
        # Define grammatical categories for common patterns
        self.suffix_grammar = {
            'ing': {'pos': 'SUFFIX', 'function': 'progressive/gerund'},
            'ed': {'pos': 'SUFFIX', 'function': 'past_tense/participle'},
            'er': {'pos': 'SUFFIX', 'function': 'comparative/agent'},
            'est': {'pos': 'SUFFIX', 'function': 'superlative'},
            'ly': {'pos': 'SUFFIX', 'function': 'adverb_forming'},
            'tion': {'pos': 'SUFFIX', 'function': 'nominalization'},
            'ment': {'pos': 'SUFFIX', 'function': 'nominalization'},
            'ness': {'pos': 'SUFFIX', 'function': 'quality_noun'},
            'ity': {'pos': 'SUFFIX', 'function': 'quality_noun'},
            'able': {'pos': 'SUFFIX', 'function': 'capability_adj'},
            'ful': {'pos': 'SUFFIX', 'function': 'characterized_by'},
            'less': {'pos': 'SUFFIX', 'function': 'without'},
            's': {'pos': 'SUFFIX', 'function': 'plural/possessive/3rd_person'},
            'es': {'pos': 'SUFFIX', 'function': 'plural/3rd_person'},
            'ies': {'pos': 'SUFFIX', 'function': 'plural'},
        }
        
        self.prefix_grammar = {
            'un': {'semantic': 'negation', 'productivity': 'high'},
            're': {'semantic': 'repetition/back', 'productivity': 'high'},
            'dis': {'semantic': 'negation/reversal', 'productivity': 'high'},
            'pre': {'semantic': 'before', 'productivity': 'high'},
            'post': {'semantic': 'after', 'productivity': 'medium'},
            'over': {'semantic': 'excess', 'productivity': 'medium'},
            'under': {'semantic': 'insufficient', 'productivity': 'medium'},
            'co': {'semantic': 'together', 'productivity': 'high'},
            'anti': {'semantic': 'against', 'productivity': 'medium'},
            'non': {'semantic': 'not', 'productivity': 'high'},
        }
        
    def label_token(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive label for a single token."""
        token_id = token_data['token_id']
        token_str = token_data['token_str']
        token_clean = token_str.strip()
        
        # Initialize label
        label = {
            'token_id': token_id,
            'token': token_str,
            'grammatical': {},
            'semantic': {},
            'morphological': {},
            'conceptual_hierarchy': [],
            'usage_context': ''
        }
        
        # Categorize by token type
        if token_data.get('is_punctuation', False):
            self._label_punctuation(token_clean, label)
        elif token_data.get('is_numeric', False):
            self._label_number(token_clean, label)
        elif len(token_clean) == 1 and token_clean.isalpha():
            self._label_single_letter(token_clean, label)
        elif token_data.get('is_subword', False) or token_clean.startswith('##'):
            self._label_subword(token_clean, label)
        else:
            self._label_word(token_clean, label, token_data)
        
        return label
    
    def _label_punctuation(self, token: str, label: Dict):
        """Label punctuation marks."""
        punct_info = {
            '.': {'subtype': 'period', 'function': 'sentence_end', 'context': 'Ends declarative sentences'},
            ',': {'subtype': 'comma', 'function': 'separator', 'context': 'Separates clauses or list items'},
            '!': {'subtype': 'exclamation', 'function': 'emphasis', 'context': 'Expresses strong emotion'},
            '?': {'subtype': 'question', 'function': 'interrogative', 'context': 'Marks questions'},
            ';': {'subtype': 'semicolon', 'function': 'clause_separator', 'context': 'Joins independent clauses'},
            ':': {'subtype': 'colon', 'function': 'introducer', 'context': 'Introduces lists or explanations'},
            '"': {'subtype': 'quotation', 'function': 'delimiter', 'context': 'Marks quoted text'},
            "'": {'subtype': 'apostrophe', 'function': 'contraction/possessive', 'context': 'Marks contractions or possession'},
            '(': {'subtype': 'paren_open', 'function': 'grouping', 'context': 'Opens parenthetical'},
            ')': {'subtype': 'paren_close', 'function': 'grouping', 'context': 'Closes parenthetical'},
            '-': {'subtype': 'hyphen', 'function': 'connector', 'context': 'Joins compound words'},
            '/': {'subtype': 'slash', 'function': 'alternative', 'context': 'Indicates alternatives'},
        }
        
        label['grammatical']['pos'] = 'PUNCT'
        info = punct_info.get(token, {'subtype': 'other', 'function': 'unknown', 'context': 'Punctuation mark'})
        label['grammatical']['subtype'] = info['subtype']
        label['semantic']['category'] = 'punctuation'
        label['semantic']['function'] = info['function']
        label['morphological']['role'] = 'punctuation'
        label['usage_context'] = info['context']
        label['conceptual_hierarchy'] = ['punctuation', 'orthographic_symbol', 'linguistic_marker']
    
    def _label_number(self, token: str, label: Dict):
        """Label numeric tokens."""
        label['grammatical']['pos'] = 'NUM'
        label['semantic']['category'] = 'number'
        
        if len(token) == 1:
            label['semantic']['subcategory'] = 'digit'
            label['usage_context'] = f'Single digit {token}, used in numbers, codes, lists'
        else:
            label['semantic']['subcategory'] = 'multi_digit'
            label['usage_context'] = f'Number {token}, could be year, quantity, identifier'
        
        label['morphological']['role'] = 'numeral'
        label['conceptual_hierarchy'] = ['number', 'quantity', 'mathematical_object']
    
    def _label_single_letter(self, token: str, label: Dict):
        """Label single letter tokens."""
        label['grammatical']['pos'] = 'LETTER'
        label['semantic']['category'] = 'alphabetic_unit'
        
        if token.isupper():
            label['grammatical']['subtype'] = 'uppercase'
            label['semantic']['subcategory'] = 'initial/abbreviation'
            label['usage_context'] = 'Uppercase letter: abbreviations, initials, emphasis'
        else:
            label['grammatical']['subtype'] = 'lowercase'
            label['semantic']['subcategory'] = 'variable/pronoun'
            if token in ['a', 'i']:
                label['grammatical']['pos'] = 'DET' if token == 'a' else 'PRON'
                label['usage_context'] = f'Single-letter word: {"indefinite article" if token == "a" else "first person pronoun"}'
            else:
                label['usage_context'] = 'Lowercase letter: variables, abbreviations, list markers'
        
        label['morphological']['role'] = 'atomic_unit'
        label['conceptual_hierarchy'] = ['letter', 'alphabetic_symbol', 'writing_system_unit']
    
    def _label_subword(self, token: str, label: Dict):
        """Label subword tokens."""
        # Remove ## prefix if present
        clean_token = token.replace('##', '')
        
        label['semantic']['category'] = 'subword'
        label['morphological']['role'] = 'fragment'
        
        # Check if it's a known suffix
        for suffix, info in self.suffix_grammar.items():
            if clean_token == suffix or clean_token.endswith(suffix):
                label['grammatical']['pos'] = 'SUFFIX'
                label['grammatical']['subtype'] = suffix
                label['semantic']['subcategory'] = 'morphological_suffix'
                label['semantic']['function'] = info['function']
                label['morphological']['role'] = 'suffix'
                label['usage_context'] = f'Suffix -{suffix}: {info["function"]}'
                label['conceptual_hierarchy'] = ['suffix', 'bound_morpheme', 'morphological_unit']
                return
        
        # Check if it's a known prefix (when appearing as subword)
        for prefix, info in self.prefix_grammar.items():
            if clean_token == prefix or clean_token.startswith(prefix):
                label['grammatical']['pos'] = 'PREFIX'
                label['grammatical']['subtype'] = prefix
                label['semantic']['subcategory'] = 'morphological_prefix'
                label['semantic']['function'] = info['semantic']
                label['morphological']['role'] = 'prefix'
                label['usage_context'] = f'Prefix {prefix}-: {info["semantic"]}'
                label['conceptual_hierarchy'] = ['prefix', 'bound_morpheme', 'morphological_unit']
                return
        
        # Generic subword
        label['grammatical']['pos'] = 'FRAGMENT'
        label['semantic']['subcategory'] = 'word_piece'
        label['usage_context'] = 'Subword unit, part of longer word'
        label['conceptual_hierarchy'] = ['subword', 'token_fragment', 'tokenization_unit']
    
    def _label_word(self, token: str, label: Dict[str, Any], token_data: Dict[str, Any]):
        """Label complete words (basic heuristics)."""
        label['semantic']['category'] = 'word'
        label['morphological']['role'] = 'root_word'
        
        # Common function words
        function_words = {
            'the': {'pos': 'DET', 'subtype': 'definite_article', 'context': 'Most common English word, definite article'},
            'a': {'pos': 'DET', 'subtype': 'indefinite_article', 'context': 'Indefinite article'},
            'an': {'pos': 'DET', 'subtype': 'indefinite_article', 'context': 'Indefinite article before vowels'},
            'and': {'pos': 'CONJ', 'subtype': 'coordinating', 'context': 'Coordinating conjunction'},
            'or': {'pos': 'CONJ', 'subtype': 'coordinating', 'context': 'Alternative conjunction'},
            'but': {'pos': 'CONJ', 'subtype': 'coordinating', 'context': 'Contrastive conjunction'},
            'in': {'pos': 'PREP', 'subtype': 'location/time', 'context': 'Preposition of place or time'},
            'on': {'pos': 'PREP', 'subtype': 'location/time', 'context': 'Preposition of surface or time'},
            'at': {'pos': 'PREP', 'subtype': 'location/time', 'context': 'Preposition of specific location or time'},
            'to': {'pos': 'PREP', 'subtype': 'direction/infinitive', 'context': 'Preposition or infinitive marker'},
            'for': {'pos': 'PREP', 'subtype': 'purpose/duration', 'context': 'Preposition of purpose or time'},
            'of': {'pos': 'PREP', 'subtype': 'possession/part', 'context': 'Preposition of relationship'},
            'with': {'pos': 'PREP', 'subtype': 'accompaniment', 'context': 'Preposition of accompaniment'},
            'is': {'pos': 'VERB', 'subtype': 'copula', 'context': 'Third person singular of "be"'},
            'are': {'pos': 'VERB', 'subtype': 'copula', 'context': 'Plural/2nd person of "be"'},
            'was': {'pos': 'VERB', 'subtype': 'copula', 'context': 'Past singular of "be"'},
            'were': {'pos': 'VERB', 'subtype': 'copula', 'context': 'Past plural/2nd person of "be"'},
            'have': {'pos': 'VERB', 'subtype': 'auxiliary/main', 'context': 'Auxiliary or possession verb'},
            'has': {'pos': 'VERB', 'subtype': 'auxiliary/main', 'context': '3rd person singular of "have"'},
            'had': {'pos': 'VERB', 'subtype': 'auxiliary/main', 'context': 'Past tense of "have"'},
        }
        
        token_lower = token.lower()
        
        if token_lower in function_words:
            info = function_words[token_lower]
            label['grammatical']['pos'] = info['pos']
            label['grammatical']['subtype'] = info['subtype']
            label['semantic']['subcategory'] = 'function_word'
            label['usage_context'] = info['context']
            label['conceptual_hierarchy'] = ['function_word', 'grammatical_marker', 'linguistic_unit']
        else:
            # Default to NOUN for unknown words (statistically most likely)
            label['grammatical']['pos'] = 'NOUN'
            label['semantic']['subcategory'] = 'content_word'
            label['usage_context'] = 'Content word, likely noun or name'
            label['conceptual_hierarchy'] = ['word', 'lexical_unit', 'linguistic_unit']
    
    def label_tokens(self, tokens: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Label a list of tokens."""
        labels = {}
        for token_data in tokens:
            token_id = token_data['token_id']
            labels[token_id] = self.label_token(token_data)
        return labels


def main():
    base_dir = Path(__file__).parent
    output_dir = base_dir / "token_labels_final"
    output_dir.mkdir(exist_ok=True)
    
    # Load tokens
    tokens_path = base_dir / "top_10k_tokens_full.json"
    if not tokens_path.exists():
        logging.error(f"Token file not found: {tokens_path}")
        return
    
    logging.info(f"Loading tokens from {tokens_path}")
    with open(tokens_path, 'r', encoding='utf-8') as f:
        all_tokens = json.load(f)
    
    # Process in batches
    batch_size = 1000
    labeler = TokenLabeler()
    all_labels = {}
    
    for i in range(0, len(all_tokens), batch_size):
        batch = all_tokens[i:i + batch_size]
        logging.info(f"Processing tokens {i} to {i + len(batch)}")
        batch_labels = labeler.label_tokens(batch)
        all_labels.update(batch_labels)
    
    # Save labels
    output_path = output_dir / "token_labels_10k.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_labels, f, indent=2)
    
    logging.info(f"Saved {len(all_labels)} token labels to {output_path}")
    
    # Generate statistics
    stats = defaultdict(lambda: defaultdict(int))
    for label in all_labels.values():
        pos = label['grammatical'].get('pos', 'UNKNOWN')
        stats['pos'][pos] += 1
        category = label['semantic'].get('category', 'unknown')
        stats['category'][category] += 1
        role = label['morphological'].get('role', 'unknown')
        stats['role'][role] += 1
    
    # Print summary
    print("\n=== Token Labeling Summary ===")
    print(f"Total tokens labeled: {len(all_labels)}")
    
    print("\nPOS Distribution:")
    for pos, count in sorted(stats['pos'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {pos}: {count} ({count/len(all_labels)*100:.1f}%)")
    
    print("\nSemantic Categories:")
    for cat, count in sorted(stats['category'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count} ({count/len(all_labels)*100:.1f}%)")
    
    print("\nMorphological Roles:")
    for role, count in sorted(stats['role'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {role}: {count} ({count/len(all_labels)*100:.1f}%)")


if __name__ == "__main__":
    main()