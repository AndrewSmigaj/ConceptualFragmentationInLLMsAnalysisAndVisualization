"""
GPT-2 Semantic Subtypes Curator for Expanded Word List

Validates and curates single-token words from the expanded semantic subtypes wordlist.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import GPT2TokenizerFast
import json
from typing import Dict, List
from collections import defaultdict

class GPT2SemanticSubtypesCurator:
    """Curates semantic subtypes word lists using GPT-2 token validation."""
    
    def __init__(self, word_lists: Dict[str, List[str]]):
        """Initialize with word lists to curate."""
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.word_lists = word_lists
        self.curated_words = defaultdict(list)
        self.validation_stats = defaultdict(dict)
        
    def is_single_token(self, word: str) -> bool:
        """Check if a word is tokenized as a single token."""
        # Add space before word to match GPT-2 tokenization
        tokens = self.tokenizer.encode(' ' + word, add_special_tokens=False)
        return len(tokens) == 1
    
    def validate_all_words(self):
        """Validate all words and create curated lists."""
        print("=== Validating Expanded Word Lists ===")
        
        total_words = 0
        total_valid = 0
        
        for subtype, words in self.word_lists.items():
            valid_words = []
            invalid_words = []
            
            for word in words:
                if self.is_single_token(word):
                    valid_words.append(word)
                else:
                    invalid_words.append(word)
            
            # Store results
            self.curated_words[subtype] = valid_words
            self.validation_stats[subtype] = {
                'total': len(words),
                'valid': len(valid_words),
                'invalid': len(invalid_words),
                'validation_rate': len(valid_words) / len(words) if words else 0
            }
            
            total_words += len(words)
            total_valid += len(valid_words)
            
            print(f"\n{subtype}:")
            print(f"  Total candidates: {len(words)}")
            print(f"  Valid single-token: {len(valid_words)}")
            print(f"  Validation rate: {len(valid_words)/len(words)*100:.1f}%")
            
            # Show some examples of invalid words
            if invalid_words and len(invalid_words) <= 5:
                print(f"  Invalid words: {invalid_words}")
            elif invalid_words:
                print(f"  Invalid words (first 5): {invalid_words[:5]}")
        
        print(f"\n=== Overall Statistics ===")
        print(f"Total candidate words: {total_words}")
        print(f"Total valid words: {total_valid}")
        print(f"Overall validation rate: {total_valid/total_words*100:.1f}%")
        
    def get_curation_statistics(self) -> Dict:
        """Get detailed statistics about the curation process."""
        # Calculate grammatical category totals
        grammatical_stats = {
            'nouns': {
                'total': len(self.curated_words.get('concrete_nouns', [])) + 
                        len(self.curated_words.get('abstract_nouns', [])),
                'concrete': len(self.curated_words.get('concrete_nouns', [])),
                'abstract': len(self.curated_words.get('abstract_nouns', []))
            },
            'adjectives': {
                'total': len(self.curated_words.get('physical_adjectives', [])) + 
                        len(self.curated_words.get('emotive_adjectives', [])),
                'physical': len(self.curated_words.get('physical_adjectives', [])),
                'emotive': len(self.curated_words.get('emotive_adjectives', []))
            },
            'adverbs': {
                'total': len(self.curated_words.get('manner_adverbs', [])) + 
                        len(self.curated_words.get('degree_adverbs', [])),
                'manner': len(self.curated_words.get('manner_adverbs', [])),
                'degree': len(self.curated_words.get('degree_adverbs', []))
            },
            'verbs': {
                'total': len(self.curated_words.get('action_verbs', [])) + 
                        len(self.curated_words.get('stative_verbs', [])),
                'action': len(self.curated_words.get('action_verbs', [])),
                'stative': len(self.curated_words.get('stative_verbs', []))
            }
        }
        
        total_words = sum(cat['total'] for cat in grammatical_stats.values())
        
        return {
            'curated_words': dict(self.curated_words),
            'subtype_statistics': dict(self.validation_stats),
            'grammatical_statistics': grammatical_stats,
            'overall_statistics': {
                'total_curated_words': total_words,
                'grammatical_distribution': {
                    'nouns': grammatical_stats['nouns']['total'],
                    'nouns_percentage': grammatical_stats['nouns']['total'] / total_words * 100,
                    'adjectives': grammatical_stats['adjectives']['total'],
                    'adjectives_percentage': grammatical_stats['adjectives']['total'] / total_words * 100,
                    'adverbs': grammatical_stats['adverbs']['total'],
                    'adverbs_percentage': grammatical_stats['adverbs']['total'] / total_words * 100,
                    'verbs': grammatical_stats['verbs']['total'],
                    'verbs_percentage': grammatical_stats['verbs']['total'] / total_words * 100
                }
            }
        }
    
    def save_curated_dataset(self, output_path: str):
        """Save the curated dataset to JSON file."""
        stats = self.get_curation_statistics()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\nCurated dataset saved to: {output_path}")

if __name__ == "__main__":
    # Test with expanded word lists
    from gpt2_semantic_subtypes_wordlists_expanded import ALL_WORD_LISTS
    
    curator = GPT2SemanticSubtypesCurator(ALL_WORD_LISTS)
    curator.validate_all_words()
    curator.save_curated_dataset("data/gpt2_semantic_subtypes_curated_expanded.json")