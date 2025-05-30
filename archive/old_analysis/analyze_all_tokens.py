"""
Analyze ALL GPT-2 tokens (50,257) to understand vocabulary composition
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from transformers import GPT2Tokenizer
import re
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class GPT2TokenAnalyzer:
    """Comprehensive analysis of GPT-2's full vocabulary"""
    
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.vocab_size = len(self.tokenizer)
        print(f"Loaded GPT-2 tokenizer with {self.vocab_size} tokens")
        
    def analyze_token(self, token_id: int) -> Dict:
        """Analyze a single token's properties"""
        
        # Get token string
        token_str = self.tokenizer.decode([token_id])
        raw_token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        
        # Basic properties
        analysis = {
            'token_id': token_id,
            'token_str': token_str,
            'raw_token': raw_token,
            'byte_length': len(token_str.encode('utf-8')),
            'char_length': len(token_str),
            'has_leading_space': raw_token.startswith('Ġ'),  # GPT-2 uses Ġ for space
        }
        
        # Token type classification
        analysis['token_type'] = self._classify_token_type(raw_token, token_str)
        
        # Subword analysis
        if '##' in raw_token or raw_token.startswith('Ġ'):
            analysis['is_subword'] = True
            analysis['subword_type'] = self._classify_subword_type(raw_token)
        else:
            analysis['is_subword'] = False
            analysis['subword_type'] = None
        
        # Character composition
        analysis['is_alphabetic'] = token_str.replace(' ', '').isalpha()
        analysis['is_numeric'] = token_str.replace(' ', '').isdigit()
        analysis['is_alphanumeric'] = token_str.replace(' ', '').isalnum()
        analysis['is_punctuation'] = all(not c.isalnum() and c != ' ' for c in token_str)
        analysis['is_mixed'] = any(c.isalpha() for c in token_str) and any(c.isdigit() for c in token_str)
        
        # Capitalization
        if analysis['is_alphabetic']:
            analysis['is_uppercase'] = token_str.isupper()
            analysis['is_lowercase'] = token_str.islower()
            analysis['is_titlecase'] = token_str.istitle()
            analysis['is_mixed_case'] = not (analysis['is_uppercase'] or analysis['is_lowercase'] or analysis['is_titlecase'])
        else:
            analysis['is_uppercase'] = False
            analysis['is_lowercase'] = False
            analysis['is_titlecase'] = False
            analysis['is_mixed_case'] = False
        
        # Special tokens
        analysis['is_special'] = token_id in [
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id
        ] if any([
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id
        ]) else False
        
        # Language detection (simple heuristic)
        analysis['likely_language'] = self._detect_language(token_str)
        
        return analysis
    
    def _classify_token_type(self, raw_token: str, token_str: str) -> str:
        """Classify token into categories"""
        
        if '<|endoftext|>' in raw_token:
            return 'special'
        elif raw_token.startswith('Ġ'):
            clean_token = raw_token[1:]  # Remove Ġ
            if clean_token.isalpha():
                return 'word_with_space'
            elif clean_token.isdigit():
                return 'number_with_space'
            else:
                return 'mixed_with_space'
        elif '##' in raw_token:
            return 'subword_suffix'
        elif token_str.isalpha():
            return 'word'
        elif token_str.isdigit():
            return 'number'
        elif all(not c.isalnum() for c in token_str):
            return 'punctuation'
        elif any(c.isalpha() for c in token_str) and any(c.isdigit() for c in token_str):
            return 'alphanumeric'
        else:
            return 'other'
    
    def _classify_subword_type(self, raw_token: str) -> str:
        """Classify subword tokens by their morphological function"""
        
        # Common English suffixes
        if raw_token.endswith('ing'):
            return 'suffix_ing'
        elif raw_token.endswith('ed'):
            return 'suffix_ed'
        elif raw_token.endswith('ly'):
            return 'suffix_ly'
        elif raw_token.endswith('er'):
            return 'suffix_er'
        elif raw_token.endswith('est'):
            return 'suffix_est'
        elif raw_token.endswith('tion'):
            return 'suffix_tion'
        elif raw_token.endswith('ment'):
            return 'suffix_ment'
        elif raw_token.endswith('ness'):
            return 'suffix_ness'
        elif raw_token.endswith('ity'):
            return 'suffix_ity'
        elif raw_token.endswith('able') or raw_token.endswith('ible'):
            return 'suffix_able'
        elif raw_token.endswith('ful'):
            return 'suffix_ful'
        elif raw_token.endswith('less'):
            return 'suffix_less'
        elif raw_token.endswith('s') and len(raw_token) > 2:
            return 'suffix_plural'
        elif raw_token.startswith('Ġ'):
            return 'prefix_space'
        else:
            return 'other_subword'
    
    def _detect_language(self, token_str: str) -> str:
        """Simple language detection based on character sets"""
        
        if not any(c.isalpha() for c in token_str):
            return 'none'
        
        # Check for non-ASCII characters
        if any(ord(c) > 127 for c in token_str):
            # Simple heuristics for common languages
            if any('\u4e00' <= c <= '\u9fff' for c in token_str):
                return 'chinese'
            elif any('\u0400' <= c <= '\u04ff' for c in token_str):
                return 'cyrillic'
            elif any('\u0600' <= c <= '\u06ff' for c in token_str):
                return 'arabic'
            elif any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in token_str):
                return 'japanese'
            elif any('\uac00' <= c <= '\ud7af' for c in token_str):
                return 'korean'
            else:
                return 'other_non_ascii'
        else:
            return 'ascii'
    
    def analyze_full_vocabulary(self) -> Dict:
        """Analyze all tokens in the vocabulary"""
        
        print("Analyzing full GPT-2 vocabulary...")
        all_analyses = []
        
        # Analyze each token
        for token_id in range(self.vocab_size):
            if token_id % 5000 == 0:
                print(f"Processed {token_id}/{self.vocab_size} tokens...")
            
            analysis = self.analyze_token(token_id)
            all_analyses.append(analysis)
        
        # Compute statistics
        stats = self._compute_statistics(all_analyses)
        
        return {
            'token_analyses': all_analyses,
            'statistics': stats
        }
    
    def _compute_statistics(self, analyses: List[Dict]) -> Dict:
        """Compute aggregate statistics"""
        
        stats = {
            'total_tokens': len(analyses),
            'token_types': Counter(a['token_type'] for a in analyses),
            'byte_lengths': Counter(a['byte_length'] for a in analyses),
            'char_lengths': Counter(a['char_length'] for a in analyses),
            'subword_types': Counter(a['subword_type'] for a in analyses if a['is_subword']),
            'languages': Counter(a['likely_language'] for a in analyses),
            'with_leading_space': sum(1 for a in analyses if a['has_leading_space']),
            'alphabetic': sum(1 for a in analyses if a['is_alphabetic']),
            'numeric': sum(1 for a in analyses if a['is_numeric']),
            'punctuation': sum(1 for a in analyses if a['is_punctuation']),
            'mixed': sum(1 for a in analyses if a['is_mixed']),
            'special': sum(1 for a in analyses if a['is_special']),
        }
        
        # Capitalization stats for alphabetic tokens
        alpha_tokens = [a for a in analyses if a['is_alphabetic']]
        stats['capitalization'] = {
            'uppercase': sum(1 for a in alpha_tokens if a['is_uppercase']),
            'lowercase': sum(1 for a in alpha_tokens if a['is_lowercase']),
            'titlecase': sum(1 for a in alpha_tokens if a['is_titlecase']),
            'mixed_case': sum(1 for a in alpha_tokens if a['is_mixed_case']),
        }
        
        return stats
    
    def visualize_statistics(self, stats: Dict, output_dir: Path):
        """Create visualizations of token statistics"""
        
        # Token type distribution
        plt.figure(figsize=(10, 6))
        token_types = stats['token_types']
        plt.bar(token_types.keys(), token_types.values())
        plt.xticks(rotation=45, ha='right')
        plt.title('GPT-2 Token Type Distribution')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(output_dir / 'token_type_distribution.png')
        plt.close()
        
        # Byte length distribution
        plt.figure(figsize=(10, 6))
        byte_lengths = stats['byte_lengths']
        max_len = max(byte_lengths.keys())
        x = list(range(max_len + 1))
        y = [byte_lengths.get(i, 0) for i in x]
        plt.bar(x[:20], y[:20])  # Show first 20
        plt.title('Token Byte Length Distribution')
        plt.xlabel('Byte Length')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(output_dir / 'byte_length_distribution.png')
        plt.close()
        
        # Subword type distribution
        plt.figure(figsize=(12, 6))
        subword_types = stats['subword_types']
        if subword_types:
            plt.bar(subword_types.keys(), subword_types.values())
            plt.xticks(rotation=45, ha='right')
            plt.title('Subword Token Type Distribution')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(output_dir / 'subword_type_distribution.png')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}")


def main():
    """Run the full vocabulary analysis"""
    
    output_dir = Path("experiments/gpt2/all_tokens")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = GPT2TokenAnalyzer()
    
    # Analyze vocabulary
    results = analyzer.analyze_full_vocabulary()
    
    # Save results
    print("\nSaving results...")
    
    # Save full analysis (large file)
    with open(output_dir / "full_token_analysis.json", 'w') as f:
        json.dump(results['token_analyses'], f)
    
    # Save statistics
    with open(output_dir / "token_statistics.json", 'w') as f:
        # Convert Counter objects to dicts for JSON serialization
        stats = results['statistics']
        stats_serializable = {
            k: dict(v) if isinstance(v, Counter) else v
            for k, v in stats.items()
        }
        json.dump(stats_serializable, f, indent=2)
    
    # Create visualizations
    analyzer.visualize_statistics(results['statistics'], output_dir)
    
    # Print summary
    print("\n=== GPT-2 Vocabulary Summary ===")
    stats = results['statistics']
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"\nToken types:")
    for token_type, count in stats['token_types'].most_common():
        print(f"  {token_type}: {count} ({count/stats['total_tokens']*100:.1f}%)")
    
    print(f"\nCharacter composition:")
    print(f"  Alphabetic: {stats['alphabetic']} ({stats['alphabetic']/stats['total_tokens']*100:.1f}%)")
    print(f"  Numeric: {stats['numeric']} ({stats['numeric']/stats['total_tokens']*100:.1f}%)")
    print(f"  Punctuation: {stats['punctuation']} ({stats['punctuation']/stats['total_tokens']*100:.1f}%)")
    print(f"  Mixed: {stats['mixed']} ({stats['mixed']/stats['total_tokens']*100:.1f}%)")
    
    print(f"\nSubword tokens: {sum(stats['subword_types'].values())}")
    print(f"Tokens with leading space: {stats['with_leading_space']}")
    
    print(f"\nLanguages detected:")
    for lang, count in stats['languages'].most_common(10):
        print(f"  {lang}: {count}")
    
    # Find interesting examples
    print("\n=== Interesting Token Examples ===")
    
    # Longest tokens
    longest_tokens = sorted(results['token_analyses'], 
                          key=lambda x: x['byte_length'], 
                          reverse=True)[:10]
    print("\nLongest tokens (by bytes):")
    for t in longest_tokens:
        print(f"  {t['token_id']}: '{t['raw_token']}' ({t['byte_length']} bytes)")
    
    # Non-ASCII tokens
    non_ascii = [t for t in results['token_analyses'] if t['likely_language'] != 'ascii' and t['likely_language'] != 'none'][:10]
    if non_ascii:
        print("\nNon-ASCII token examples:")
        for t in non_ascii:
            print(f"  {t['token_id']}: '{t['raw_token']}' ({t['likely_language']})")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()