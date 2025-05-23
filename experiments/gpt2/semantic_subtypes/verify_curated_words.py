#!/usr/bin/env python3
"""Verify the 774 curated words are available and valid."""

import json
import sys
from pathlib import Path

def verify_curated_words():
    """Verify the curated words file exists and contains expected data."""
    data_file = Path("data/gpt2_semantic_subtypes_curated.json")
    
    if not data_file.exists():
        print(f"✗ Curated words file not found: {data_file}")
        return False
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Expected structure
        expected_subtypes = [
            'concrete_nouns',
            'abstract_nouns', 
            'physical_adjectives',
            'emotive_adjectives',
            'manner_adverbs',
            'degree_adverbs',
            'action_verbs',
            'stative_verbs'
        ]
        
        print("Curated Words Verification")
        print("=" * 50)
        
        # Check metadata
        if 'metadata' in data:
            print(f"\nMetadata:")
            print(f"  Creation date: {data['metadata'].get('creation_date', 'N/A')}")
            print(f"  Total candidates: {data['metadata'].get('total_candidates', 'N/A')}")
            print(f"  Validation method: {data['metadata'].get('validation_method', 'N/A')}")
        
        # Check semantic subtypes (key is 'curated_words')
        if 'curated_words' not in data:
            print("\n✗ Missing 'curated_words' key")
            return False
        
        subtypes = data['curated_words']
        total_words = 0
        
        print(f"\nSemantic Subtypes (Expected 8):")
        for subtype in expected_subtypes:
            if subtype in subtypes:
                count = len(subtypes[subtype])
                total_words += count
                print(f"  ✓ {subtype}: {count} words")
            else:
                print(f"  ✗ {subtype}: MISSING")
        
        # Check unexpected subtypes
        unexpected = set(subtypes.keys()) - set(expected_subtypes)
        if unexpected:
            print(f"\n⚠ Unexpected subtypes found: {unexpected}")
        
        print(f"\nTotal Words: {total_words} (Expected: ~774)")
        
        # Validate statistics file
        stats_file = Path("data/gpt2_semantic_subtypes_statistics.json")
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            if 'distribution_analysis' in stats:
                dist = stats['distribution_analysis']
                print(f"\nStatistics Validation:")
                print(f"  Total words (stats): {dist.get('total_words', 'N/A')}")
                print(f"  Balance score: {dist.get('balance_score', 'N/A'):.2%}")
                print(f"  Achievement rate: {dist.get('achievement_rate', 'N/A'):.2%}")
        
        # Summary
        if total_words >= 700 and total_words <= 800:
            print(f"\n✓ Data validation successful! {total_words} words ready for analysis.")
            return True
        else:
            print(f"\n✗ Unexpected word count: {total_words}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error reading curated words: {e}")
        return False

if __name__ == "__main__":
    success = verify_curated_words()
    sys.exit(0 if success else 1)