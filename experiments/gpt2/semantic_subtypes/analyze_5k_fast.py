"""
Fast analysis of 5k results using statistics only
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

def main():
    output_dir = Path("experiments/gpt2/semantic_subtypes/5k_common_words")
    
    # Load word features
    print("Loading word features...")
    with open(output_dir / "wordnet_features.json", 'r') as f:
        words = json.load(f)
    
    print(f"\n=== Dataset Overview ===")
    print(f"Total words: {len(words)}")
    
    # POS distribution
    pos_counts = Counter(w['primary_pos'] for w in words if w['primary_pos'])
    print(f"\nPOS Distribution:")
    for pos, count in pos_counts.most_common():
        print(f"  {pos}: {count} ({count/len(words)*100:.1f}%)")
    
    # Words with multiple POS
    multi_pos = sum(1 for w in words if w['secondary_pos'])
    print(f"\nWords with multiple POS: {multi_pos} ({multi_pos/len(words)*100:.1f}%)")
    
    # Polysemy
    polysemy = [w['polysemy_count'] for w in words]
    print(f"\nPolysemy Statistics:")
    print(f"  Average senses: {np.mean(polysemy):.2f}")
    print(f"  Monosemous: {sum(1 for p in polysemy if p == 1)} ({sum(1 for p in polysemy if p == 1)/len(polysemy)*100:.1f}%)")
    print(f"  Highly polysemous (>5): {sum(1 for p in polysemy if p > 5)}")
    
    # Semantic properties
    properties = ['is_concrete', 'is_abstract', 'is_animate', 'is_artifact', 'is_physical', 'is_mental']
    print(f"\nSemantic Properties:")
    for prop in properties:
        count = sum(1 for w in words if w.get(prop, False))
        print(f"  {prop}: {count} ({count/len(words)*100:.1f}%)")
    
    # Interesting combinations
    print(f"\nInteresting Combinations:")
    
    # Concrete vs Abstract nouns
    nouns = [w for w in words if w['primary_pos'] == 'n']
    concrete_nouns = sum(1 for w in nouns if w.get('is_concrete', False))
    abstract_nouns = sum(1 for w in nouns if w.get('is_abstract', False))
    print(f"  Nouns ({len(nouns)}): {concrete_nouns} concrete, {abstract_nouns} abstract")
    
    # Physical vs Mental verbs  
    verbs = [w for w in words if w['primary_pos'] == 'v']
    physical_verbs = sum(1 for w in verbs if w.get('is_physical', False))
    mental_verbs = sum(1 for w in verbs if w.get('is_mental', False))
    print(f"  Verbs ({len(verbs)}): {physical_verbs} physical, {mental_verbs} mental")
    
    # Semantic domains
    all_domains = []
    for w in words:
        all_domains.extend(w.get('semantic_domains', []))
    
    domain_counts = Counter(all_domains)
    print(f"\nTop 10 Semantic Domains:")
    for domain, count in domain_counts.most_common(10):
        print(f"  {domain}: {count}")
    
    # Sample words by category
    print(f"\nSample Words by Category:")
    
    # Concrete animate nouns
    concrete_animate = [w['word'] for w in words 
                       if w['primary_pos'] == 'n' 
                       and w.get('is_concrete', False) 
                       and w.get('is_animate', False)][:5]
    print(f"  Concrete animate nouns: {', '.join(concrete_animate)}")
    
    # Abstract nouns
    abstract_sample = [w['word'] for w in words 
                      if w['primary_pos'] == 'n' 
                      and w.get('is_abstract', False)][:5]
    print(f"  Abstract nouns: {', '.join(abstract_sample)}")
    
    # Highly polysemous verbs
    poly_verbs = sorted([w for w in words if w['primary_pos'] == 'v'], 
                       key=lambda x: x['polysemy_count'], reverse=True)[:5]
    poly_verb_strs = [f"{w['word']}({w['polysemy_count']})" for w in poly_verbs]
    print(f"  Highly polysemous verbs: {', '.join(poly_verb_strs)}")
    
    print(f"\n=== Key Insights ===")
    print(f"1. Successfully extracted {len(words)} single-token words")
    print(f"2. Rich linguistic diversity: {len(pos_counts)} POS types, {len(domain_counts)} semantic domains")
    print(f"3. Dataset includes both concrete ({sum(1 for w in words if w.get('is_concrete', False))}) and abstract ({sum(1 for w in words if w.get('is_abstract', False))}) concepts")
    print(f"4. Ready for testing whether grammatical organization dominates semantic grouping")

if __name__ == "__main__":
    main()