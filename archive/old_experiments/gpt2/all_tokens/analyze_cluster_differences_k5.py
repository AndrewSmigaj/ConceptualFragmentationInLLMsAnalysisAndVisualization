#!/usr/bin/env python3
"""
Analyze differences between clusters to create more meaningful, differentiated labels.
Focus on what makes each cluster unique rather than broad categories.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

def analyze_cluster_differences():
    """Analyze what makes each cluster unique within each layer."""
    base_dir = Path(__file__).parent
    
    # Load full clustering data
    with open(base_dir / "llm_labels_k5" / "llm_labeling_data.json", 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    print("DETAILED CLUSTER ANALYSIS")
    print("=" * 80)
    
    for layer in range(12):
        print(f"\n\nLAYER {layer}")
        print("-" * 40)
        
        # Get all clusters for this layer
        layer_clusters = []
        for cluster_idx in range(5):
            cluster_key = f"L{layer}_C{cluster_idx}"
            cluster_data = full_data["clusters"][cluster_key]
            layer_clusters.append({
                'key': cluster_key,
                'idx': cluster_idx,
                'tokens': cluster_data.get("common_tokens", [])[:30],  # Look at more tokens
                'size': cluster_data["size"],
                'pct': cluster_data["percentage"]
            })
        
        # Analyze what distinguishes each cluster
        for cluster in layer_clusters:
            print(f"\nCluster {cluster['idx']} ({cluster['pct']:.1f}%):")
            tokens = cluster['tokens']
            
            # Clean tokens
            clean_tokens = [t.strip() for t in tokens]
            
            # Analyze characteristics
            characteristics = analyze_token_characteristics(clean_tokens)
            
            print(f"  Top tokens: {', '.join(tokens[:10])}")
            print(f"  Characteristics:")
            for char, value in characteristics.items():
                if value:
                    print(f"    - {char}: {value}")
            
            # Suggest label based on unique characteristics
            suggested_label = suggest_label(clean_tokens, characteristics)
            print(f"  Suggested label: {suggested_label}")


def analyze_token_characteristics(tokens):
    """Analyze detailed characteristics of tokens."""
    chars = {}
    
    # Length distribution
    lengths = [len(t) for t in tokens]
    avg_length = np.mean(lengths) if lengths else 0
    chars['avg_length'] = f"{avg_length:.1f}"
    
    # Check for specific patterns
    has_quotes = sum(1 for t in tokens if t in ["''", '``', '"', "'"]) 
    has_punct = sum(1 for t in tokens if t in '.,;:!?-')
    has_parens = sum(1 for t in tokens if t in '()')
    
    if has_quotes > 2:
        chars['quotes'] = f"{has_quotes} quote marks"
    if has_punct > 3:
        chars['punctuation'] = f"{has_punct} punct marks"
    if has_parens > 1:
        chars['parentheses'] = f"{has_parens} parens"
    
    # Pronouns
    pronouns = ['I', 'he', 'she', 'it', 'we', 'they', 'you', 'me', 'him', 'her', 'us', 'them']
    pron_count = sum(1 for t in tokens if t.lower() in pronouns)
    if pron_count > 2:
        chars['pronouns'] = f"{pron_count} pronouns"
    
    # Determiners/Articles
    determiners = ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their']
    det_count = sum(1 for t in tokens if t.lower() in determiners)
    if det_count > 2:
        chars['determiners'] = f"{det_count} determiners"
    
    # Prepositions
    preps = ['of', 'to', 'in', 'for', 'with', 'on', 'at', 'by', 'from', 'up', 'about', 'into', 'through', 'after', 'over', 'between', 'out', 'against', 'during', 'without', 'before', 'under', 'around', 'among']
    prep_count = sum(1 for t in tokens if t.lower() in preps)
    if prep_count > 3:
        chars['prepositions'] = f"{prep_count} prepositions"
    
    # Conjunctions
    conjs = ['and', 'or', 'but', 'if', 'because', 'as', 'that', 'when', 'while', 'although', 'since', 'unless', 'than', 'whether', 'so']
    conj_count = sum(1 for t in tokens if t.lower() in conjs)
    if conj_count > 2:
        chars['conjunctions'] = f"{conj_count} conjunctions"
    
    # Be verbs
    be_verbs = ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'am']
    be_count = sum(1 for t in tokens if t.lower() in be_verbs)
    if be_count > 1:
        chars['be_verbs'] = f"{be_count} be-verbs"
    
    # Auxiliaries
    aux = ['have', 'has', 'had', 'will', 'would', 'can', 'could', 'may', 'might', 'shall', 'should', 'must', 'do', 'does', 'did']
    aux_count = sum(1 for t in tokens if t.lower() in aux)
    if aux_count > 2:
        chars['auxiliaries'] = f"{aux_count} auxiliaries"
    
    # Common nouns
    common_nouns = ['time', 'people', 'year', 'way', 'day', 'man', 'thing', 'woman', 'life', 'child', 'world', 'school', 'state', 'family', 'student', 'group', 'country', 'problem', 'hand', 'part', 'place', 'case', 'week', 'company', 'system', 'program', 'question', 'work', 'government', 'number', 'night', 'point', 'home', 'water', 'room', 'mother', 'area', 'money']
    noun_count = sum(1 for t in tokens if t.lower() in common_nouns)
    if noun_count > 3:
        chars['common_nouns'] = f"{noun_count} common nouns"
    
    # Action verbs
    action_verbs = ['said', 'get', 'make', 'go', 'take', 'come', 'see', 'know', 'give', 'find', 'think', 'tell', 'become', 'leave', 'feel', 'put', 'bring', 'begin', 'keep', 'hold', 'write', 'stand', 'hear', 'let', 'mean', 'set', 'meet', 'run', 'pay', 'sit', 'speak', 'lie', 'lead', 'read', 'grow', 'lose', 'fall', 'send', 'build', 'understand', 'draw', 'break', 'spend', 'kill', 'remain', 'suggest', 'raise', 'pass', 'sell', 'require', 'report', 'decide', 'pull']
    verb_count = sum(1 for t in tokens if t.lower() in action_verbs)
    if verb_count > 2:
        chars['action_verbs'] = f"{verb_count} action verbs"
    
    # Morphological endings
    ing_count = sum(1 for t in tokens if t.endswith('ing'))
    ed_count = sum(1 for t in tokens if t.endswith('ed'))
    ly_count = sum(1 for t in tokens if t.endswith('ly'))
    s_count = sum(1 for t in tokens if t.endswith('s') and len(t) > 2)
    
    if ing_count > 2:
        chars['ing_endings'] = f"{ing_count} -ing words"
    if ed_count > 2:
        chars['ed_endings'] = f"{ed_count} -ed words"
    if ly_count > 2:
        chars['ly_endings'] = f"{ly_count} -ly words"
    if s_count > 3:
        chars['s_endings'] = f"{s_count} plural/3rd person"
    
    # Capitalized words (proper nouns, sentence starters)
    cap_count = sum(1 for t in tokens if t and t[0].isupper())
    if cap_count > 3:
        chars['capitalized'] = f"{cap_count} capitalized"
    
    # Short tokens (1-2 chars)
    short_count = sum(1 for t in tokens if len(t) <= 2)
    if short_count > 5:
        chars['short_tokens'] = f"{short_count} short (<=2 char)"
    
    # Special tokens
    special_count = sum(1 for t in tokens if any(c in t for c in ['@', '#', '$', '%', '&', '*']))
    if special_count > 0:
        chars['special_chars'] = f"{special_count} special chars"
    
    return chars


def suggest_label(tokens, characteristics):
    """Suggest a specific label based on token characteristics."""
    
    # Priority-based labeling
    if 'quotes' in characteristics and int(characteristics['quotes'].split()[0]) > 3:
        return "Quotation Markers"
    
    if 'punctuation' in characteristics and int(characteristics['punctuation'].split()[0]) > 5:
        return "Sentence Punctuation"
    
    if 'parentheses' in characteristics:
        return "Parentheticals"
    
    if 'pronouns' in characteristics and int(characteristics['pronouns'].split()[0]) > 4:
        if any(t in ['I', 'we', 'you'] for t in tokens[:10]):
            return "Personal Pronouns (1st/2nd)"
        else:
            return "3rd Person Pronouns"
    
    if 'determiners' in characteristics and int(characteristics['determiners'].split()[0]) > 3:
        return "Determiners & Articles"
    
    if 'prepositions' in characteristics and int(characteristics['prepositions'].split()[0]) > 4:
        return "Prepositional Words"
    
    if 'conjunctions' in characteristics and int(characteristics['conjunctions'].split()[0]) > 3:
        return "Conjunctions & Connectives"
    
    if 'be_verbs' in characteristics and int(characteristics['be_verbs'].split()[0]) > 2:
        return "Copular Verbs (be)"
    
    if 'auxiliaries' in characteristics and int(characteristics['auxiliaries'].split()[0]) > 3:
        return "Modal & Auxiliary Verbs"
    
    if 'common_nouns' in characteristics and int(characteristics['common_nouns'].split()[0]) > 4:
        # Look for semantic patterns
        if any(t in ['time', 'year', 'day', 'week', 'night'] for t in tokens[:15]):
            return "Temporal Nouns"
        elif any(t in ['people', 'man', 'woman', 'child', 'family'] for t in tokens[:15]):
            return "Human/Social Nouns"
        elif any(t in ['place', 'area', 'country', 'state', 'world'] for t in tokens[:15]):
            return "Location Nouns"
        else:
            return "General Nouns"
    
    if 'action_verbs' in characteristics and int(characteristics['action_verbs'].split()[0]) > 3:
        if any(t in ['said', 'tell', 'speak', 'ask'] for t in tokens[:15]):
            return "Communication Verbs"
        elif any(t in ['think', 'know', 'feel', 'understand'] for t in tokens[:15]):
            return "Cognitive/Mental Verbs"
        else:
            return "Action Verbs"
    
    if 'ing_endings' in characteristics and int(characteristics['ing_endings'].split()[0]) > 3:
        return "Progressive/Gerund Forms"
    
    if 'ed_endings' in characteristics and int(characteristics['ed_endings'].split()[0]) > 3:
        return "Past Tense Forms"
    
    if 'ly_endings' in characteristics and int(characteristics['ly_endings'].split()[0]) > 3:
        return "Adverbial Forms"
    
    if 'short_tokens' in characteristics and int(characteristics['short_tokens'].split()[0]) > 6:
        return "Short Function Words"
    
    if 'capitalized' in characteristics and int(characteristics['capitalized'].split()[0]) > 5:
        return "Sentence Starters/Proper"
    
    # Check for mixed patterns
    char_count = len(characteristics)
    if char_count >= 4:
        return "Mixed Grammatical"
    elif char_count >= 2:
        # Get the two most prominent features
        features = list(characteristics.keys())[:2]
        return f"Mixed {features[0]}/{features[1]}"
    
    # Default based on token length
    avg_len = float(characteristics.get('avg_length', 0))
    if avg_len < 3:
        return "Short Tokens"
    elif avg_len > 6:
        return "Long Words"
    else:
        return "Medium-Length Words"


if __name__ == "__main__":
    analyze_cluster_differences()