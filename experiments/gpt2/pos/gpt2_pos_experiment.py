"""
GPT-2 Parts-of-Speech (POS) Clustering Experiment

Simple experiment to study how GPT-2 clusters different grammatical categories:
nouns, adjectives, adverbs, and verbs as single tokens.

This will show if GPT-2 organizes representations by grammatical category
across its 13 layers.

Reuses existing infrastructure: gpt2_activation_extractor.py, gpt2_pivot_clusterer.py, etc.
"""

from gpt2_activation_extractor import SimpleGPT2ActivationExtractor

def generate_pos_words():
    """Generate 30 examples each of nouns, adjectives, adverbs, and verbs."""
    
    # 30 common nouns
    nouns = [
        "cat", "dog", "house", "car", "book", "tree", "chair", "table", "phone", "computer",
        "water", "food", "school", "friend", "family", "music", "movie", "game", "park", "store",
        "doctor", "teacher", "student", "child", "parent", "brother", "sister", "city", "country", "world"
    ]
    
    # 30 common adjectives  
    adjectives = [
        "big", "small", "good", "bad", "happy", "sad", "fast", "slow", "hot", "cold",
        "new", "old", "young", "tall", "short", "long", "wide", "thin", "thick", "light",
        "dark", "bright", "quiet", "loud", "clean", "dirty", "easy", "hard", "soft", "rough"
    ]
    
    # 30 common adverbs
    adverbs = [
        "quickly", "slowly", "carefully", "loudly", "quietly", "easily", "hardly", "really", "very", "quite",
        "always", "never", "often", "sometimes", "usually", "rarely", "early", "late", "here", "there",
        "well", "badly", "clearly", "simply", "truly", "mostly", "nearly", "almost", "probably", "definitely"
    ]
    
    # 30 common verbs (base form)
    verbs = [
        "run", "walk", "eat", "drink", "sleep", "work", "play", "read", "write", "think",
        "talk", "listen", "see", "look", "hear", "feel", "touch", "smell", "taste", "move",
        "stop", "start", "finish", "begin", "come", "go", "take", "give", "get", "put"
    ]
    
    return nouns, adjectives, adverbs, verbs

def save_pos_words():
    """Save POS words to files for the experiment."""
    
    nouns, adjectives, adverbs, verbs = generate_pos_words()
    
    # Verify counts
    assert len(nouns) == 30, f"Expected 30 nouns, got {len(nouns)}"
    assert len(adjectives) == 30, f"Expected 30 adjectives, got {len(adjectives)}"
    assert len(adverbs) == 30, f"Expected 30 adverbs, got {len(adverbs)}"
    assert len(verbs) == 30, f"Expected 30 verbs, got {len(verbs)}"
    
    # Save individual files
    with open("gpt2_pos_nouns.txt", "w") as f:
        for noun in nouns:
            f.write(noun + "\n")
    
    with open("gpt2_pos_adjectives.txt", "w") as f:
        for adj in adjectives:
            f.write(adj + "\n")
    
    with open("gpt2_pos_adverbs.txt", "w") as f:
        for adv in adverbs:
            f.write(adv + "\n")
    
    with open("gpt2_pos_verbs.txt", "w") as f:
        for verb in verbs:
            f.write(verb + "\n")
    
    # Save combined file with labels
    all_words = []
    labels = []
    
    for noun in nouns:
        all_words.append(noun)
        labels.append("noun")
    
    for adj in adjectives:
        all_words.append(adj)
        labels.append("adjective")
    
    for adv in adverbs:
        all_words.append(adv)
        labels.append("adverb")
    
    for verb in verbs:
        all_words.append(verb)
        labels.append("verb")
    
    with open("gpt2_pos_all_words.txt", "w") as f:
        for word in all_words:
            f.write(word + "\n")
    
    # Save labels file
    with open("gpt2_pos_labels.txt", "w") as f:
        for label in labels:
            f.write(label + "\n")
    
    print(f"SUCCESS: Generated POS experiment data")
    print(f"- {len(nouns)} nouns")
    print(f"- {len(adjectives)} adjectives") 
    print(f"- {len(adverbs)} adverbs")
    print(f"- {len(verbs)} verbs")
    print(f"- Total: {len(all_words)} words")
    print()
    print("Files created:")
    print("- gpt2_pos_nouns.txt")
    print("- gpt2_pos_adjectives.txt") 
    print("- gpt2_pos_adverbs.txt")
    print("- gpt2_pos_verbs.txt")
    print("- gpt2_pos_all_words.txt")
    print("- gpt2_pos_labels.txt")
    
    return all_words, labels

def extract_pos_activations():
    """Extract activations for POS experiment using existing infrastructure."""
    
    # Generate and save word lists
    words, labels = save_pos_words()
    
    print("\nExtracting GPT-2 activations using existing infrastructure...")
    
    # Reuse existing extractor
    extractor = SimpleGPT2ActivationExtractor()
    activations = extractor.extract_activations(words)
    
    # Add POS labels to results
    activations['labels'] = labels
    activations['metadata']['experiment_type'] = 'pos_classification'
    activations['metadata']['pos_counts'] = {
        'noun': 30, 'adjective': 30, 'adverb': 30, 'verb': 30
    }
    
    # Save with POS-specific filename
    extractor.save_activations(activations, "gpt2_pos_activations.pkl")
    
    print("SUCCESS: POS activations extracted using existing infrastructure")
    return activations

if __name__ == "__main__":
    activations = extract_pos_activations()