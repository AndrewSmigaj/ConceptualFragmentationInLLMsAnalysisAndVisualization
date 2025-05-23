"""
Generate 3-token sentences for GPT-2 APA Pivot Experiment.

Pattern: <positive word> but <negative/positive word>
100 contrast class, 100 consistent class
Activations collected after third token.
"""

def generate_contrast_sentences():
    """Generate positive-but-negative 3-token sentences."""
    return [
        "good but bad",
        "great but awful",
        "nice but terrible",
        "excellent but horrible",
        "wonderful but disgusting",
        "amazing but dreadful",
        "fantastic but appalling",
        "brilliant but atrocious",
        "perfect but ghastly",
        "superb but revolting",
        "outstanding but vile",
        "magnificent but repulsive",
        "marvelous but hideous",
        "splendid but loathsome",
        "fabulous but detestable",
        "terrific but abominable",
        "incredible but execrable",
        "remarkable but odious",
        "exceptional but repugnant",
        "phenomenal but nauseating",
        "impressive but sickening",
        "admirable but disturbing",
        "delightful but offensive",
        "charming but repellent",
        "appealing but disagreeable",
        "attractive but unpleasant",
        "pleasant but nasty",
        "enjoyable but harsh",
        "satisfying but bitter",
        "refreshing but sour",
        "comfortable but uncomfortable",
        "relaxing but stressful",
        "peaceful but chaotic",
        "calm but frantic",
        "serene but turbulent",
        "tranquil but agitated",
        "soothing but irritating",
        "gentle but rough",
        "soft but hard",
        "smooth but jagged",
        "elegant but crude",
        "graceful but clumsy",
        "stylish but tacky",
        "chic but gaudy",
        "sophisticated but vulgar",
        "refined but coarse",
        "polished but shabby",
        "pristine but dirty",
        "clean but filthy",
        "fresh but stale",
        # Additional 50 contrast sentences
        "beautiful but ugly",
        "lovely but hideous",
        "gorgeous but revolting",
        "stunning but repulsive",
        "handsome but disgusting",
        "pretty but ghastly",
        "adorable but awful",
        "cute but horrible",
        "sweet but bitter",
        "kind but cruel",
        "gentle but violent",
        "warm but cold",
        "bright but dark",
        "light but heavy",
        "fast but slow",
        "quick but sluggish",
        "sharp but dull",
        "clear but murky",
        "pure but contaminated",
        "clean but polluted",
        "healthy but sick",
        "strong but weak",
        "powerful but feeble",
        "robust but fragile",
        "solid but flimsy",
        "stable but unstable",
        "secure but vulnerable",
        "safe but dangerous",
        "easy but difficult",
        "simple but complex",
        "plain but ornate",
        "humble but arrogant",
        "modest but boastful",
        "quiet but noisy",
        "silent but loud",
        "peaceful but violent",
        "calm but agitated",
        "still but restless",
        "steady but erratic",
        "consistent but inconsistent",
        "reliable but unreliable",
        "trustworthy but deceptive",
        "honest but dishonest",
        "truthful but false",
        "genuine but fake",
        "real but artificial",
        "natural but synthetic",
        "organic but processed",
        "fresh but rotten",
        "ripe but spoiled",
        "new but old"
    ]

def generate_consistent_sentences():
    """Generate positive-but-positive 3-token sentences."""
    return [
        "good but great",
        "nice but wonderful",
        "fine but excellent",
        "decent but amazing",
        "okay but fantastic",
        "solid but brilliant",
        "fair but perfect",
        "adequate but superb",
        "reasonable but outstanding",
        "acceptable but magnificent",
        "satisfactory but marvelous",
        "tolerable but splendid",
        "passable but fabulous",
        "suitable but terrific",
        "appropriate but incredible",
        "proper but remarkable",
        "correct but exceptional",
        "right but phenomenal",
        "valid but impressive",
        "sound but admirable",
        "healthy but delightful",
        "strong but charming",
        "sturdy but appealing",
        "robust but attractive",
        "stable but pleasant",
        "secure but enjoyable",
        "safe but satisfying",
        "reliable but refreshing",
        "dependable but comfortable",
        "trustworthy but relaxing",
        "honest but peaceful",
        "genuine but calm",
        "authentic but serene",
        "real but tranquil",
        "true but soothing",
        "actual but gentle",
        "legitimate but soft",
        "valid but smooth",
        "justified but elegant",
        "warranted but graceful",
        "deserved but stylish",
        "earned but chic",
        "merited but sophisticated",
        "worthy but refined",
        "valuable but polished",
        "useful but pristine",
        "helpful but clean",
        "beneficial but fresh",
        "positive but uplifting",
        "bright but radiant",
        # Additional 50 consistent sentences
        "beautiful but gorgeous",
        "lovely but stunning",
        "pretty but beautiful",
        "handsome but attractive",
        "cute but adorable",
        "sweet but delightful",
        "kind but compassionate",
        "gentle but tender",
        "warm but cozy",
        "bright but brilliant",
        "light but luminous",
        "fast but rapid",
        "quick but swift",
        "sharp but keen",
        "clear but transparent",
        "pure but pristine",
        "clean but spotless",
        "healthy but vigorous",
        "strong but powerful",
        "robust but sturdy",
        "solid but firm",
        "stable but steady",
        "secure but protected",
        "safe but sheltered",
        "easy but simple",
        "plain but modest",
        "humble but unassuming",
        "quiet but peaceful",
        "silent but tranquil",
        "calm but serene",
        "still but motionless",
        "steady but consistent",
        "reliable but dependable",
        "trustworthy but faithful",
        "honest but truthful",
        "genuine but authentic",
        "real but actual",
        "natural but organic",
        "fresh but crisp",
        "ripe but mature",
        "new but modern",
        "smart but intelligent",
        "wise but knowledgeable",
        "clever but ingenious",
        "talented but gifted",
        "skilled but expert",
        "able but capable",
        "competent but proficient",
        "efficient but effective",
        "productive but fruitful",
        "successful but triumphant"
    ]

def save_sentences():
    """Save all sentences to files."""
    contrast = generate_contrast_sentences()
    consistent = generate_consistent_sentences()
    
    # Save contrast sentences
    with open("gpt2_pivot_contrast_sentences.txt", "w") as f:
        for sentence in contrast:
            f.write(sentence + "\n")
    
    # Save consistent sentences  
    with open("gpt2_pivot_consistent_sentences.txt", "w") as f:
        for sentence in consistent:
            f.write(sentence + "\n")
            
    # Save combined
    with open("gpt2_pivot_all_sentences.txt", "w") as f:
        f.write("# CONTRAST CLASS (positive but negative)\n")
        for sentence in contrast:
            f.write(sentence + "\n")
        f.write("\n# CONSISTENT CLASS (positive but positive)\n")
        for sentence in consistent:
            f.write(sentence + "\n")
    
    print(f"Generated {len(contrast)} contrast sentences")
    print(f"Generated {len(consistent)} consistent sentences")
    print(f"Total: {len(contrast) + len(consistent)} sentences")
    
    return contrast, consistent

if __name__ == "__main__":
    contrast, consistent = save_sentences()