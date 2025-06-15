"""
GPT-2 Semantic Subtypes Expanded Word Lists

Expanded word lists with better grammatical balance, especially more verbs.
Target: ~1200 words total with more even distribution across grammatical categories.

Semantic Subtypes:
- Concrete nouns: Physical, tangible objects
- Abstract nouns: Concepts, ideas, emotions  
- Physical adjectives: Observable physical properties
- Emotive adjectives: Emotional and evaluative properties
- Manner adverbs: How actions are performed
- Degree adverbs: Intensity and extent modifiers
- Action verbs: Dynamic, observable actions
- Stative verbs: States, mental processes, conditions
"""

# Concrete Nouns (Physical, tangible objects) - Expanded to 150
CONCRETE_NOUNS = [
    # Animals (30)
    "cat", "dog", "bird", "fish", "horse", "cow", "pig", "sheep", "goat", "duck",
    "chicken", "rabbit", "mouse", "rat", "bear", "wolf", "fox", "deer", "lion", "tiger",
    "elephant", "monkey", "snake", "frog", "spider", "bee", "ant", "fly", "worm", "shark",
    
    # Household objects (40)
    "chair", "table", "bed", "sofa", "lamp", "book", "pen", "paper", "cup", "plate",
    "bowl", "fork", "knife", "spoon", "glass", "bottle", "box", "bag", "key", "lock",
    "door", "window", "wall", "floor", "roof", "stairs", "mirror", "clock", "phone", "computer",
    "desk", "shelf", "drawer", "cabinet", "rug", "curtain", "pillow", "blanket", "towel", "soap",
    
    # Vehicles & Transportation (20)
    "car", "truck", "bus", "train", "plane", "boat", "ship", "bike", "wheel", "engine",
    "road", "bridge", "track", "station", "airport", "garage", "gas", "oil", "tire", "seat",
    
    # Nature & Environment (30)
    "tree", "flower", "grass", "leaf", "rock", "stone", "water", "river", "lake", "ocean",
    "mountain", "hill", "valley", "forest", "field", "beach", "sand", "snow", "ice", "fire",
    "sun", "moon", "star", "cloud", "rain", "wind", "earth", "soil", "mud", "dirt",
    
    # Food & Kitchen (20)
    "bread", "meat", "cheese", "milk", "egg", "apple", "orange", "banana", "tomato", "potato",
    "rice", "pasta", "soup", "cake", "cookie", "candy", "sugar", "salt", "pepper", "butter",
    
    # Tools & Equipment (10)
    "hammer", "saw", "drill", "nail", "screw", "rope", "wire", "tool", "machine", "camera"
]

# Abstract Nouns (Concepts, ideas, emotions) - Expanded to 150
ABSTRACT_NOUNS = [
    # Emotions & Feelings (40)
    "love", "hate", "fear", "joy", "anger", "peace", "hope", "worry", "trust", "doubt",
    "shame", "pride", "guilt", "envy", "pity", "awe", "grief", "bliss", "rage", "calm",
    "stress", "relief", "shock", "thrill", "desire", "lust", "panic", "humor", "mood", "feeling",
    "emotion", "passion", "sympathy", "empathy", "contempt", "disgust", "surprise", "wonder", "curiosity", "boredom",
    
    # Abstract concepts & ideas (40)
    "freedom", "justice", "truth", "beauty", "wisdom", "power", "strength", "courage", "honor", "fame",
    "success", "failure", "luck", "fate", "chance", "risk", "change", "growth", "decline", "progress",
    "order", "chaos", "balance", "harmony", "conflict", "unity", "diversity", "equality", "fairness", "bias",
    "democracy", "liberty", "rights", "duty", "responsibility", "authority", "control", "influence", "impact", "effect",
    
    # Knowledge & Learning (35)
    "idea", "thought", "mind", "memory", "dream", "vision", "image", "concept", "theory", "fact",
    "data", "info", "news", "story", "tale", "myth", "legend", "history", "future", "past",
    "science", "art", "music", "poetry", "drama", "logic", "reason", "wisdom", "knowledge", "ignorance",
    "education", "learning", "teaching", "lesson", "skill",
    
    # Social & Cultural (35)
    "culture", "society", "law", "rule", "custom", "habit", "trend", "fashion", "style", "class",
    "status", "role", "duty", "right", "wrong", "good", "evil", "virtue", "vice", "sin",
    "crime", "punishment", "reward", "gift", "prize", "marriage", "divorce", "family", "friendship", "relationship",
    "community", "tradition", "heritage", "legacy", "reputation"
]

# Physical Adjectives (Observable physical properties) - Keep at 150
PHYSICAL_ADJECTIVES = [
    # Size & Dimension (40)
    "big", "small", "large", "tiny", "huge", "giant", "mini", "vast", "wide", "narrow", 
    "thick", "thin", "tall", "short", "long", "brief", "deep", "shallow", "high", "low",
    "broad", "slim", "fat", "skinny", "lean", "bulk", "dense", "light", "heavy", "solid",
    "great", "grand", "mega", "super", "ultra", "micro", "nano", "macro", "maxi", "jumbo",
    
    # Shape & Form (30)
    "round", "square", "flat", "curved", "straight", "bent", "sharp", "dull", "pointed", "blunt",
    "smooth", "rough", "bumpy", "even", "uneven", "level", "sloped", "steep", "gentle", "jagged",
    "hollow", "full", "empty", "packed", "loose", "tight", "slack", "tense", "rigid", "flexible",
    
    # Temperature & Texture (40)
    "hot", "cold", "warm", "cool", "frozen", "boiling", "mild", "harsh", "soft", "hard",
    "wet", "dry", "damp", "moist", "soaked", "dried", "sticky", "slippery", "grainy", "silky",
    "furry", "hairy", "bald", "fuzzy", "crisp", "mushy", "firm", "tender", "tough", "fragile",
    "burning", "freezing", "chilly", "cozy", "humid", "arid", "parched", "saturated", "brittle", "elastic",
    
    # Color & Appearance (40)
    "red", "blue", "green", "yellow", "black", "white", "brown", "pink", "purple", "orange",
    "gray", "dark", "light", "bright", "dim", "clear", "cloudy", "shiny", "dull", "glossy",
    "clean", "dirty", "neat", "messy", "fresh", "stale", "new", "old", "young", "aged",
    "transparent", "opaque", "translucent", "vivid", "pale", "faded", "radiant", "murky", "pristine", "tarnished"
]

# Emotive Adjectives (Emotional and evaluative properties) - Keep at 150
EMOTIVE_ADJECTIVES = [
    # Positive emotions & evaluations (40)
    "happy", "sad", "good", "bad", "great", "awful", "nice", "mean", "kind", "cruel",
    "gentle", "harsh", "sweet", "bitter", "pleasant", "nasty", "lovely", "ugly", "pretty", "gross",
    "beautiful", "hideous", "cute", "scary", "funny", "boring", "exciting", "dull", "amazing", "terrible",
    "wonderful", "horrible", "delightful", "disgusting", "charming", "repulsive", "attractive", "repellent", "appealing", "revolting",
    
    # Personality & Character (40)
    "smart", "dumb", "clever", "stupid", "wise", "foolish", "brave", "timid", "bold", "shy",
    "confident", "nervous", "calm", "anxious", "relaxed", "tense", "patient", "restless", "careful", "reckless",
    "honest", "fake", "real", "false", "true", "loyal", "selfish", "generous", "greedy", "modest",
    "humble", "proud", "vain", "sincere", "deceitful", "trustworthy", "unreliable", "responsible", "careless", "diligent",
    
    # Social & Interpersonal (35)
    "friendly", "hostile", "polite", "rude", "humble", "proud", "arrogant", "modest", "boastful", "quiet",
    "loud", "gentle", "violent", "peaceful", "wild", "tame", "fierce", "mild", "aggressive", "passive",
    "social", "shy", "outgoing", "reserved", "popular", "lonely", "loved", "hated", "respected", "ignored",
    "admired", "despised", "cherished", "neglected", "appreciated",
    
    # Mental & Emotional States (35)
    "sane", "crazy", "rational", "mad", "logical", "absurd", "sensible", "silly", "serious", "playful",
    "mature", "childish", "adult", "naive", "cynical", "hopeful", "gloomy", "cheerful", "moody", "stable",
    "weird", "normal", "strange", "ordinary", "special", "common", "rare", "unique", "typical", "odd",
    "confused", "clear", "certain", "doubtful", "confident"
]

# Manner Adverbs (How actions are performed) - Expanded to 200
MANNER_ADVERBS = [
    # Speed & Movement (40)
    "fast", "slow", "quickly", "slowly", "rapidly", "gradually", "suddenly", "steadily", "swiftly", "leisurely",
    "hastily", "urgently", "promptly", "immediately", "eventually", "instantly", "briskly", "sluggishly", "speedily", "tardily",
    "abruptly", "smoothly", "jerkily", "gracefully", "clumsily", "awkwardly", "skillfully", "expertly", "professionally", "amateurishly",
    "efficiently", "inefficiently", "productively", "lazily", "actively", "passively", "energetically", "lethargically", "vigorously", "weakly",
    
    # Care & Precision (40)
    "carefully", "carelessly", "neatly", "messily", "precisely", "loosely", "exactly", "roughly", "clearly", "vaguely",
    "specifically", "generally", "directly", "indirectly", "openly", "secretly", "honestly", "falsely", "truly", "wrongly",
    "accurately", "inaccurately", "correctly", "incorrectly", "properly", "improperly", "appropriately", "inappropriately", "suitably", "unsuitably",
    "perfectly", "imperfectly", "completely", "partially", "thoroughly", "superficially", "meticulously", "sloppily", "systematically", "randomly",
    
    # Emotion & Attitude (40)
    "happily", "sadly", "angrily", "calmly", "nervously", "bravely", "fearfully", "proudly", "humbly", "kindly",
    "cruelly", "sweetly", "bitterly", "warmly", "coldly", "lovingly", "hatefully", "hopefully", "desperately", "confidently",
    "anxiously", "peacefully", "violently", "gently", "harshly", "tenderly", "roughly", "softly", "loudly", "quietly",
    "cheerfully", "gloomily", "optimistically", "pessimistically", "enthusiastically", "reluctantly", "eagerly", "hesitantly", "willingly", "unwillingly",
    
    # Style & Method (40)
    "naturally", "artificially", "casually", "formally", "simply", "complexly", "creatively", "mechanically", "manually", "automatically",
    "voluntarily", "forcibly", "freely", "strictly", "flexibly", "rigidly", "spontaneously", "deliberately", "intentionally", "accidentally",
    "purposefully", "aimlessly", "methodically", "chaotically", "orderly", "disorderly", "neatly", "messily", "elegantly", "plainly",
    "beautifully", "uglily", "graciously", "rudely", "politely", "impolitely", "respectfully", "disrespectfully", "courteously", "discourteously",
    
    # Frequency & Continuity (40)
    "often", "rarely", "always", "never", "sometimes", "usually", "frequently", "occasionally", "regularly", "irregularly",
    "constantly", "intermittently", "continuously", "sporadically", "repeatedly", "once", "twice", "daily", "weekly", "monthly",
    "yearly", "hourly", "momentarily", "temporarily", "permanently", "briefly", "lengthily", "endlessly", "finitely", "eternally",
    "perpetually", "periodically", "cyclically", "seasonally", "annually", "quarterly", "biweekly", "fortnightly", "nightly", "morning"
]

# Degree Adverbs (Intensity and extent modifiers) - Keep at 150
DEGREE_ADVERBS = [
    # High intensity (40)
    "very", "so", "too", "quite", "really", "truly", "super", "extra", "ultra", "mega",
    "all", "fully", "totally", "completely", "entirely", "perfectly", "utterly", "thoroughly", "deeply", "highly",
    "extremely", "exceptionally", "extraordinarily", "remarkably", "incredibly", "amazingly", "astonishingly", "overwhelmingly", "tremendously", "enormously",
    "vastly", "hugely", "massively", "immensely", "intensely", "profoundly", "supremely", "ultimately", "absolutely", "positively",
    
    # Medium intensity (35)
    "fairly", "pretty", "somewhat", "relatively", "moderately", "reasonably", "considerably", "significantly", "notably", "substantially",
    "largely", "mostly", "mainly", "primarily", "chiefly", "generally", "typically", "normally", "usually", "commonly",
    "adequately", "sufficiently", "decently", "acceptably", "tolerably", "passably", "satisfactorily", "averagely", "ordinarily", "standardly",
    "regularly", "routinely", "habitually", "customarily", "traditionally",
    
    # Low intensity (35)
    "slightly", "barely", "hardly", "scarcely", "minimally", "marginally", "nominally", "superficially", "partially", "partly",
    "somewhat", "rather", "mildly", "gently", "softly", "lightly", "faintly", "weakly", "dimly", "vaguely",
    "loosely", "roughly", "approximately", "nearly", "almost", "virtually", "practically", "essentially", "basically", "fundamentally",
    "remotely", "distantly", "tangentially", "peripherally", "indirectly",
    
    # Comparison & Certainty (40)
    "more", "less", "most", "least", "better", "worse", "best", "worst", "greater", "smaller",
    "higher", "lower", "stronger", "weaker", "faster", "slower", "bigger", "tinier", "longer", "shorter",
    "definitely", "certainly", "surely", "probably", "possibly", "maybe", "perhaps", "likely", "unlikely", "obviously",
    "clearly", "apparently", "evidently", "presumably", "supposedly", "allegedly", "reportedly", "seemingly", "virtually", "practically"
]

# Action Verbs (Dynamic, observable actions) - Expanded to 200
ACTION_VERBS = [
    # Physical movement (50)
    "run", "walk", "jump", "hop", "skip", "dance", "climb", "crawl", "swim", "fly",
    "drive", "ride", "travel", "move", "go", "come", "leave", "arrive", "enter", "exit",
    "push", "pull", "lift", "drop", "throw", "catch", "kick", "hit", "punch", "slap",
    "slide", "glide", "roll", "spin", "turn", "twist", "bend", "stretch", "reach", "grab",
    "sprint", "jog", "leap", "bounce", "stumble", "fall", "rise", "stand", "sit", "kneel",
    
    # Manual actions (50)
    "eat", "drink", "chew", "swallow", "bite", "taste", "smell", "touch", "feel", "hold",
    "carry", "pick", "place", "put", "set", "lay", "open", "close", "lock", "unlock",
    "cut", "slice", "chop", "tear", "rip", "break", "fix", "repair", "build", "destroy",
    "paint", "draw", "write", "erase", "type", "print", "copy", "paste", "delete", "save",
    "wash", "clean", "scrub", "wipe", "dust", "sweep", "mop", "vacuum", "polish", "shine",
    
    # Communication actions (40)
    "speak", "talk", "say", "tell", "ask", "answer", "call", "shout", "whisper", "sing",
    "read", "listen", "hear", "watch", "see", "look", "observe", "notice", "ignore", "focus",
    "explain", "describe", "discuss", "argue", "debate", "convince", "persuade", "inform", "announce", "declare",
    "chat", "converse", "communicate", "express", "convey", "signal", "gesture", "nod", "wave", "point",
    
    # Work & Creation (30)
    "work", "labor", "toil", "strive", "effort", "create", "produce", "design", "plan", "organize",
    "arrange", "prepare", "cook", "bake", "brew", "mix", "blend", "combine", "assemble", "construct",
    "manufacture", "fabricate", "forge", "craft", "sculpt", "mold", "shape", "form", "fashion", "style",
    
    # Social & Interactive (30)
    "meet", "greet", "visit", "invite", "welcome", "join", "leave", "follow", "lead", "guide",
    "help", "assist", "support", "protect", "defend", "attack", "fight", "compete", "cooperate", "collaborate",
    "share", "give", "take", "lend", "borrow", "trade", "exchange", "buy", "sell", "pay"
]

# Stative Verbs (States, mental processes, conditions) - Expanded to 200
STATIVE_VERBS = [
    # Mental states & processes (50)
    "think", "know", "understand", "realize", "recognize", "remember", "forget", "learn", "believe", "doubt",
    "wonder", "imagine", "dream", "hope", "wish", "want", "need", "desire", "prefer", "choose",
    "decide", "consider", "suppose", "assume", "guess", "expect", "predict", "fear", "worry", "trust",
    "comprehend", "grasp", "perceive", "conceive", "envision", "visualize", "contemplate", "meditate", "reflect", "ponder",
    "analyze", "evaluate", "assess", "judge", "estimate", "calculate", "determine", "conclude", "deduce", "infer",
    
    # Emotional & psychological states (50)
    "love", "hate", "like", "dislike", "enjoy", "mind", "care", "matter", "concern", "bother",
    "please", "annoy", "frustrate", "satisfy", "disappoint", "surprise", "amaze", "shock", "scare", "comfort",
    "admire", "respect", "despise", "loathe", "adore", "cherish", "treasure", "value", "appreciate", "resent",
    "envy", "pity", "sympathize", "empathize", "relate", "connect", "bond", "attach", "detach", "alienate",
    "fascinate", "intrigue", "interest", "bore", "excite", "thrill", "delight", "disgust", "repel", "attract",
    
    # Being & Existence (40)
    "be", "exist", "live", "die", "survive", "remain", "stay", "become", "seem", "appear",
    "look", "sound", "feel", "taste", "smell", "weigh", "measure", "cost", "worth", "equal",
    "belong", "own", "have", "possess", "contain", "include", "consist", "comprise", "involve", "require",
    "lack", "miss", "need", "want", "deserve", "merit", "warrant", "justify", "qualify", "suffice",
    
    # Relationships & States (30)
    "relate", "connect", "link", "associate", "compare", "contrast", "differ", "vary", "change", "match",
    "fit", "suit", "belong", "depend", "rely", "base", "rest", "lie", "stand", "sit",
    "mean", "signify", "represent", "symbolize", "indicate", "suggest", "imply", "express", "convey", "denote",
    
    # Abilities & Conditions (30)
    "can", "could", "may", "might", "should", "would", "must", "ought", "dare", "need",
    "enable", "allow", "permit", "forbid", "prevent", "hinder", "facilitate", "promote", "encourage", "discourage",
    "qualify", "disqualify", "entitle", "empower", "authorize", "license", "certify", "validate", "verify", "confirm"
]

# Combine all word lists for easy access
ALL_WORD_LISTS = {
    "concrete_nouns": CONCRETE_NOUNS,
    "abstract_nouns": ABSTRACT_NOUNS,
    "physical_adjectives": PHYSICAL_ADJECTIVES,
    "emotive_adjectives": EMOTIVE_ADJECTIVES,
    "manner_adverbs": MANNER_ADVERBS,
    "degree_adverbs": DEGREE_ADVERBS,
    "action_verbs": ACTION_VERBS,
    "stative_verbs": STATIVE_VERBS
}

def get_word_counts():
    """Get word count statistics for all subtypes."""
    counts = {subtype: len(words) for subtype, words in ALL_WORD_LISTS.items()}
    
    # Calculate grammatical totals
    nouns = counts['concrete_nouns'] + counts['abstract_nouns']
    adjectives = counts['physical_adjectives'] + counts['emotive_adjectives']
    adverbs = counts['manner_adverbs'] + counts['degree_adverbs']
    verbs = counts['action_verbs'] + counts['stative_verbs']
    total = nouns + adjectives + adverbs + verbs
    
    print("\n=== Expanded Word List Statistics ===")
    print(f"\nBy semantic subtype:")
    for subtype, count in counts.items():
        print(f"  {subtype}: {count}")
    
    print(f"\nBy grammatical category:")
    print(f"  Nouns: {nouns} ({nouns/total*100:.1f}%)")
    print(f"  Adjectives: {adjectives} ({adjectives/total*100:.1f}%)")
    print(f"  Adverbs: {adverbs} ({adverbs/total*100:.1f}%)")
    print(f"  Verbs: {verbs} ({verbs/total*100:.1f}%)")
    print(f"\nTotal words: {total}")
    
    return counts

if __name__ == "__main__":
    get_word_counts()