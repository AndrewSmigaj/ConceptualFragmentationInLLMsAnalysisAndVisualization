"""
Minimal 5k experiment - just the essentials
"""
import json
import numpy as np
from pathlib import Path
import torch
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.cluster import KMeans
from collections import defaultdict

# Common English words (hardcoded for speed)
COMMON_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "is", "was", "are", "been", "has", "had", "were", "said", "did", "getting",
    "made", "find", "where", "much", "too", "very", "still", "being", "going", "why",
    "before", "never", "here", "more", "through", "again", "same", "under", "last", "right",
    "move", "thing", "tell", "does", "let", "help", "put", "different", "away", "turn",
    "hand", "place", "such", "high", "keep", "point", "child", "few", "small", "since",
    "against", "ask", "late", "home", "interest", "large", "off", "end", "open", "public",
    "follow", "during", "present", "without", "again", "hold", "member", "around", "every", "family",
    "leave", "feel", "fact", "group", "play", "stand", "increase", "early", "course", "change",
    "old", "great", "big", "long", "little", "own", "other", "old", "right", "big"
]

def main():
    output_dir = Path("experiments/gpt2/semantic_subtypes/5k_common_words")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing {len(COMMON_WORDS)} common words...")
    
    # Load model and tokenizer
    print("Loading GPT-2...")
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Filter to single tokens
    single_token_words = []
    for word in COMMON_WORDS:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 1:
            single_token_words.append((word, tokens[0]))
    
    print(f"Found {len(single_token_words)} single-token words")
    
    # Extract activations
    print("Extracting activations...")
    all_activations = []
    
    with torch.no_grad():
        for i, (word, token_id) in enumerate(single_token_words):
            if i % 20 == 0:
                print(f"  Processing word {i}/{len(single_token_words)}", end='\r')
            
            inputs = tokenizer(word, return_tensors='pt').to(device)
            outputs = model(**inputs, output_hidden_states=True)
            
            # Get activations from each layer
            word_acts = []
            for layer_idx in range(1, 13):  # Skip embedding layer
                act = outputs.hidden_states[layer_idx][0, 0, :].cpu().numpy()
                word_acts.append(act)
            
            all_activations.append(word_acts)
    
    activations = np.array(all_activations)  # [n_words, n_layers, hidden_dim]
    print(f"\nActivations shape: {activations.shape}")
    
    # Cluster each layer
    print("\nClustering layers...")
    k = 5
    cluster_assignments = {}
    
    for layer_idx in range(12):
        print(f"  Clustering layer {layer_idx}...", end='\r')
        layer_acts = activations[:, layer_idx, :]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
        labels = kmeans.fit_predict(layer_acts)
        cluster_assignments[layer_idx] = labels
    
    print("\nExtracting paths...")
    paths = []
    for word_idx in range(len(single_token_words)):
        path = []
        for layer_idx in range(12):
            cluster = cluster_assignments[layer_idx][word_idx]
            path.append(f"L{layer_idx}_C{cluster}")
        paths.append(path)
    
    # Analyze convergence
    print("\nAnalyzing path convergence...")
    
    # Group words by simple categories
    nouns = ["people", "year", "time", "way", "day", "thing", "child", "hand", "place", "home", "family", "fact", "group", "member"]
    verbs = ["be", "have", "do", "say", "get", "make", "go", "know", "take", "see", "come", "think", "look", "want", "give", "use", "find", "tell", "ask", "work", "feel", "leave", "play", "move", "help", "put", "turn", "hold", "follow", "stand", "increase", "change"]
    
    # Get indices
    word_list = [w for w, _ in single_token_words]
    noun_indices = [i for i, w in enumerate(word_list) if w in nouns]
    verb_indices = [i for i, w in enumerate(word_list) if w in verbs]
    
    # Check convergence in final layers
    for layer in [9, 10, 11]:
        print(f"\nLayer {layer}:")
        
        # Nouns
        if noun_indices:
            noun_clusters = [paths[i][layer] for i in noun_indices]
            unique_noun = len(set(noun_clusters))
            print(f"  Nouns: {len(noun_indices)} words -> {unique_noun} clusters")
        
        # Verbs
        if verb_indices:
            verb_clusters = [paths[i][layer] for i in verb_indices]
            unique_verb = len(set(verb_clusters))
            print(f"  Verbs: {len(verb_indices)} words -> {unique_verb} clusters")
    
    # Save results
    results = {
        'words': [w for w, _ in single_token_words],
        'paths': paths,
        'n_words': len(single_token_words),
        'n_layers': 12,
        'k': k
    }
    
    with open(output_dir / "minimal_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDone! Results saved to {output_dir}")

if __name__ == "__main__":
    main()