"""
GPT-2 5000 Most Common Words Experiment with WordNet Enrichment

Uses existing infrastructure to analyze the 5000 most common English words,
enriched with WordNet annotations for deeper linguistic analysis.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
import requests

# NLP libraries
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Add parent directory to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Use existing infrastructure
try:
    from experiments.gpt2.shared.gpt2_token_validator import GPT2TokenValidator
except ImportError:
    print("Warning: Could not import GPT2TokenValidator, implementing basic version")
    GPT2TokenValidator = None


@dataclass
class EnrichedWord:
    """Word enriched with linguistic annotations"""
    word: str
    token_id: int
    rank: int  # Frequency rank (1 = most common)
    
    # Multiple POS tags with frequencies
    pos_tags: Dict[str, float] = field(default_factory=dict)  # POS -> frequency
    primary_pos: str = ""
    secondary_pos: Optional[str] = None
    
    # WordNet features
    synsets: List[str] = field(default_factory=list)
    polysemy_count: int = 0
    has_multiple_pos: bool = False
    
    # Semantic features
    hypernym_chains: Dict[str, List[str]] = field(default_factory=dict)  # synset -> chain
    semantic_domains: List[str] = field(default_factory=list)
    
    # Morphological features
    is_inflected: bool = False
    base_form: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'word': self.word,
            'token_id': self.token_id,
            'rank': self.rank,
            'pos_tags': self.pos_tags,
            'primary_pos': self.primary_pos,
            'secondary_pos': self.secondary_pos,
            'synsets': self.synsets,
            'polysemy_count': self.polysemy_count,
            'has_multiple_pos': self.has_multiple_pos,
            'hypernym_chains': self.hypernym_chains,
            'semantic_domains': self.semantic_domains,
            'is_inflected': self.is_inflected,
            'base_form': self.base_form
        }


class CommonWordExtractor:
    """Extract and enrich the most common English words"""
    
    def __init__(self):
        self.validator = GPT2TokenValidator()
        
    def get_frequency_list(self, n: int = 5000) -> List[Tuple[str, int]]:
        """Get n most common words from multiple sources"""
        
        # Try to use Google's 10,000 most common words list
        try:
            # This is a well-known public dataset
            url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
            response = requests.get(url)
            if response.status_code == 200:
                words = response.text.strip().split('\n')
                return [(word, i+1) for i, word in enumerate(words[:n*2])]  # Get extra for filtering
        except:
            pass
        
        # Fallback: Use Brown corpus
        print("Using Brown corpus for frequency data...")
        word_freq = Counter()
        
        for word in brown.words():
            word_lower = word.lower()
            if word_lower.isalpha():
                word_freq[word_lower] += 1
        
        # Get most common
        return word_freq.most_common(n * 2)  # Get extra for filtering
    
    def get_pos_distribution(self, word: str) -> Dict[str, float]:
        """Get POS tag distribution from Brown corpus"""
        pos_counts = Counter()
        total = 0
        
        # Count POS tags in Brown corpus
        for sent in brown.tagged_sents():
            for token, pos in sent:
                if token.lower() == word:
                    # Simplify POS tag to major category
                    if pos.startswith('NN'):
                        pos_counts['n'] += 1
                    elif pos.startswith('VB'):
                        pos_counts['v'] += 1
                    elif pos.startswith('JJ'):
                        pos_counts['a'] += 1
                    elif pos.startswith('RB'):
                        pos_counts['r'] += 1
                    else:
                        pos_counts['other'] += 1
                    total += 1
        
        # Convert to frequencies
        if total > 0:
            return {pos: count/total for pos, count in pos_counts.items()}
        return {}
    
    def enrich_word(self, word: str, rank: int) -> Optional[EnrichedWord]:
        """Enrich a word with linguistic annotations"""
        
        # Validate it's a single GPT-2 token
        if self.validator:
            validation = self.validator.validate_word(word)
            if not validation.is_single_token:
                return None
            token_id = validation.token_ids[0]
        else:
            # Basic validation without GPT2TokenValidator
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokens = tokenizer.encode(word, add_special_tokens=False)
            if len(tokens) != 1:
                return None
            token_id = tokens[0]
        
        enriched = EnrichedWord(
            word=word,
            token_id=token_id,
            rank=rank
        )
        
        # Get POS distribution
        enriched.pos_tags = self.get_pos_distribution(word)
        if enriched.pos_tags:
            # Primary POS is most frequent
            sorted_pos = sorted(enriched.pos_tags.items(), key=lambda x: x[1], reverse=True)
            enriched.primary_pos = sorted_pos[0][0]
            if len(sorted_pos) > 1 and sorted_pos[1][1] > 0.2:  # Secondary if >20%
                enriched.secondary_pos = sorted_pos[1][0]
            enriched.has_multiple_pos = len([p for p, f in enriched.pos_tags.items() if f > 0.1]) > 1
        
        # Get WordNet synsets
        synsets = wn.synsets(word)
        enriched.synsets = [s.name() for s in synsets]
        enriched.polysemy_count = len(synsets)
        
        # Extract hypernym chains for each synset
        for synset in synsets[:3]:  # Limit to top 3 senses
            chain = []
            current = synset
            while current.hypernyms():
                current = current.hypernyms()[0]
                chain.append(current.name())
            if chain:
                enriched.hypernym_chains[synset.name()] = chain
                # Semantic domain is the top-level hypernym
                if chain[-1] not in enriched.semantic_domains:
                    enriched.semantic_domains.append(chain[-1])
        
        # Check if inflected
        if synsets:
            lemmas = synsets[0].lemmas()
            if lemmas:
                base = lemmas[0].name()
                if base != word:
                    enriched.is_inflected = True
                    enriched.base_form = base
        
        return enriched
    
    def extract_common_words(self, n: int = 5000) -> List[EnrichedWord]:
        """Extract and enrich the n most common single-token words"""
        
        print(f"Extracting {n} most common single-token words...")
        
        # Get frequency list
        freq_list = self.get_frequency_list(n)
        
        enriched_words = []
        for word, rank in freq_list:
            if len(enriched_words) >= n:
                break
                
            enriched = self.enrich_word(word, len(enriched_words) + 1)
            if enriched:
                enriched_words.append(enriched)
                
                if len(enriched_words) % 500 == 0:
                    print(f"Processed {len(enriched_words)} words...")
        
        print(f"Successfully extracted {len(enriched_words)} single-token words")
        
        # Print statistics
        self._print_statistics(enriched_words)
        
        return enriched_words
    
    def _print_statistics(self, words: List[EnrichedWord]):
        """Print dataset statistics"""
        pos_counts = Counter()
        for word in words:
            if word.primary_pos:
                pos_counts[word.primary_pos] += 1
        
        print("\n=== Dataset Statistics ===")
        print(f"Total words: {len(words)}")
        print(f"POS distribution:")
        for pos, count in pos_counts.most_common():
            print(f"  {pos}: {count} ({count/len(words)*100:.1f}%)")
        
        print(f"\nPolysemy statistics:")
        polysemy_counts = [w.polysemy_count for w in words]
        print(f"  Average senses per word: {np.mean(polysemy_counts):.2f}")
        print(f"  Max senses: {max(polysemy_counts)}")
        print(f"  Words with multiple POS: {sum(1 for w in words if w.has_multiple_pos)}")


def run_5k_experiment():
    """Run the 5000 common words experiment"""
    
    output_dir = Path("experiments/gpt2/semantic_subtypes/5k_common_words")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract common words
    extractor = CommonWordExtractor()
    enriched_words = extractor.extract_common_words(5000)
    
    # Save enriched words
    with open(output_dir / "enriched_words.json", 'w') as f:
        json.dump([w.to_dict() for w in enriched_words], f, indent=2)
    
    # Extract activations with batching for efficiency
    print("\nExtracting GPT-2 activations...")
    import torch
    from transformers import GPT2Model, GPT2Tokenizer
    
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Extract activations in batches
    all_activations = []
    batch_size = 32
    
    with torch.no_grad():
        for batch_start in range(0, len(enriched_words), batch_size):
            batch_end = min(batch_start + batch_size, len(enriched_words))
            batch_words = enriched_words[batch_start:batch_end]
            
            if batch_start % (batch_size * 10) == 0:
                print(f"Processing words {batch_start}-{batch_end}/{len(enriched_words)}...")
            
            # Process each word in batch (can't batch different words due to tokenization)
            for word in batch_words:
                # Tokenize
                inputs = tokenizer(word.word, return_tensors='pt').to(device)
                
                # Get hidden states from all layers
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of (n_layers+1) x [batch, seq, hidden]
                
                # Extract activations for the single token (ignore embedding layer)
                word_activations = []
                for layer_idx in range(1, 13):  # Layers 0-11 (skip embedding)
                    layer_act = hidden_states[layer_idx][0, 0, :].cpu().numpy()  # [hidden_dim]
                    word_activations.append(layer_act)
                
                all_activations.append(np.array(word_activations))
    
    activations = np.array(all_activations)  # [n_words, n_layers, hidden_dim]
    print(f"Extracted activations shape: {activations.shape}")
    
    # Save activations
    np.save(output_dir / "activations.npy", activations)
    
    # Save metadata
    metadata = {
        'n_words': len(words),
        'n_layers': 12,
        'model': 'gpt2',
        'extraction_time': datetime.now().isoformat(),
        'pos_distribution': dict(Counter(w.primary_pos for w in enriched_words if w.primary_pos))
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Run clustering and path analysis
    print("\nRunning clustering and path analysis...")
    
    # Prepare data in expected format
    layer_activations = {}
    for layer_idx in range(12):
        layer_activations[f'layer_{layer_idx}'] = activations[:, layer_idx, :]
    
    # Run clustering for each layer
    from sklearn.cluster import KMeans
    cluster_results = {}
    
    for k in [3, 5, 8, 10]:
        print(f"\nClustering with k={k}...")
        layer_clusters = {}
        
        for layer_idx in range(12):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(activations[:, layer_idx, :])
            layer_clusters[f'layer_{layer_idx}'] = labels
        
        cluster_results[f'k_{k}'] = layer_clusters
    
    # Extract paths for each word
    path_results = {}
    for k_config, layer_clusters in cluster_results.items():
        paths = []
        for word_idx in range(len(enriched_words)):
            path = []
            for layer_idx in range(12):
                cluster_id = layer_clusters[f'layer_{layer_idx}'][word_idx]
                path.append(f'L{layer_idx}_C{cluster_id}')
            paths.append(path)
        path_results[k_config] = paths
    
    results = {
        'cluster_results': cluster_results,
        'path_results': path_results,
        'n_words': len(enriched_words),
        'n_layers': 12
    }
    
    # Save results
    with open(output_dir / "path_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analyze grammatical vs semantic organization
    print("\nAnalyzing organization patterns...")
    organization_results = analyze_organization_patterns(enriched_words, results)
    
    with open(output_dir / "organization_analysis.json", 'w') as f:
        json.dump(organization_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


def analyze_organization_patterns(words: List[EnrichedWord], path_results: Dict) -> Dict:
    """Analyze whether organization is grammatical or semantic"""
    
    # Group words by various criteria
    pos_groups = defaultdict(list)
    domain_groups = defaultdict(list)
    polysemy_groups = defaultdict(list)
    
    for i, word in enumerate(words):
        # By POS
        if word.primary_pos:
            pos_groups[word.primary_pos].append(i)
        
        # By semantic domain
        for domain in word.semantic_domains[:1]:  # Primary domain only
            domain_groups[domain].append(i)
        
        # By polysemy level
        if word.polysemy_count == 1:
            polysemy_groups['monosemous'].append(i)
        elif word.polysemy_count <= 3:
            polysemy_groups['low_polysemy'].append(i)
        else:
            polysemy_groups['high_polysemy'].append(i)
    
    # Analyze trajectory convergence for each grouping
    results = {
        'grammatical_organization': {},
        'semantic_organization': {},
        'polysemy_effects': {}
    }
    
    # Analyze POS-based organization
    for pos, indices in pos_groups.items():
        if len(indices) >= 20:  # Minimum group size
            results['grammatical_organization'][pos] = {
                'count': len(indices),
                'trajectory_convergence': calculate_convergence(indices, path_results),
                'dominant_paths': get_dominant_paths(indices, path_results)
            }
    
    # Analyze semantic domain organization  
    for domain, indices in domain_groups.items():
        if len(indices) >= 10:
            results['semantic_organization'][domain] = {
                'count': len(indices),
                'trajectory_convergence': calculate_convergence(indices, path_results),
                'dominant_paths': get_dominant_paths(indices, path_results)
            }
    
    # Analyze polysemy effects
    for level, indices in polysemy_groups.items():
        results['polysemy_effects'][level] = {
            'count': len(indices),
            'trajectory_convergence': calculate_convergence(indices, path_results),
            'avg_path_diversity': calculate_path_diversity(indices, path_results)
        }
    
    return results


def calculate_convergence(indices: List[int], path_results: Dict) -> float:
    """Calculate how much trajectories converge in late layers"""
    # Placeholder - implement based on path_results structure
    return 0.0


def get_dominant_paths(indices: List[int], path_results: Dict) -> List[str]:
    """Get the most common paths for a group"""
    # Placeholder - implement based on path_results structure  
    return []


def calculate_path_diversity(indices: List[int], path_results: Dict) -> float:
    """Calculate diversity of paths taken by a group"""
    # Placeholder - implement based on path_results structure
    return 0.0


if __name__ == "__main__":
    import torch  # Import here to check availability
    run_5k_experiment()