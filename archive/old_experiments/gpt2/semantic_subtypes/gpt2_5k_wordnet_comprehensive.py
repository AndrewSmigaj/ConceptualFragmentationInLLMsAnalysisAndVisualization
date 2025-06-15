"""
Comprehensive GPT-2 5000 Words Experiment with Full WordNet Integration

Optimized for performance and comprehensive linguistic analysis.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
import pickle

# NLP libraries
import nltk
from nltk.corpus import wordnet as wn

# ML/DL libraries
import torch
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import existing infrastructure
from experiments.gpt2.shared.gpt2_token_validator import GPT2TokenValidator


@dataclass
class WordNetFeatures:
    """Complete WordNet features for a word"""
    word: str
    token_id: int
    frequency_rank: int
    
    # POS information
    all_pos: List[str] = field(default_factory=list)  # All POS from synsets
    primary_pos: str = ""
    secondary_pos: Optional[str] = None
    pos_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Semantic information
    synsets: List[str] = field(default_factory=list)
    primary_synset: Optional[str] = None
    polysemy_count: int = 0
    
    # Hierarchical information
    hypernym_chains: List[List[str]] = field(default_factory=list)  # Multiple chains
    max_depth: int = 0
    semantic_domains: Set[str] = field(default_factory=set)  # Top-level categories
    
    # Semantic properties
    is_concrete: bool = False
    is_abstract: bool = False
    is_animate: bool = False
    is_artifact: bool = False
    is_natural: bool = False
    is_physical: bool = False
    is_mental: bool = False
    
    # Morphological properties
    has_antonym: bool = False
    has_meronym: bool = False
    is_compound: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        d['semantic_domains'] = list(self.semantic_domains)
        return d


class EnhancedWordNetAnalyzer:
    """Enhanced WordNet analyzer with caching and batch processing"""
    
    def __init__(self):
        self.validator = GPT2TokenValidator()
        
        # Cache for expensive computations
        self.synset_cache = {}
        self.hypernym_cache = {}
        
        # Semantic category markers
        self.concrete_markers = {
            'physical_entity.n.01', 'object.n.01', 'artifact.n.01',
            'organism.n.01', 'substance.n.01', 'food.n.01', 'body.n.01'
        }
        self.abstract_markers = {
            'abstraction.n.06', 'psychological_feature.n.01', 'cognition.n.01',
            'attribute.n.02', 'relation.n.01', 'measure.n.02', 'communication.n.02'
        }
        self.animate_markers = {
            'organism.n.01', 'living_thing.n.01', 'animal.n.01', 'person.n.01'
        }
        self.artifact_markers = {
            'artifact.n.01', 'creation.n.02', 'product.n.02'
        }
        self.mental_markers = {
            'psychological_feature.n.01', 'cognition.n.01', 'feeling.n.01'
        }
    
    def analyze_word(self, word: str, token_id: int, rank: int) -> WordNetFeatures:
        """Comprehensive WordNet analysis of a word"""
        features = WordNetFeatures(word=word, token_id=token_id, frequency_rank=rank)
        
        # Get all synsets
        synsets = wn.synsets(word)
        if not synsets:
            return features
        
        features.synsets = [s.name() for s in synsets]
        features.primary_synset = synsets[0].name()
        features.polysemy_count = len(synsets)
        
        # Analyze POS distribution
        pos_counts = Counter(s.pos() for s in synsets)
        total_pos = sum(pos_counts.values())
        features.all_pos = list(pos_counts.keys())
        features.pos_distribution = {pos: count/total_pos for pos, count in pos_counts.items()}
        
        # Primary and secondary POS
        pos_ranked = pos_counts.most_common()
        features.primary_pos = pos_ranked[0][0] if pos_ranked else ""
        if len(pos_ranked) > 1 and pos_ranked[1][1] / total_pos > 0.2:
            features.secondary_pos = pos_ranked[1][0]
        
        # Analyze hypernym structure (limit to top 3 senses)
        all_hypernyms = set()
        for synset in synsets[:3]:
            chain = self._get_hypernym_chain(synset)
            if chain:
                features.hypernym_chains.append([h.name() for h in chain])
                features.max_depth = max(features.max_depth, len(chain))
                all_hypernyms.update(chain)
                
                # Top-level semantic domain
                if chain:
                    features.semantic_domains.add(chain[-1].name())
        
        # Determine semantic properties
        hypernym_names = {h.name() for h in all_hypernyms}
        features.is_concrete = bool(hypernym_names & self.concrete_markers)
        features.is_abstract = bool(hypernym_names & self.abstract_markers)
        features.is_animate = bool(hypernym_names & self.animate_markers)
        features.is_artifact = bool(hypernym_names & self.artifact_markers)
        features.is_physical = features.is_concrete or features.is_artifact
        features.is_mental = bool(hypernym_names & self.mental_markers)
        
        # Check relationships
        features.has_antonym = any(lemma.antonyms() for s in synsets for lemma in s.lemmas())
        features.has_meronym = any(s.part_meronyms() or s.substance_meronyms() for s in synsets)
        features.is_compound = '_' in word or '-' in word
        
        return features
    
    def _get_hypernym_chain(self, synset) -> List:
        """Get hypernym chain with caching"""
        if synset.name() in self.hypernym_cache:
            return self.hypernym_cache[synset.name()]
        
        chain = [synset]
        current = synset
        
        while current.hypernyms():
            current = current.hypernyms()[0]
            chain.append(current)
        
        self.hypernym_cache[synset.name()] = chain
        return chain


class WordNetClusterAnalyzer:
    """Analyzes clustering results with respect to WordNet features"""
    
    def __init__(self, words: List[WordNetFeatures], activations: np.ndarray, 
                 cluster_assignments: Dict[int, np.ndarray], paths: List[List[str]]):
        self.words = words
        self.activations = activations
        self.cluster_assignments = cluster_assignments
        self.paths = paths
        self.n_layers = activations.shape[1]
    
    def analyze_cluster_composition(self, layer_idx: int) -> Dict:
        """Analyze what types of words are in each cluster at a given layer"""
        clusters = self.cluster_assignments[layer_idx]
        n_clusters = len(np.unique(clusters))
        
        results = {
            'layer': layer_idx,
            'n_clusters': n_clusters,
            'clusters': {}
        }
        
        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            cluster_words = [w for i, w in enumerate(self.words) if mask[i]]
            
            # Analyze composition
            composition = {
                'size': len(cluster_words),
                'pos_distribution': self._get_pos_distribution(cluster_words),
                'semantic_properties': self._get_semantic_properties(cluster_words),
                'polysemy_stats': self._get_polysemy_stats(cluster_words),
                'semantic_domains': self._get_domain_distribution(cluster_words),
                'example_words': [w.word for w in cluster_words[:10]]
            }
            
            results['clusters'][f'C{cluster_id}'] = composition
        
        return results
    
    def analyze_path_patterns(self) -> Dict:
        """Analyze path patterns with respect to WordNet features"""
        results = {
            'total_paths': len(self.paths),
            'unique_paths': len(set(tuple(p) for p in self.paths)),
            'convergence_by_layer': {},
            'path_patterns_by_feature': {}
        }
        
        # Analyze convergence by layer
        for layer_idx in range(self.n_layers):
            clusters_at_layer = [path[layer_idx] for path in self.paths]
            unique_clusters = len(set(clusters_at_layer))
            results['convergence_by_layer'][layer_idx] = {
                'unique_clusters': unique_clusters,
                'convergence_ratio': 1.0 - (unique_clusters - 1) / max(1, len(self.paths) - 1)
            }
        
        # Analyze paths by linguistic features
        feature_groups = self._group_words_by_features()
        
        for feature_name, word_indices in feature_groups.items():
            if len(word_indices) < 5:  # Skip small groups
                continue
                
            group_paths = [self.paths[i] for i in word_indices]
            
            # Calculate path similarity within group
            path_similarity = self._calculate_path_similarity(group_paths)
            
            # Find dominant paths
            path_counts = Counter(tuple(p) for p in group_paths)
            dominant_paths = path_counts.most_common(3)
            
            results['path_patterns_by_feature'][feature_name] = {
                'n_words': len(word_indices),
                'n_unique_paths': len(set(tuple(p) for p in group_paths)),
                'path_similarity': path_similarity,
                'dominant_paths': [
                    {'path': ' -> '.join(p), 'count': c, 'percentage': c/len(group_paths)*100}
                    for p, c in dominant_paths
                ],
                'late_layer_convergence': self._calculate_late_convergence(group_paths)
            }
        
        return results
    
    def _group_words_by_features(self) -> Dict[str, List[int]]:
        """Group word indices by various linguistic features"""
        groups = defaultdict(list)
        
        for i, word in enumerate(self.words):
            # POS groups
            if word.primary_pos:
                groups[f'pos_{word.primary_pos}'].append(i)
            
            # Polysemy groups
            if word.polysemy_count == 1:
                groups['monosemous'].append(i)
            elif word.polysemy_count <= 3:
                groups['low_polysemy'].append(i)
            else:
                groups['high_polysemy'].append(i)
            
            # Semantic property groups
            if word.is_concrete:
                groups['concrete'].append(i)
            if word.is_abstract:
                groups['abstract'].append(i)
            if word.is_animate:
                groups['animate'].append(i)
            if word.is_artifact:
                groups['artifact'].append(i)
            
            # Semantic domain groups
            for domain in word.semantic_domains:
                groups[f'domain_{domain}'].append(i)
            
            # Combined features
            if word.primary_pos == 'n' and word.is_concrete:
                groups['concrete_nouns'].append(i)
            if word.primary_pos == 'n' and word.is_abstract:
                groups['abstract_nouns'].append(i)
            if word.primary_pos == 'v' and word.is_physical:
                groups['physical_verbs'].append(i)
            if word.primary_pos == 'v' and word.is_mental:
                groups['mental_verbs'].append(i)
        
        return dict(groups)
    
    def _get_pos_distribution(self, words: List[WordNetFeatures]) -> Dict[str, float]:
        """Get POS distribution for a group of words"""
        pos_counts = Counter(w.primary_pos for w in words if w.primary_pos)
        total = sum(pos_counts.values())
        return {pos: count/total for pos, count in pos_counts.items()} if total > 0 else {}
    
    def _get_semantic_properties(self, words: List[WordNetFeatures]) -> Dict[str, float]:
        """Get semantic property distribution"""
        total = len(words)
        if total == 0:
            return {}
        
        return {
            'concrete': sum(1 for w in words if w.is_concrete) / total,
            'abstract': sum(1 for w in words if w.is_abstract) / total,
            'animate': sum(1 for w in words if w.is_animate) / total,
            'artifact': sum(1 for w in words if w.is_artifact) / total,
            'physical': sum(1 for w in words if w.is_physical) / total,
            'mental': sum(1 for w in words if w.is_mental) / total
        }
    
    def _get_polysemy_stats(self, words: List[WordNetFeatures]) -> Dict[str, float]:
        """Get polysemy statistics"""
        if not words:
            return {}
        
        polysemy_counts = [w.polysemy_count for w in words]
        return {
            'mean': np.mean(polysemy_counts),
            'std': np.std(polysemy_counts),
            'max': max(polysemy_counts),
            'monosemous_ratio': sum(1 for c in polysemy_counts if c == 1) / len(polysemy_counts)
        }
    
    def _get_domain_distribution(self, words: List[WordNetFeatures]) -> Dict[str, int]:
        """Get semantic domain distribution"""
        domain_counts = Counter()
        for word in words:
            domain_counts.update(word.semantic_domains)
        return dict(domain_counts.most_common(10))  # Top 10 domains
    
    def _calculate_path_similarity(self, paths: List[List[str]]) -> float:
        """Calculate average pairwise path similarity"""
        if len(paths) < 2:
            return 1.0
        
        similarities = []
        # Sample for efficiency
        sample_size = min(100, len(paths))
        sample_indices = np.random.choice(len(paths), sample_size, replace=False)
        
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                path1 = paths[sample_indices[i]]
                path2 = paths[sample_indices[j]]
                # Similarity = fraction of matching positions
                matches = sum(1 for a, b in zip(path1, path2) if a == b)
                similarities.append(matches / len(path1))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_late_convergence(self, paths: List[List[str]]) -> Dict[str, float]:
        """Calculate convergence in late layers (9-11)"""
        convergence = {}
        
        for layer_idx in range(9, min(12, self.n_layers)):
            clusters_at_layer = [p[layer_idx] for p in paths]
            unique = len(set(clusters_at_layer))
            convergence[f'layer_{layer_idx}'] = 1.0 - (unique - 1) / max(1, len(paths) - 1)
        
        return convergence


def extract_common_words_with_wordnet(n_words: int = 5000) -> List[WordNetFeatures]:
    """Extract common words with full WordNet analysis"""
    logger.info(f"Extracting {n_words} common single-token words with WordNet features...")
    
    # Try to load from cache first
    cache_file = Path(f"experiments/gpt2/semantic_subtypes/5k_common_words/wordnet_words_{n_words}.pkl")
    if cache_file.exists():
        logger.info("Loading from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Get word frequency list
    logger.info("Getting word frequency list...")
    # Use a predefined list for consistency and speed
    with open(Path(__file__).parent / "common_words_10k.txt", 'r') as f:
        freq_words = [line.strip() for line in f if line.strip()]
    
    # Fallback to NLTK Brown corpus if file doesn't exist
    if not freq_words:
        from nltk.corpus import brown
        word_freq = Counter()
        for word in brown.words()[:500000]:  # Limit for speed
            if word.isalpha() and len(word) > 1:
                word_freq[word.lower()] += 1
        freq_words = [word for word, _ in word_freq.most_common(n_words * 3)]
    
    # Validate and analyze
    analyzer = EnhancedWordNetAnalyzer()
    validator = GPT2TokenValidator()
    enriched_words = []
    
    logger.info("Validating and analyzing words...")
    for i, word in enumerate(freq_words):
        if len(enriched_words) >= n_words:
            break
            
        if i % 500 == 0:
            logger.info(f"Progress: {i} tested, {len(enriched_words)} valid")
        
        # Validate single token
        validation = validator.validate_word(word)
        if not validation.is_single_token:
            continue
        
        # Analyze with WordNet
        features = analyzer.analyze_word(word, validation.token_ids[0], len(enriched_words) + 1)
        enriched_words.append(features)
    
    # Cache results
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(enriched_words, f)
    
    logger.info(f"Extracted {len(enriched_words)} words with WordNet features")
    return enriched_words


def extract_gpt2_activations(words: List[WordNetFeatures], batch_size: int = 32) -> np.ndarray:
    """Extract GPT-2 activations efficiently"""
    logger.info("Extracting GPT-2 activations...")
    
    # Load model
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_activations = []
    
    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch = words[i:i+batch_size]
            if i % (batch_size * 5) == 0:
                logger.info(f"Processing words {i}-{i+len(batch)}/{len(words)}")
            
            batch_activations = []
            for word_features in batch:
                # Create input
                input_ids = torch.tensor([[word_features.token_id]], device=device)
                
                # Get hidden states
                outputs = model(input_ids, output_hidden_states=True)
                
                # Extract activations from each layer (skip embedding)
                word_acts = []
                for layer_idx in range(1, 13):
                    act = outputs.hidden_states[layer_idx][0, 0, :].cpu().numpy()
                    word_acts.append(act)
                
                batch_activations.append(word_acts)
            
            all_activations.extend(batch_activations)
    
    return np.array(all_activations)


def cluster_activations(activations: np.ndarray, k_values: List[int] = [3, 5, 8]) -> Dict:
    """Cluster activations with multiple k values"""
    logger.info("Clustering activations...")
    
    n_words, n_layers, hidden_dim = activations.shape
    results = {}
    
    for k in k_values:
        logger.info(f"Clustering with k={k}...")
        cluster_assignments = {}
        silhouette_scores = {}
        
        for layer_idx in range(n_layers):
            layer_acts = activations[:, layer_idx, :]
            
            # Cluster
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(layer_acts)
            cluster_assignments[layer_idx] = labels
            
            # Calculate silhouette score
            if k > 1:
                score = silhouette_score(layer_acts, labels, sample_size=min(1000, n_words))
                silhouette_scores[layer_idx] = score
        
        # Extract paths
        paths = []
        for word_idx in range(n_words):
            path = []
            for layer_idx in range(n_layers):
                cluster_id = cluster_assignments[layer_idx][word_idx]
                path.append(f'L{layer_idx}_C{cluster_id}')
            paths.append(path)
        
        results[k] = {
            'cluster_assignments': cluster_assignments,
            'paths': paths,
            'silhouette_scores': silhouette_scores
        }
    
    return results


def run_comprehensive_analysis():
    """Run the full 5000-word experiment with comprehensive analysis"""
    output_dir = Path("experiments/gpt2/semantic_subtypes/5k_common_words")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract words with WordNet features
    words = extract_common_words_with_wordnet(5000)
    
    # Save word features
    logger.info("Saving word features...")
    with open(output_dir / "wordnet_features.json", 'w') as f:
        json.dump([w.to_dict() for w in words], f, indent=2)
    
    # Extract activations
    activations = extract_gpt2_activations(words)
    np.save(output_dir / "activations.npy", activations)
    
    # Cluster with multiple k values
    clustering_results = cluster_activations(activations, k_values=[3, 5, 8])
    
    # Analyze each clustering configuration
    all_analyses = {}
    
    for k, cluster_data in clustering_results.items():
        logger.info(f"\nAnalyzing k={k} clustering...")
        
        analyzer = WordNetClusterAnalyzer(
            words=words,
            activations=activations,
            cluster_assignments=cluster_data['cluster_assignments'],
            paths=cluster_data['paths']
        )
        
        # Analyze cluster composition at different layers
        layer_analyses = {}
        for layer_idx in [0, 3, 6, 9, 11]:  # Sample layers
            layer_analyses[f'layer_{layer_idx}'] = analyzer.analyze_cluster_composition(layer_idx)
        
        # Analyze path patterns
        path_analysis = analyzer.analyze_path_patterns()
        
        all_analyses[f'k_{k}'] = {
            'cluster_composition': layer_analyses,
            'path_patterns': path_analysis,
            'silhouette_scores': cluster_data['silhouette_scores']
        }
    
    # Save comprehensive results
    logger.info("Saving analysis results...")
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    serializable_analyses = convert_numpy_types(all_analyses)
    
    with open(output_dir / "comprehensive_analysis.json", 'w') as f:
        json.dump(serializable_analyses, f, indent=2)
    
    # Generate summary report
    generate_summary_report(words, all_analyses, output_dir)
    
    logger.info(f"Analysis complete! Results saved to {output_dir}")


def generate_summary_report(words: List[WordNetFeatures], analyses: Dict, output_dir: Path):
    """Generate a human-readable summary report"""
    logger.info("Generating summary report...")
    
    report = []
    report.append("# GPT-2 5000 Common Words Analysis with WordNet\n")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Dataset statistics
    report.append("## Dataset Statistics\n")
    report.append(f"- Total words analyzed: {len(words)}\n")
    
    # POS distribution
    pos_counts = Counter(w.primary_pos for w in words if w.primary_pos)
    report.append("\n### POS Distribution:\n")
    for pos, count in pos_counts.most_common():
        report.append(f"- {pos}: {count} ({count/len(words)*100:.1f}%)\n")
    
    # Polysemy statistics
    polysemy_counts = [w.polysemy_count for w in words]
    report.append(f"\n### Polysemy Statistics:\n")
    report.append(f"- Average senses per word: {np.mean(polysemy_counts):.2f}\n")
    report.append(f"- Monosemous words: {sum(1 for p in polysemy_counts if p == 1)} "
                  f"({sum(1 for p in polysemy_counts if p == 1)/len(words)*100:.1f}%)\n")
    
    # Semantic properties
    report.append("\n### Semantic Properties:\n")
    properties = ['is_concrete', 'is_abstract', 'is_animate', 'is_artifact']
    for prop in properties:
        count = sum(1 for w in words if getattr(w, prop))
        report.append(f"- {prop}: {count} ({count/len(words)*100:.1f}%)\n")
    
    # Key findings from each k
    report.append("\n## Key Findings\n")
    
    for k_config, analysis in analyses.items():
        report.append(f"\n### {k_config.upper()}\n")
        
        # Path convergence
        path_data = analysis['path_patterns']
        report.append(f"- Unique paths: {path_data['unique_paths']} / {path_data['total_paths']}\n")
        
        # Late layer convergence
        late_convergence = []
        for layer in [9, 10, 11]:
            if layer in path_data['convergence_by_layer']:
                conv = path_data['convergence_by_layer'][layer]['convergence_ratio']
                late_convergence.append(f"L{layer}: {conv:.3f}")
        report.append(f"- Late layer convergence: {', '.join(late_convergence)}\n")
        
        # Most interesting patterns
        report.append("\n#### Path Patterns by Feature:\n")
        patterns = path_data['path_patterns_by_feature']
        
        # Sort by path similarity to find most coherent groups
        sorted_patterns = sorted(patterns.items(), 
                               key=lambda x: x[1]['path_similarity'], 
                               reverse=True)
        
        for feature, data in sorted_patterns[:10]:  # Top 10
            report.append(f"\n**{feature}** ({data['n_words']} words):\n")
            report.append(f"- Path similarity: {data['path_similarity']:.3f}\n")
            report.append(f"- Unique paths: {data['n_unique_paths']}\n")
            
            # Late convergence
            late_conv = data.get('late_layer_convergence', {})
            if late_conv:
                conv_str = ', '.join([f"{k}: {v:.3f}" for k, v in late_conv.items()])
                report.append(f"- Late convergence: {conv_str}\n")
    
    # Write report
    with open(output_dir / "analysis_report.md", 'w') as f:
        f.writelines(report)
    
    logger.info("Report generated!")


if __name__ == "__main__":
    # First, ensure we have the common words list
    common_words_file = Path(__file__).parent / "common_words_10k.txt"
    if not common_words_file.exists():
        logger.info("Downloading common words list...")
        import requests
        try:
            url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(common_words_file, 'w') as f:
                    f.write(response.text)
                logger.info("Downloaded common words list")
        except Exception as e:
            logger.warning(f"Could not download word list: {e}")
    
    # Run the analysis
    run_comprehensive_analysis()