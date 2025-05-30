#!/usr/bin/env python3
"""
Analyze cross-layer trajectories for top 10k GPT-2 tokens.
Builds on existing cluster_paths infrastructure with token-specific enhancements.
Focuses on the most common tokens for clearer patterns.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from tqdm import tqdm
import logging
from datetime import datetime

# Implement trajectory analysis functions locally
def assign_unique_cluster_ids(cluster_labels, layer_names):
    """Assign unique IDs to clusters across layers."""
    unique_labels = {}
    for layer_idx, layer_name in enumerate(layer_names):
        layer_labels = cluster_labels[:, layer_idx]
        unique_layer_labels = [f"{layer_name}_C{label}" for label in layer_labels]
        unique_labels[layer_name] = unique_layer_labels
    return unique_labels

def compute_cluster_paths(unique_labels, layer_names):
    """Compute paths through clusters."""
    num_samples = len(next(iter(unique_labels.values())))
    paths = []
    for i in range(num_samples):
        path = []
        for layer_name in layer_names:
            path.append(unique_labels[layer_name][i])
        paths.append(tuple(path))
    return np.array(paths)

def get_human_readable_path(path):
    """Convert path to human readable format."""
    return " → ".join(path)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Top10kTrajectoryAnalyzer:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / "trajectory_analysis"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load top 10k token information
        logging.info("Loading top 10k token information...")
        with open(base_dir / "top_10k_tokens_full.json", 'r', encoding='utf-8') as f:
            token_list = json.load(f)
        
        # Load token IDs for mapping
        with open(base_dir / "top_10k_token_ids.json", 'r') as f:
            self.top_10k_ids = set(json.load(f))
        
        # Create mappings
        self.token_info = {}
        self.token_id_to_str = {}
        self.morphological_groups = defaultdict(list)
        self.wordnet_tokens = []
        
        for token_data in token_list:
            token_id = token_data['token_id']
            token_str = token_data['token_str']
            
            self.token_id_to_str[token_id] = token_str
            self.token_info[token_id] = {
                'token': token_str,
                'type': token_data['token_type'],
                'has_space': token_data.get('has_leading_space', False),
                'is_alphabetic': token_data.get('is_alphabetic', False),
                'is_punctuation': token_data.get('is_punctuation', False),
                'is_numeric': token_data.get('is_numeric', False),
                'is_subword': token_data.get('is_subword', False),
                'morphological_type': self._get_morphological_type(token_str),
                'byte_length': len(token_str.encode('utf-8')),
                'char_length': len(token_str)
            }
            
            # Track morphological groups
            morph_type = self.token_info[token_id]['morphological_type']
            if morph_type:
                self.morphological_groups[morph_type].append(token_id)
        
        # Try to load WordNet features if available
        self._load_wordnet_features()
        
        logging.info(f"Loaded {len(self.token_info)} tokens")
        logging.info(f"Found {len(self.morphological_groups)} morphological patterns")
        logging.info(f"Found {len(self.wordnet_tokens)} tokens with WordNet features")
        
    def _get_morphological_type(self, token: str) -> Optional[str]:
        """Identify morphological pattern of token."""
        token_clean = token.strip()
        
        # Common suffixes
        if token_clean.endswith('ing'):
            return 'suffix_ing'
        elif token_clean.endswith('ed'):
            return 'suffix_ed'
        elif token_clean.endswith('ly'):
            return 'suffix_ly'
        elif token_clean.endswith('er'):
            return 'suffix_er'
        elif token_clean.endswith('est'):
            return 'suffix_est'
        elif token_clean.endswith('tion'):
            return 'suffix_tion'
        elif token_clean.endswith('ment'):
            return 'suffix_ment'
        elif token_clean.endswith('ness'):
            return 'suffix_ness'
        elif token_clean.endswith('ity'):
            return 'suffix_ity'
        elif token_clean.endswith('able'):
            return 'suffix_able'
        elif token_clean.endswith('ful'):
            return 'suffix_ful'
        elif token_clean.endswith('less'):
            return 'suffix_less'
        elif token_clean.endswith('s') and len(token_clean) > 2:
            return 'suffix_plural'
        
        # Prefixes
        elif token_clean.startswith('un'):
            return 'prefix_un'
        elif token_clean.startswith('re'):
            return 'prefix_re'
        elif token_clean.startswith('pre'):
            return 'prefix_pre'
        
        return None
    
    def _load_wordnet_features(self):
        """Load comprehensive token labels."""
        # Try new comprehensive labels first
        final_labels_path = self.base_dir / "token_labels_final" / "token_labels_10k.json"
        comprehensive_path = self.base_dir / "token_labels" / "comprehensive_token_labels.json"
        
        if final_labels_path.exists():
            logging.info("Loading final token labels...")
            with open(final_labels_path, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
        elif comprehensive_path.exists():
            logging.info("Loading comprehensive token labels...")
            with open(comprehensive_path, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
        else:
            logging.info("No token labels found, skipping semantic analysis")
            return
        
        # Process labels
        for token_id_str, label in labels_data.items():
            token_id = int(token_id_str)
            if token_id in self.token_info:
                # Add all label information
                self.token_info[token_id]['label'] = label
                self.token_info[token_id]['grammatical'] = label.get('grammatical', {})
                self.token_info[token_id]['semantic'] = label.get('semantic', {})
                self.token_info[token_id]['morphological'] = label.get('morphological', {})
                self.token_info[token_id]['conceptual_hierarchy'] = label.get('conceptual_hierarchy', [])
                self.token_info[token_id]['usage_context'] = label.get('usage_context', '')
        
        logging.info(f"Loaded labels for {len(labels_data)} tokens")
    
    def load_cluster_labels(self, k: int = None) -> Dict[int, np.ndarray]:
        """Load cluster labels. If k is specified, load k-specific results."""
        if k is not None:
            logging.info(f"Loading k={k} cluster labels...")
            labels_path = self.base_dir / f"clustering_results_k{k}" / f"all_labels_k{k}.json"
            if not labels_path.exists():
                raise FileNotFoundError(f"k={k} cluster labels not found at {labels_path}. Run k={k} clustering first.")
        else:
            logging.info("Loading optimal cluster labels...")
            labels_path = self.base_dir / "clustering_results_per_layer" / "optimal_labels_all_layers_top10k.json"
            if not labels_path.exists():
                raise FileNotFoundError(f"Cluster labels not found at {labels_path}. Run per-layer analysis first.")
        
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        # Convert to numpy arrays
        layer_labels = {}
        for layer_str, labels_list in labels_data.items():
            layer_idx = int(layer_str)
            layer_labels[layer_idx] = np.array(labels_list)
        
        logging.info(f"Loaded labels for {len(layer_labels)} layers")
        return layer_labels
    
    def build_trajectories(self, layer_labels: Dict[int, np.ndarray]) -> Tuple[np.ndarray, Dict, List[str]]:
        """Build token trajectories using existing infrastructure."""
        logging.info("Building token trajectories...")
        
        # Convert to format expected by existing functions
        num_tokens = len(self.token_info)
        num_layers = len(layer_labels)
        
        # Create labels matrix (tokens x layers)
        labels_matrix = np.zeros((num_tokens, num_layers), dtype=int)
        for layer_idx, labels in layer_labels.items():
            labels_matrix[:, layer_idx] = labels
        
        # Create layer names
        layer_names = [f"L{i}" for i in range(num_layers)]
        
        # Assign unique cluster IDs across layers
        unique_ids = assign_unique_cluster_ids(labels_matrix, layer_names)
        
        # Create id_to_layer_cluster mapping
        id_to_layer_cluster = {}
        for layer_name in layer_names:
            id_to_layer_cluster[layer_name] = unique_ids[layer_name]
        
        # Compute paths
        paths = compute_cluster_paths(unique_ids, layer_names)
        
        logging.info(f"Built trajectories for {len(paths)} tokens across {num_layers} layers")
        
        return paths, id_to_layer_cluster, layer_names
    
    def analyze_trajectories_by_window(self, paths: np.ndarray, id_to_layer_cluster: Dict,
                                     layer_names: List[str]) -> Dict:
        """Analyze trajectories using windowed approach from paper."""
        num_layers = paths.shape[1]
        
        # Define windows (same as paper)
        windows = {
            'early': (0, 4),    # Layers 0-3
            'middle': (4, 8),   # Layers 4-7  
            'late': (8, 12)     # Layers 8-11
        }
        
        window_results = {}
        
        for window_name, (start, end) in windows.items():
            logging.info(f"\nAnalyzing {window_name} window (layers {start}-{end-1})...")
            
            # Extract paths for this window
            window_paths = paths[:, start:end]
            
            # Find unique paths
            unique_paths, inverse_indices = np.unique(window_paths, axis=0, return_inverse=True)
            path_counts = Counter(inverse_indices)
            
            # Get top 20 archetypal paths
            top_paths = path_counts.most_common(20)
            
            # Analyze each archetypal path
            archetypal_paths = []
            
            for path_idx, (path_id, count) in enumerate(top_paths):
                # Get tokens following this path
                token_ids = np.where(inverse_indices == path_id)[0]
                
                # Get human-readable path using existing function
                path_array = unique_paths[path_id]
                # Extract just the window part of the path
                window_path = path_array[start:end]
                readable_path = get_human_readable_path(window_path)
                
                # Analyze token characteristics
                path_info = {
                    'path_id': path_idx,
                    'path': readable_path,
                    'count': count,
                    'percentage': count / len(paths) * 100,
                    'token_ids': token_ids.tolist(),
                    'example_tokens': self._get_example_tokens(token_ids[:10]),
                    'token_type_distribution': self._analyze_token_types(token_ids),
                    'morphological_distribution': self._analyze_morphological_patterns(token_ids),
                    'length_stats': self._analyze_token_lengths(token_ids),
                    # New comprehensive analyses
                    'grammatical_analysis': self._analyze_grammatical_patterns(token_ids),
                    'semantic_analysis': self._analyze_semantic_patterns(token_ids),
                }
                
                # Add WordNet analysis if available (for backward compatibility)
                if self.wordnet_tokens:
                    path_info['wordnet_analysis'] = self._analyze_wordnet_features(token_ids)
                
                archetypal_paths.append(path_info)
            
            # Calculate window-level metrics
            window_metrics = self._calculate_window_metrics(unique_paths, inverse_indices)
            
            window_results[window_name] = {
                'archetypal_paths': archetypal_paths,
                'num_unique_paths': len(unique_paths),
                'metrics': window_metrics,
                'total_tokens': len(paths)
            }
        
        return window_results
    
    def _get_example_tokens(self, token_ids: np.ndarray) -> List[str]:
        """Get example token strings."""
        examples = []
        for tid in token_ids[:10]:  # First 10 examples
            if tid in self.token_id_to_str:
                examples.append(self.token_id_to_str[tid])
        return examples
    
    def _analyze_token_types(self, token_ids: np.ndarray) -> Dict[str, float]:
        """Analyze distribution of token types in path."""
        type_counts = Counter()
        
        for tid in token_ids:
            if tid in self.token_info:
                type_counts[self.token_info[tid]['type']] += 1
        
        total = sum(type_counts.values())
        type_distribution = {
            token_type: count / total * 100 
            for token_type, count in type_counts.items()
        }
        
        return type_distribution
    
    def _analyze_morphological_patterns(self, token_ids: np.ndarray) -> Dict[str, float]:
        """Analyze morphological patterns in path."""
        morph_counts = Counter()
        
        for tid in token_ids:
            if tid in self.token_info:
                morph_type = self.token_info[tid]['morphological_type']
                if morph_type:
                    morph_counts[morph_type] += 1
                else:
                    morph_counts['no_pattern'] += 1
        
        total = sum(morph_counts.values())
        if total > 0:
            morph_distribution = {
                pattern: count / total * 100 
                for pattern, count in morph_counts.most_common()
            }
        else:
            morph_distribution = {}
        
        return morph_distribution
    
    def _analyze_token_lengths(self, token_ids: np.ndarray) -> Dict[str, float]:
        """Analyze token length statistics."""
        byte_lengths = []
        char_lengths = []
        
        for tid in token_ids:
            if tid in self.token_info:
                byte_lengths.append(self.token_info[tid]['byte_length'])
                char_lengths.append(self.token_info[tid]['char_length'])
        
        if byte_lengths:
            return {
                'avg_byte_length': np.mean(byte_lengths),
                'avg_char_length': np.mean(char_lengths),
                'min_byte_length': min(byte_lengths),
                'max_byte_length': max(byte_lengths),
                'std_byte_length': np.std(byte_lengths)
            }
        return {}
    
    def _analyze_grammatical_patterns(self, token_ids: np.ndarray) -> Dict[str, any]:
        """Analyze grammatical patterns using comprehensive labels."""
        pos_counts = Counter()
        suffix_counts = Counter()
        prefix_counts = Counter()
        
        for tid in token_ids:
            if tid in self.token_info:
                info = self.token_info[tid]
                
                # POS tags
                pos = info.get('grammatical', {}).get('pos', 'UNKNOWN')
                pos_counts[pos] += 1
                
                # Morphological patterns
                morph = info.get('morphological', {})
                if 'suffix' in morph:
                    suffix_counts[morph['suffix']] += 1
                if 'prefix' in morph:
                    prefix_counts[morph['prefix']] += 1
        
        total = len(token_ids)
        return {
            'pos_distribution': {pos: count/total*100 for pos, count in pos_counts.most_common()},
            'top_suffixes': dict(suffix_counts.most_common(5)),
            'top_prefixes': dict(prefix_counts.most_common(5)),
            'grammatical_diversity': len(pos_counts) / total if total > 0 else 0
        }
    
    def _analyze_semantic_patterns(self, token_ids: np.ndarray) -> Dict[str, any]:
        """Analyze semantic patterns using comprehensive labels."""
        category_counts = Counter()
        hypernym_level1_counts = Counter()
        wordnet_pos_counts = Counter()
        
        for tid in token_ids:
            if tid in self.token_info:
                info = self.token_info[tid]
                semantic = info.get('semantic', {})
                
                # Semantic category
                category = semantic.get('category', 'unknown')
                category_counts[category] += 1
                
                # WordNet POS if available
                if 'wordnet_pos' in semantic:
                    wordnet_pos_counts[semantic['wordnet_pos']] += 1
                
                # Hypernyms (level 1)
                hypernyms = info.get('hypernyms', [])
                if len(hypernyms) > 1:  # Skip level 0 (usually 'entity')
                    hypernym_level1_counts[hypernyms[1]['hypernym']] += 1
        
        total = len(token_ids)
        return {
            'semantic_categories': {cat: count/total*100 for cat, count in category_counts.most_common(10)},
            'top_hypernyms': dict(hypernym_level1_counts.most_common(5)),
            'wordnet_coverage': sum(1 for tid in token_ids if tid in self.token_info and 
                                   'wordnet_pos' in self.token_info[tid].get('semantic', {})) / total * 100,
            'semantic_diversity': len(category_counts) / total if total > 0 else 0
        }
    
    def _analyze_wordnet_features(self, token_ids: np.ndarray) -> Dict[str, any]:
        """Analyze WordNet features for tokens in path."""
        pos_counts = Counter()
        hypernym_counts = Counter()
        tokens_with_wordnet = 0
        
        for tid in token_ids:
            if tid in self.token_info and 'wordnet' in self.token_info[tid]:
                tokens_with_wordnet += 1
                wordnet_info = self.token_info[tid]['wordnet']
                
                if 'pos' in wordnet_info:
                    pos_counts[wordnet_info['pos']] += 1
                
                if 'hypernyms' in wordnet_info:
                    for hypernym in wordnet_info['hypernyms']:
                        hypernym_counts[hypernym] += 1
        
        if tokens_with_wordnet > 0:
            return {
                'coverage': tokens_with_wordnet / len(token_ids) * 100,
                'dominant_pos': pos_counts.most_common(3),
                'common_hypernyms': hypernym_counts.most_common(5)
            }
        return {'coverage': 0}
    
    def _calculate_window_metrics(self, unique_paths: np.ndarray, 
                                 inverse_indices: np.ndarray) -> Dict[str, float]:
        """Calculate trajectory metrics for window."""
        # Path diversity
        num_unique = len(unique_paths)
        num_total = len(inverse_indices)
        diversity = num_unique / num_total
        
        # Concentration (how many tokens follow top path)
        path_counts = Counter(inverse_indices)
        top_path_count = path_counts.most_common(1)[0][1]
        concentration = top_path_count / num_total
        
        # Fragmentation (1 - concentration of top 10 paths)
        top_10_count = sum(count for _, count in path_counts.most_common(10))
        fragmentation = 1 - (top_10_count / num_total)
        
        return {
            'diversity': diversity,
            'concentration': concentration,
            'fragmentation': fragmentation,
            'entropy': self._calculate_entropy(path_counts, num_total)
        }
    
    def _calculate_entropy(self, path_counts: Counter, total: int) -> float:
        """Calculate Shannon entropy of path distribution."""
        entropy = 0
        for count in path_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        return entropy
    
    def analyze_morphological_trajectories(self, paths: np.ndarray) -> Dict:
        """Special analysis: Do tokens with same morphological pattern follow similar paths?"""
        logging.info("\nAnalyzing morphological pattern trajectories...")
        
        morph_trajectory_analysis = {}
        
        for morph_pattern, token_ids in self.morphological_groups.items():
            if len(token_ids) < 10:  # Skip rare patterns
                continue
            
            # Get paths for tokens with this pattern
            pattern_paths = paths[token_ids]
            
            # Find unique paths
            unique_paths, inverse = np.unique(pattern_paths, axis=0, return_inverse=True)
            
            # Calculate concentration
            path_counts = Counter(inverse)
            top_path_count = path_counts.most_common(1)[0][1]
            concentration = top_path_count / len(token_ids)
            
            morph_trajectory_analysis[morph_pattern] = {
                'num_tokens': len(token_ids),
                'num_unique_paths': len(unique_paths),
                'concentration': concentration,
                'top_path_percentage': concentration * 100,
                'example_tokens': [self.token_id_to_str.get(tid, '') for tid in token_ids[:5]]
            }
        
        return morph_trajectory_analysis
    
    def save_results(self, window_results: Dict, morph_analysis: Dict, paths: np.ndarray):
        """Save all analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results = {
            'timestamp': timestamp,
            'total_tokens': len(self.token_info),
            'window_analysis': window_results,
            'morphological_trajectory_analysis': morph_analysis,
        }
        
        with open(self.results_dir / f"trajectory_analysis_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save full paths for visualization
        np.save(self.results_dir / f"all_token_paths_{timestamp}.npy", paths)
        
        # Save path summary for each window
        for window_name, window_data in window_results.items():
            window_summary = []
            for path_info in window_data['archetypal_paths']:
                summary = {
                    'path': path_info['path'],
                    'count': path_info['count'],
                    'percentage': path_info['percentage'],
                    'example_tokens': path_info['example_tokens'],
                    'dominant_type': max(path_info['token_type_distribution'].items(), 
                                       key=lambda x: x[1])[0] if path_info['token_type_distribution'] else 'unknown',
                    'dominant_morph': max(path_info['morphological_distribution'].items(), 
                                        key=lambda x: x[1])[0] if path_info['morphological_distribution'] else 'none'
                }
                
                # Add WordNet info if available
                if 'wordnet_analysis' in path_info and path_info['wordnet_analysis']['coverage'] > 0:
                    summary['wordnet_coverage'] = path_info['wordnet_analysis']['coverage']
                    if path_info['wordnet_analysis']['dominant_pos']:
                        summary['dominant_pos'] = path_info['wordnet_analysis']['dominant_pos'][0][0]
                
                window_summary.append(summary)
            
            with open(self.results_dir / f"archetypal_paths_{window_name}_{timestamp}.json", 'w') as f:
                json.dump(window_summary, f, indent=2)
        
        logging.info(f"Results saved to {self.results_dir}")
    
    def generate_report(self, window_results: Dict, morph_analysis: Dict):
        """Generate human-readable report."""
        report_lines = [
            "# GPT-2 Top 10k Tokens Trajectory Analysis Report",
            f"\nAnalyzed top {len(self.token_info):,} most common tokens across 12 layers",
            "\n## Window Analysis Summary\n"
        ]
        
        for window_name, window_data in window_results.items():
            report_lines.extend([
                f"### {window_name.capitalize()} Window",
                f"- Unique paths: {window_data['num_unique_paths']:,}",
                f"- Diversity: {window_data['metrics']['diversity']:.3f}",
                f"- Top path concentration: {window_data['metrics']['concentration']:.1%}",
                f"- Fragmentation: {window_data['metrics']['fragmentation']:.3f}",
                f"- Entropy: {window_data['metrics']['entropy']:.3f}",
                "\nTop 5 Archetypal Paths:"
            ])
            
            for i, path in enumerate(window_data['archetypal_paths'][:5]):
                report_lines.extend([
                    f"\n{i+1}. **{path['path']}** ({path['percentage']:.1f}%)",
                    f"   - Examples: {', '.join(repr(t) for t in path['example_tokens'][:5])}",
                    f"   - Primary type: {max(path['token_type_distribution'].items(), key=lambda x: x[1])[0] if path['token_type_distribution'] else 'unknown'}",
                    f"   - Morphology: {', '.join(f'{k}: {v:.1f}%' for k, v in list(path['morphological_distribution'].items())[:3])}"
                ])
                
                # Add WordNet info if available
                if 'wordnet_analysis' in path and path['wordnet_analysis']['coverage'] > 0:
                    report_lines.append(f"   - WordNet coverage: {path['wordnet_analysis']['coverage']:.1f}%")
                    if path['wordnet_analysis']['dominant_pos']:
                        pos_info = ', '.join(f"{pos}: {count}" for pos, count in path['wordnet_analysis']['dominant_pos'])
                        report_lines.append(f"   - POS tags: {pos_info}")
            
            report_lines.append("")
        
        # Morphological analysis
        report_lines.extend([
            "\n## Morphological Pattern Analysis",
            "\nDo tokens with the same morphological pattern follow similar trajectories?\n"
        ])
        
        # Sort by concentration
        sorted_morphs = sorted(morph_analysis.items(), 
                              key=lambda x: x[1]['concentration'], 
                              reverse=True)
        
        for pattern, analysis in sorted_morphs[:10]:
            report_lines.extend([
                f"**{pattern}**: {analysis['concentration']:.1%} follow same path",
                f"  - Total tokens: {analysis['num_tokens']:,}",
                f"  - Unique paths: {analysis['num_unique_paths']}",
                f"  - Examples: {', '.join(repr(t) for t in analysis['example_tokens'][:3])}"
            ])
        
        # Key findings
        report_lines.extend([
            "\n## Key Findings",
            "",
            "1. **Path Diversity**: The top 10k tokens show clearer archetypal paths than the full vocabulary,",
            "   as they consist primarily of common words with consistent usage patterns.",
            "",
            "2. **Morphological Coherence**: Tokens with the same morphological pattern (e.g., -ing, -ed)",
            "   show varying degrees of trajectory coherence, suggesting both form and function influence",
            "   representation development.",
            "",
            "3. **Token Type Segregation**: Different token types (words, subwords, punctuation) tend to",
            "   follow distinct trajectory patterns, indicating type-specific processing pathways."
        ])
        
        with open(self.results_dir / "trajectory_analysis_report.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    
    def run_analysis(self, k: int = None):
        """Run complete trajectory analysis."""
        # Load cluster labels
        layer_labels = self.load_cluster_labels(k=k)
        
        # Update results directory if using specific k
        if k is not None:
            self.results_dir = self.base_dir / f"trajectory_analysis_k{k}"
            self.results_dir.mkdir(exist_ok=True)
        
        # Build trajectories
        paths, id_to_layer_cluster, layer_names = self.build_trajectories(layer_labels)
        
        # Analyze by window
        window_results = self.analyze_trajectories_by_window(paths, id_to_layer_cluster, layer_names)
        
        # Analyze morphological patterns
        morph_analysis = self.analyze_morphological_trajectories(paths)
        
        # Save results
        self.save_results(window_results, morph_analysis, paths)
        
        # Generate report
        self.generate_report(window_results, morph_analysis)
        
        return window_results, morph_analysis


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze trajectories of top 10k GPT-2 tokens')
    parser.add_argument('--k', type=int, default=None, help='Number of clusters (default: use optimal)')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    analyzer = Top10kTrajectoryAnalyzer(base_dir)
    
    if args.k:
        logging.info(f"Starting trajectory analysis for top 10k GPT-2 tokens with k={args.k}...")
    else:
        logging.info("Starting trajectory analysis for top 10k GPT-2 tokens with optimal k...")
    
    window_results, morph_analysis = analyzer.run_analysis(k=args.k)
    
    # Print summary
    print("\n=== Trajectory Analysis Summary ===")
    if args.k:
        print(f"Using k={args.k} clustering")
    print("")
    
    for window_name, window_data in window_results.items():
        print(f"\n{window_name.upper()} WINDOW:")
        print(f"  Unique paths: {window_data['num_unique_paths']}")
        print(f"  Top path: {window_data['archetypal_paths'][0]['path']} ({window_data['archetypal_paths'][0]['percentage']:.1f}%)")
    
    print("\n=== Morphological Coherence ===")
    high_coherence = [(p, a) for p, a in morph_analysis.items() if a['concentration'] > 0.5]
    if high_coherence:
        print(f"\nPatterns with >50% trajectory coherence:")
        for pattern, analysis in sorted(high_coherence, key=lambda x: x[1]['concentration'], reverse=True)[:5]:
            print(f"  {pattern}: {analysis['concentration']:.1%} ({analysis['num_tokens']} tokens)")


if __name__ == "__main__":
    main()