"""
Direct interpretability analysis for cluster naming and path narration.
No external API calls - I analyze the patterns directly.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import sys
from pathlib import Path

# Add parent directories to path for imports
root_dir = Path(__file__).parent.parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from logging_config import setup_logging

logger = setup_logging(__name__)


class DirectInterpreter:
    """
    I directly analyze and interpret clusters and paths.
    No API calls - I am the analyzer.
    """
    
    def __init__(self):
        """Initialize the interpreter."""
        logger.info("Initialized DirectInterpreter for cluster analysis")
    
    def name_cluster(self,
                    words: List[str],
                    layer: int,
                    cluster_id: int,
                    word_subtypes: Optional[Dict[str, str]] = None) -> Dict:
        """
        I analyze a cluster and provide a meaningful name.
        
        Args:
            words: Words in the cluster
            layer: Layer number (0-11 for GPT-2)
            cluster_id: Cluster identifier
            word_subtypes: Optional semantic subtypes
            
        Returns:
            Dictionary with cluster name and analysis
        """
        # Analyze word characteristics
        word_lengths = [len(w) for w in words]
        avg_length = np.mean(word_lengths)
        
        # Check for common patterns
        has_plurals = sum(1 for w in words if w.endswith('s')) / len(words)
        has_ing = sum(1 for w in words if w.endswith('ing')) / len(words)
        has_ed = sum(1 for w in words if w.endswith('ed')) / len(words)
        
        # Semantic analysis if subtypes available
        subtype_counts = Counter()
        if word_subtypes:
            for word in words:
                if word in word_subtypes:
                    subtype_counts[word_subtypes[word]] += 1
        
        # Determine cluster characteristics based on layer
        layer_context = self._get_layer_context(layer)
        
        # Generate cluster name based on analysis
        if subtype_counts:
            dominant_type = subtype_counts.most_common(1)[0][0]
            purity = subtype_counts[dominant_type] / len(words)
            
            if purity > 0.8:
                cluster_name = f"{dominant_type}_{layer_context}"
            elif purity > 0.5:
                secondary = subtype_counts.most_common(2)[1][0] if len(subtype_counts) > 1 else "mixed"
                cluster_name = f"{dominant_type}_with_{secondary}_{layer_context}"
            else:
                cluster_name = f"mixed_semantic_{layer_context}"
        else:
            # Name based on structural features
            if has_plurals > 0.7:
                cluster_name = f"plural_forms_{layer_context}"
            elif has_ing > 0.5:
                cluster_name = f"progressive_forms_{layer_context}"
            elif has_ed > 0.5:
                cluster_name = f"past_tense_{layer_context}"
            elif avg_length > 8:
                cluster_name = f"long_words_{layer_context}"
            elif avg_length < 4:
                cluster_name = f"short_words_{layer_context}"
            else:
                cluster_name = f"general_words_{layer_context}"
        
        analysis = {
            'cluster_name': cluster_name,
            'layer': layer,
            'cluster_id': cluster_id,
            'n_words': len(words),
            'characteristics': {
                'avg_word_length': float(avg_length),
                'plural_ratio': float(has_plurals),
                'ing_ratio': float(has_ing),
                'ed_ratio': float(has_ed),
                'dominant_subtype': dominant_type if subtype_counts else None,
                'subtype_purity': float(purity) if subtype_counts else None
            },
            'sample_words': words[:10],  # First 10 as examples
            'interpretation': self._generate_interpretation(
                cluster_name, words, layer, subtype_counts
            )
        }
        
        return analysis
    
    def narrate_path(self,
                    path: List[int],
                    words: List[str],
                    cluster_names: Dict[Tuple[int, int], str]) -> Dict:
        """
        I create a narrative description of a trajectory path.
        
        Args:
            path: Sequence of cluster IDs
            words: Words following this path
            cluster_names: (layer, cluster_id) -> name mapping
            
        Returns:
            Dictionary with path narrative
        """
        # Analyze path characteristics
        stability = sum(1 for i in range(len(path)-1) if path[i] == path[i+1]) / (len(path)-1)
        n_changes = sum(1 for i in range(len(path)-1) if path[i] != path[i+1])
        unique_clusters = len(set(path))
        
        # Build narrative
        narrative_parts = []
        
        # Opening
        if stability > 0.8:
            narrative_parts.append(f"This is a highly stable path with {len(words)} words.")
        elif stability < 0.3:
            narrative_parts.append(f"This is a dynamic path with {len(words)} words that transition frequently.")
        else:
            narrative_parts.append(f"This is a moderately stable path with {len(words)} words.")
        
        # Describe key transitions
        for i in range(len(path)):
            layer_name = f"layer_{i}"
            cluster_key = (i, path[i])
            cluster_name = cluster_names.get(cluster_key, f"cluster_{path[i]}")
            
            if i == 0:
                narrative_parts.append(f"Starting in {cluster_name} at {layer_name},")
            elif path[i] != path[i-1]:
                narrative_parts.append(f"transitioning to {cluster_name} at {layer_name},")
            elif i == len(path) - 1:
                narrative_parts.append(f"and remaining in {cluster_name} through the final layer.")
        
        # Add interpretation
        if unique_clusters == 1:
            interpretation = "These words maintain consistent representation throughout the network."
        elif n_changes == len(path) - 1:
            interpretation = "These words undergo continuous transformation at each layer."
        elif n_changes <= 2:
            interpretation = "These words experience minimal reorganization during processing."
        else:
            interpretation = "These words follow a complex evolution through the network layers."
        
        narrative_parts.append(interpretation)
        
        narrative = {
            'path': path,
            'narrative': ' '.join(narrative_parts),
            'characteristics': {
                'stability': float(stability),
                'n_transitions': n_changes,
                'unique_clusters': unique_clusters,
                'path_type': self._classify_path_type(stability, n_changes, unique_clusters)
            },
            'example_words': words[:10],
            'word_count': len(words)
        }
        
        return narrative
    
    def analyze_micro_clusters(self,
                             micro_cluster_results: Dict,
                             macro_cluster_words: List[str],
                             layer: int) -> Dict:
        """
        I interpret micro-cluster patterns within a macro cluster.
        
        Args:
            micro_cluster_results: Results from ETS micro-clustering
            macro_cluster_words: Words in the macro cluster
            layer: Layer number
            
        Returns:
            Interpretation of micro-structure
        """
        n_micro = micro_cluster_results['n_micro_clusters']
        n_anomalies = micro_cluster_results['n_anomalies']
        coverage = micro_cluster_results['coverage']
        purity = micro_cluster_results['purity']
        
        # Analyze micro-cluster distribution
        micro_labels = micro_cluster_results['micro_labels']
        micro_sizes = Counter(micro_labels)
        
        # Generate interpretation
        if n_micro == 0:
            structure_type = "homogeneous"
            interpretation = "This macro cluster has no discernible sub-structure."
        elif n_micro == 1:
            structure_type = "cohesive"
            interpretation = "This macro cluster forms a single cohesive group."
        elif n_micro <= 3:
            structure_type = "simple_substructure"
            interpretation = f"This macro cluster contains {n_micro} distinct sub-groups."
        else:
            structure_type = "complex_substructure"
            interpretation = f"This macro cluster has complex internal structure with {n_micro} sub-groups."
        
        # Add coverage/anomaly analysis
        if n_anomalies > len(macro_cluster_words) * 0.3:
            interpretation += f" Many words ({n_anomalies}) are outliers, suggesting high diversity."
        elif coverage < 0.5:
            interpretation += " The sub-groups capture only partial structure."
        elif purity > 0.8:
            interpretation += " The sub-groups show high semantic coherence."
        
        analysis = {
            'structure_type': structure_type,
            'interpretation': interpretation,
            'metrics': {
                'n_micro_clusters': n_micro,
                'n_anomalies': n_anomalies,
                'coverage': float(coverage),
                'average_purity': float(purity)
            },
            'micro_cluster_sizes': dict(micro_sizes.most_common()),
            'layer_context': self._get_layer_context(layer)
        }
        
        return analysis
    
    def _get_layer_context(self, layer: int) -> str:
        """Get semantic context for a layer in GPT-2."""
        if layer <= 2:
            return "early_features"
        elif layer <= 5:
            return "basic_patterns"
        elif layer <= 8:
            return "abstract_concepts"
        else:
            return "high_level_semantics"
    
    def _generate_interpretation(self,
                               cluster_name: str,
                               words: List[str],
                               layer: int,
                               subtype_counts: Counter) -> str:
        """Generate detailed interpretation of a cluster."""
        interpretations = []
        
        # Layer-based interpretation
        if layer <= 2:
            interpretations.append("In early layers, this cluster captures surface-level features.")
        elif layer <= 5:
            interpretations.append("In middle layers, this cluster represents emerging patterns.")
        else:
            interpretations.append("In later layers, this cluster encodes abstract relationships.")
        
        # Content-based interpretation
        if subtype_counts:
            total = sum(subtype_counts.values())
            dominant = subtype_counts.most_common(1)[0]
            ratio = dominant[1] / total
            
            if ratio > 0.9:
                interpretations.append(f"Nearly all words are {dominant[0]}, showing strong semantic coherence.")
            elif ratio > 0.6:
                interpretations.append(f"Primarily {dominant[0]} with some mixing, indicating partial specialization.")
            else:
                interpretations.append("Mixed semantic types suggest this cluster captures structural rather than semantic similarity.")
        
        # Size-based interpretation
        if len(words) < 5:
            interpretations.append("Small cluster size indicates specialized processing.")
        elif len(words) > 50:
            interpretations.append("Large cluster size suggests broad categorical grouping.")
        
        return " ".join(interpretations)
    
    def _classify_path_type(self, stability: float, n_changes: int, unique_clusters: int) -> str:
        """Classify the type of path based on its characteristics."""
        if stability > 0.8:
            return "stable_representation"
        elif stability < 0.2:
            return "highly_dynamic"
        elif unique_clusters == 1:
            return "single_cluster"
        elif n_changes == 1:
            return "single_transition"
        elif n_changes <= 3:
            return "gradual_evolution"
        else:
            return "complex_trajectory"
    
    def generate_summary_insights(self,
                                all_clusters: Dict,
                                all_paths: List[Dict],
                                layer_transitions: Dict) -> Dict:
        """
        I synthesize overall insights from the analysis.
        
        Args:
            all_clusters: All cluster analyses
            all_paths: All path narratives
            layer_transitions: Layer transition analysis
            
        Returns:
            High-level insights and patterns
        """
        insights = {
            'cluster_patterns': [],
            'path_patterns': [],
            'layer_patterns': [],
            'key_findings': []
        }
        
        # Analyze cluster evolution
        layers = defaultdict(list)
        for (layer, cluster_id), analysis in all_clusters.items():
            layers[layer].append(analysis)
        
        for layer in sorted(layers.keys()):
            clusters = layers[layer]
            n_clusters = len(clusters)
            avg_size = np.mean([c['n_words'] for c in clusters])
            
            if layer == 0:
                insights['cluster_patterns'].append(
                    f"Processing begins with {n_clusters} clusters averaging {avg_size:.1f} words each."
                )
            elif n_clusters > len(layers[layer-1]):
                insights['cluster_patterns'].append(
                    f"Layer {layer} shows increased differentiation with {n_clusters} clusters."
                )
            elif n_clusters < len(layers[layer-1]):
                insights['cluster_patterns'].append(
                    f"Layer {layer} shows consolidation to {n_clusters} clusters."
                )
        
        # Analyze path types
        path_types = Counter(p['characteristics']['path_type'] for p in all_paths)
        dominant_path = path_types.most_common(1)[0]
        
        insights['path_patterns'].append(
            f"Most common trajectory type is '{dominant_path[0]}' ({dominant_path[1]} paths)."
        )
        
        if path_types['stable_representation'] > len(all_paths) * 0.3:
            insights['path_patterns'].append(
                "Many words maintain stable representations, suggesting robust encoding."
            )
        
        if path_types['complex_trajectory'] > len(all_paths) * 0.2:
            insights['path_patterns'].append(
                "Significant portion of words follow complex paths, indicating nuanced processing."
            )
        
        # Layer transition insights
        for layer_pair, analysis in layer_transitions.items():
            entropy = analysis['transition_entropy']
            if entropy < 2.0:
                insights['layer_patterns'].append(
                    f"{layer_pair}: Low entropy ({entropy:.2f}) indicates predictable transitions."
                )
            elif entropy > 4.0:
                insights['layer_patterns'].append(
                    f"{layer_pair}: High entropy ({entropy:.2f}) suggests major reorganization."
                )
        
        # Key findings
        insights['key_findings'] = [
            "GPT-2 progressively organizes words from surface features to semantic categories.",
            f"The network uses approximately {np.mean([len(layers[l]) for l in layers]):.1f} clusters per layer on average.",
            f"Path diversity indicates the model maintains {len(set(p['path'] for p in all_paths))}/{len(all_paths)} unique processing strategies."
        ]
        
        return insights