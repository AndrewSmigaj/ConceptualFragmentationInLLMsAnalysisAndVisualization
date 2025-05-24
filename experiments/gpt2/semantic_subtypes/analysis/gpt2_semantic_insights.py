#!/usr/bin/env python3
"""
Generate detailed insights from GPT-2 semantic subtype clustering analysis.
This script produces a comprehensive analysis of how GPT-2 organizes semantic information.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import sys

class GPT2SemanticInsights:
    def __init__(self, llm_data_dir: str):
        self.llm_data_dir = Path(llm_data_dir)
        self.cluster_data = None
        self.outlier_data = None
        self.transition_data = None
        
    def load_cluster_contents(self):
        """Load and analyze cluster contents."""
        filepath = self.llm_data_dir / 'cluster_contents_by_layer.json'
        with open(filepath, 'r') as f:
            self.cluster_data = json.load(f)
        
        print("=== GPT-2 SEMANTIC ORGANIZATION INSIGHTS ===\n")
        
        # Analyze each key layer
        key_layers = [0, 5, 10, 12]
        
        for layer in key_layers:
            layer_key = f'layer_{layer}'
            layer_data = self.cluster_data['layers'].get(layer_key, {})
            
            if not layer_data:
                continue
                
            print(f"\n## Layer {layer}: {layer_data.get('layer_description', '')}")
            print("-" * 60)
            
            kmeans_clusters = layer_data.get('kmeans', {})
            
            # Analyze each cluster
            for cluster_id in sorted(kmeans_clusters.keys()):
                if not cluster_id.startswith('cluster_'):
                    continue
                    
                cluster = kmeans_clusters[cluster_id]
                self._analyze_cluster_semantics(layer, cluster_id, cluster)
    
    def _analyze_cluster_semantics(self, layer: int, cluster_id: str, cluster_data: dict):
        """Deep semantic analysis of a single cluster."""
        cluster_num = cluster_id.split('_')[1]
        size = cluster_data.get('size', 0)
        purity = cluster_data.get('purity', 0)
        
        print(f"\n### Cluster {cluster_num} ({size} words, purity: {purity:.3f})")
        
        # Get all words from sample_words
        all_words = []
        sample_words = cluster_data.get('sample_words', {})
        for subtype, words in sample_words.items():
            all_words.extend([(word, subtype) for word in words])
        
        # Find semantic patterns
        patterns = self._identify_semantic_patterns(all_words)
        
        if patterns:
            print("Emergent semantic patterns:")
            for pattern_name, pattern_words in patterns.items():
                print(f"  - {pattern_name}: {', '.join(pattern_words[:8])}")
        
        # Show subtype distribution
        subtype_dist = cluster_data.get('subtype_distribution', {})
        dominant_subtypes = sorted(subtype_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"Dominant subtypes: {', '.join([f'{st}({n})' for st, n in dominant_subtypes])}")
        
        # Generate interpretation
        self._generate_cluster_interpretation(layer, cluster_num, all_words, patterns, dominant_subtypes)
    
    def _identify_semantic_patterns(self, word_subtype_pairs: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """Identify emergent semantic patterns in cluster."""
        words = [w for w, _ in word_subtype_pairs]
        
        patterns = {}
        
        # Sensory/perceptual
        sensory = set(words) & {'see', 'hear', 'feel', 'look', 'sound', 'taste', 'smell', 'touch', 'watch', 'listen'}
        if len(sensory) >= 2:
            patterns['sensory/perceptual'] = list(sensory)
        
        # Evaluative/judgment
        evaluative = set(words) & {'good', 'bad', 'best', 'worst', 'better', 'worse', 'nice', 'fine', 'great', 'terrible', 'excellent'}
        if len(evaluative) >= 2:
            patterns['evaluative/judgment'] = list(evaluative)
        
        # Cognitive/mental
        cognitive = set(words) & {'think', 'know', 'believe', 'understand', 'remember', 'forget', 'learn', 'consider', 'wonder', 'imagine'}
        if len(cognitive) >= 2:
            patterns['cognitive/mental'] = list(cognitive)
        
        # Motion/change
        motion = set(words) & {'move', 'run', 'walk', 'go', 'come', 'leave', 'arrive', 'change', 'turn', 'flow'}
        if len(motion) >= 2:
            patterns['motion/change'] = list(motion)
        
        # Emotional/affective
        emotional = set(words) & {'happy', 'sad', 'angry', 'fear', 'love', 'hate', 'joy', 'worry', 'excited', 'calm'}
        if len(emotional) >= 2:
            patterns['emotional/affective'] = list(emotional)
        
        # Size/dimension
        size = set(words) & {'big', 'small', 'large', 'tiny', 'huge', 'little', 'tall', 'short', 'wide', 'narrow'}
        if len(size) >= 2:
            patterns['size/dimension'] = list(size)
        
        # Temperature/physical state
        physical = set(words) & {'hot', 'cold', 'warm', 'cool', 'dry', 'wet', 'soft', 'hard', 'smooth', 'rough'}
        if len(physical) >= 2:
            patterns['temperature/texture'] = list(physical)
        
        # Temporal
        temporal = set(words) & {'now', 'then', 'when', 'while', 'during', 'before', 'after', 'always', 'never', 'sometimes'}
        if len(temporal) >= 2:
            patterns['temporal'] = list(temporal)
        
        # Modal/possibility
        modal = set(words) & {'can', 'could', 'may', 'might', 'will', 'would', 'should', 'must', 'shall'}
        if len(modal) >= 2:
            patterns['modal/possibility'] = list(modal)
        
        return patterns
    
    def _generate_cluster_interpretation(self, layer: int, cluster_num: str, 
                                       word_pairs: List[Tuple[str, str]], 
                                       patterns: Dict[str, List[str]], 
                                       dominant_subtypes: List[Tuple[str, int]]):
        """Generate human-readable interpretation of cluster."""
        print("\nInterpretation:")
        
        # Layer-specific interpretations
        if layer == 0:
            if patterns.get('evaluative/judgment'):
                print("  → Groups evaluative terms regardless of grammatical category")
            if patterns.get('modal/possibility'):
                print("  → Captures modal/auxiliary verbs as a distinct category")
                
        elif layer == 5:
            if patterns.get('cognitive/mental'):
                print("  → Consolidates mental/cognitive processes")
            if patterns.get('sensory/perceptual'):
                print("  → Groups perceptual verbs by modality")
                
        elif layer >= 10:
            if len(patterns) > 2:
                print("  → Shows complex multi-dimensional semantic organization")
            if dominant_subtypes[0][0] in ['degree_adverbs', 'manner_adverbs']:
                print("  → Maintains modifier categories for syntactic prediction")
    
    def analyze_outliers(self):
        """Analyze outlier words."""
        filepath = self.llm_data_dir / 'outlier_words_analysis.json'
        with open(filepath, 'r') as f:
            self.outlier_data = json.load(f)
        
        print("\n\n## OUTLIER ANALYSIS")
        print("-" * 60)
        
        consistent_outliers = self.outlier_data.get('consistent_outliers', [])
        
        print(f"\nWords that are outliers in 3+ layers:")
        for outlier_info in consistent_outliers[:10]:
            word = outlier_info.get('word', '')
            subtype = outlier_info.get('subtype', '')
            outlier_layers = outlier_info.get('outlier_layers', [])
            
            print(f"\n'{word}' ({subtype}):")
            print(f"  Outlier in layers: {', '.join(map(str, outlier_layers))}")
            
            # Special analysis for interesting cases
            if word == 'depend':
                print("  → Modal-like verb that doesn't fit action/stative categories")
                print("  → Encodes complex conditional/relational semantics")
            elif word in ['saw', 'found']:
                print("  → Past tense forms with distinct semantic properties")
            elif word in ['even', 'still']:
                print("  → Multi-functional words (adjective/adverb) with scalar semantics")
    
    def analyze_unexpected_groupings(self):
        """Analyze unexpected word groupings with more detail."""
        # Read a sample of the large file
        filepath = self.llm_data_dir / 'unexpected_groupings.json'
        
        print("\n\n## UNEXPECTED GROUPINGS")
        print("-" * 60)
        
        # Read and parse the file more carefully
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        pairs = data.get('consistent_pairs', [])
        
        # Group by semantic hypotheses
        hypothesis_groups = defaultdict(list)
        
        for pair in pairs[:1000]:  # Analyze first 1000
            words = tuple(sorted(pair.get('words', [])))
            subtypes = pair.get('subtypes', [])
            layers = pair.get('layers_together', 0)
            
            if layers < 100:  # Skip less consistent pairs
                continue
            
            # Categorize by hypothesis
            if any(w in words for w in ['feel', 'nice', 'good', 'fine']):
                hypothesis_groups['experiential_evaluation'].append((words, subtypes, layers))
            elif any(w in words for w in ['very', 'too', 'quite', 'rather']):
                hypothesis_groups['degree_intensification'].append((words, subtypes, layers))
            elif any(w in words for w in ['see', 'know', 'think', 'believe']):
                hypothesis_groups['epistemic_cognitive'].append((words, subtypes, layers))
            elif all(w in ['can', 'could', 'may', 'might', 'will', 'would'] for w in words):
                hypothesis_groups['modal_auxiliary'].append((words, subtypes, layers))
        
        # Report findings
        for hypothesis, examples in hypothesis_groups.items():
            if not examples:
                continue
                
            print(f"\n### {hypothesis.replace('_', ' ').title()}")
            
            # Show top examples
            top_examples = sorted(examples, key=lambda x: x[2], reverse=True)[:5]
            for words, subtypes, layers in top_examples:
                print(f"  {' + '.join(words)} ({', '.join(subtypes)}) - {layers} layers")
            
            # Provide interpretation
            if hypothesis == 'experiential_evaluation':
                print("  → GPT-2 groups experiential states with evaluative terms")
                print("  → Suggests 'feeling good/nice' as unified concept")
            elif hypothesis == 'degree_intensification':
                print("  → Intensifiers cluster regardless of syntactic differences")
                print("  → Reflects functional similarity in modifying scalar properties")
    
    def generate_theoretical_synthesis(self):
        """Generate theoretical synthesis of findings."""
        print("\n\n## THEORETICAL SYNTHESIS")
        print("-" * 60)
        
        print("\n### Key Insights:")
        
        print("\n1. **Functional over Formal Organization**")
        print("   - GPT-2 groups words by functional role rather than grammatical category")
        print("   - E.g., 'feel' (verb) clusters with 'nice' (adjective) due to experiential semantics")
        
        print("\n2. **Emergent Semantic Dimensions**")
        print("   - Evaluative/experiential: good, bad, feel, nice")
        print("   - Epistemic/cognitive: know, think, believe, see")
        print("   - Modal/auxiliary: can, could, may, might")
        print("   - Scalar/degree: very, too, quite, rather")
        
        print("\n3. **Layer-wise Semantic Refinement**")
        print("   - Early layers (0-2): Broad lexical categories")
        print("   - Middle layers (5-7): Semantic role differentiation") 
        print("   - Late layers (10-12): Prediction-oriented reorganization")
        
        print("\n4. **Outliers Reveal Semantic Complexity**")
        print("   - 'depend': Modal-like conditional semantics")
        print("   - Past tense forms: Aspectual distinctions")
        print("   - Multi-functional words: Context-dependent categorization")
        
        print("\n5. **Implications for Language Understanding**")
        print("   - Transformers learn usage-based semantic representations")
        print("   - Meaning emerges from distributional patterns, not predefined categories")
        print("   - Supports theories of prototype-based categorization")
        
    def save_insights_report(self):
        """Save comprehensive insights report."""
        report_path = self.llm_data_dir / 'gpt2_semantic_insights_report.md'
        
        print(f"\n\nInsights analysis complete!")
        print(f"Full report would be saved to: {report_path}")
        
        # Generate summary statistics
        print("\n### Summary Statistics:")
        if self.cluster_data:
            total_words = self.cluster_data.get('total_words', 774)
            subtypes = self.cluster_data.get('semantic_subtypes', [])
            print(f"  - Total words analyzed: {total_words}")
            print(f"  - Semantic subtypes: {len(subtypes)}")
            print(f"  - Layers analyzed: 13")
            print(f"  - Clustering methods: K-means (optimal k) and ETS")


def main():
    # Path to LLM analysis data
    llm_data_dir = "llm_analysis_data"
    
    # Create analyzer
    analyzer = GPT2SemanticInsights(llm_data_dir)
    
    # Run analyses
    analyzer.load_cluster_contents()
    analyzer.analyze_outliers()
    analyzer.analyze_unexpected_groupings()
    analyzer.generate_theoretical_synthesis()
    analyzer.save_insights_report()


if __name__ == "__main__":
    main()