#!/usr/bin/env python3
"""
Prepare LLM analysis data for all GPT-2 tokens experiment.

Focuses on:
1. Morphological coherence analysis
2. Grammatical highway identification
3. Token type segregation patterns
4. Subword semantic preservation
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Top10kLLMDataPreparer:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / "trajectory_analysis"
        self.llm_dir = base_dir / "llm_analysis"
        self.llm_dir.mkdir(exist_ok=True)
        
        # Load token information
        logging.info("Loading token information...")
        with open(base_dir / "top_10k_tokens_full.json", 'r', encoding='utf-8') as f:
            token_list = json.load(f)
        
        self.token_info = {}
        self.token_id_to_str = {}
        for token_data in token_list:
            token_id = token_data['token_id']
            self.token_id_to_str[token_id] = token_data['token_str']
            self.token_info[token_id] = token_data
    
    def load_trajectory_results(self):
        """Load the most recent trajectory analysis results."""
        result_files = list(self.results_dir.glob("trajectory_analysis_*.json"))
        if not result_files:
            raise FileNotFoundError("No trajectory analysis results found. Run trajectory analysis first.")
        
        latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
        logging.info(f"Loading trajectory results from {latest_file}")
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def prepare_contextual_role_analysis(self, trajectory_results: Dict) -> Dict:
        """
        Prepare data for contextual role analysis.
        Focus: How do grammatical functions cluster and evolve?
        """
        logging.info("Preparing contextual role analysis...")
        
        analysis_data = {
            "question": "How do grammatical functions cluster and evolve through GPT-2's layers?",
            "windows": {}
        }
        
        for window_name, window_data in trajectory_results['window_analysis'].items():
            window_analysis = {
                "total_paths": window_data['num_unique_paths'],
                "grammatical_patterns": [],
                "evolution_patterns": []
            }
            
            # Analyze top paths for grammatical patterns
            for path_info in window_data['archetypal_paths'][:10]:
                # Extract grammatical information
                morph_dist = path_info.get('morphological_distribution', {})
                
                # Identify if this path represents a grammatical function
                grammatical_indicator = None
                if morph_dist:
                    # Check for strong morphological patterns
                    for pattern, percentage in morph_dist.items():
                        if percentage > 50 and pattern != 'no_pattern':
                            grammatical_indicator = pattern
                            break
                
                path_analysis = {
                    "path": path_info['path'],
                    "token_count": path_info['count'],
                    "percentage": path_info['percentage'],
                    "grammatical_indicator": grammatical_indicator,
                    "dominant_type": self._get_dominant_type(path_info),
                    "example_tokens": path_info['example_tokens'][:5],
                    "morphological_breakdown": morph_dist
                }
                
                # Add WordNet POS if available
                if 'wordnet_analysis' in path_info and path_info['wordnet_analysis'].get('dominant_pos'):
                    path_analysis['dominant_pos'] = path_info['wordnet_analysis']['dominant_pos']
                
                window_analysis['grammatical_patterns'].append(path_analysis)
            
            # Analyze how paths change between windows
            if window_name == 'middle' and 'early' in analysis_data['windows']:
                window_analysis['evolution_from_early'] = self._analyze_evolution(
                    analysis_data['windows']['early']['grammatical_patterns'],
                    window_analysis['grammatical_patterns']
                )
            elif window_name == 'late' and 'middle' in analysis_data['windows']:
                window_analysis['evolution_from_middle'] = self._analyze_evolution(
                    analysis_data['windows']['middle']['grammatical_patterns'],
                    window_analysis['grammatical_patterns']
                )
            
            analysis_data['windows'][window_name] = window_analysis
        
        return analysis_data
    
    def prepare_subword_composition_analysis(self, trajectory_results: Dict) -> Dict:
        """
        Prepare data for subword composition pattern analysis.
        Focus: How do decomposable tokens organize?
        """
        logging.info("Preparing subword composition analysis...")
        
        analysis_data = {
            "question": "How do subword tokens organize based on their morphological structure?",
            "morphological_coherence": trajectory_results.get('morphological_trajectory_analysis', {}),
            "subword_patterns": {},
            "bpe_artifacts": []
        }
        
        # Analyze paths dominated by subwords
        for window_name, window_data in trajectory_results['window_analysis'].items():
            subword_paths = []
            
            for path_info in window_data['archetypal_paths']:
                type_dist = path_info.get('token_type_distribution', {})
                if type_dist.get('subword', 0) > 30:  # Paths with >30% subwords
                    subword_analysis = {
                        "path": path_info['path'],
                        "subword_percentage": type_dist.get('subword', 0),
                        "examples": path_info['example_tokens'][:10],
                        "morphological_patterns": path_info.get('morphological_distribution', {}),
                        "token_count": path_info['count']
                    }
                    
                    # Analyze BPE characteristics
                    bpe_chars = self._analyze_bpe_characteristics(path_info['token_ids'][:100])
                    subword_analysis['bpe_characteristics'] = bpe_chars
                    
                    subword_paths.append(subword_analysis)
            
            analysis_data['subword_patterns'][window_name] = subword_paths
        
        # Add specific BPE artifact analysis
        analysis_data['bpe_artifacts'] = self._identify_bpe_artifacts(trajectory_results)
        
        return analysis_data
    
    def prepare_semantic_preservation_analysis(self, trajectory_results: Dict) -> Dict:
        """
        Prepare data for semantic field preservation analysis.
        Focus: Do tokens maintain semantic relationships within grammatical highways?
        """
        logging.info("Preparing semantic preservation analysis...")
        
        analysis_data = {
            "question": "Do semantically related tokens follow similar trajectories within grammatical highways?",
            "semantic_coherence_by_window": {},
            "cross_type_patterns": []
        }
        
        # For each window, analyze semantic coherence within paths
        for window_name, window_data in trajectory_results['window_analysis'].items():
            semantic_patterns = []
            
            for path_info in window_data['archetypal_paths'][:15]:
                # Skip if no WordNet data
                if 'wordnet_analysis' not in path_info or path_info['wordnet_analysis']['coverage'] < 10:
                    continue
                
                semantic_analysis = {
                    "path": path_info['path'],
                    "wordnet_coverage": path_info['wordnet_analysis']['coverage'],
                    "dominant_pos": path_info['wordnet_analysis'].get('dominant_pos', []),
                    "common_hypernyms": path_info['wordnet_analysis'].get('common_hypernyms', []),
                    "token_examples": path_info['example_tokens'][:10],
                    "type_distribution": path_info.get('token_type_distribution', {})
                }
                
                # Analyze semantic diversity within path
                if semantic_analysis['common_hypernyms']:
                    semantic_analysis['semantic_coherence'] = self._calculate_semantic_coherence(
                        semantic_analysis['common_hypernyms']
                    )
                
                semantic_patterns.append(semantic_analysis)
            
            analysis_data['semantic_coherence_by_window'][window_name] = semantic_patterns
        
        # Analyze cross-type semantic preservation
        analysis_data['cross_type_patterns'] = self._analyze_cross_type_semantics(trajectory_results)
        
        return analysis_data
    
    def prepare_token_type_segregation_analysis(self, trajectory_results: Dict) -> Dict:
        """
        Prepare data for token type segregation analysis.
        Focus: How do different token types organize in representation space?
        """
        logging.info("Preparing token type segregation analysis...")
        
        analysis_data = {
            "question": "How do different token types (words, subwords, punctuation, numbers) segregate in GPT-2's representation space?",
            "type_specific_highways": {},
            "type_mixing_patterns": [],
            "space_prefix_analysis": {}
        }
        
        # Identify type-specific highways
        for window_name, window_data in trajectory_results['window_analysis'].items():
            type_highways = defaultdict(list)
            
            for path_info in window_data['archetypal_paths']:
                type_dist = path_info.get('token_type_distribution', {})
                
                # Find dominant type
                if type_dist:
                    dominant_type = max(type_dist.items(), key=lambda x: x[1])[0]
                    if type_dist[dominant_type] > 70:  # Strong type dominance
                        type_highways[dominant_type].append({
                            "path": path_info['path'],
                            "purity": type_dist[dominant_type],
                            "token_count": path_info['count'],
                            "examples": path_info['example_tokens'][:5]
                        })
            
            analysis_data['type_specific_highways'][window_name] = dict(type_highways)
            
            # Analyze mixed-type paths
            mixed_paths = []
            for path_info in window_data['archetypal_paths']:
                type_dist = path_info.get('token_type_distribution', {})
                if type_dist and max(type_dist.values()) < 50:  # No dominant type
                    mixed_paths.append({
                        "path": path_info['path'],
                        "type_distribution": type_dist,
                        "examples": path_info['example_tokens'][:10]
                    })
            
            if mixed_paths:
                analysis_data['type_mixing_patterns'].append({
                    "window": window_name,
                    "mixed_paths": mixed_paths[:5]
                })
        
        # Special analysis for space prefix tokens
        analysis_data['space_prefix_analysis'] = self._analyze_space_prefix_patterns(trajectory_results)
        
        return analysis_data
    
    def _get_dominant_type(self, path_info: Dict) -> str:
        """Get dominant token type for a path."""
        type_dist = path_info.get('token_type_distribution', {})
        if not type_dist:
            return 'unknown'
        return max(type_dist.items(), key=lambda x: x[1])[0]
    
    def _analyze_evolution(self, prev_patterns: List[Dict], curr_patterns: List[Dict]) -> Dict:
        """Analyze how grammatical patterns evolve between windows."""
        evolution = {
            "stable_patterns": [],
            "emerging_patterns": [],
            "disappearing_patterns": []
        }
        
        # Find patterns that remain stable
        prev_indicators = {p['grammatical_indicator'] for p in prev_patterns if p['grammatical_indicator']}
        curr_indicators = {p['grammatical_indicator'] for p in curr_patterns if p['grammatical_indicator']}
        
        evolution['stable_patterns'] = list(prev_indicators & curr_indicators)
        evolution['emerging_patterns'] = list(curr_indicators - prev_indicators)
        evolution['disappearing_patterns'] = list(prev_indicators - curr_indicators)
        
        return evolution
    
    def _analyze_bpe_characteristics(self, token_ids: List[int]) -> Dict:
        """Analyze BPE characteristics of tokens."""
        characteristics = {
            "avg_byte_length": 0,
            "incomplete_word_ratio": 0,
            "common_prefixes": [],
            "common_suffixes": []
        }
        
        if not token_ids:
            return characteristics
        
        byte_lengths = []
        incomplete_count = 0
        prefixes = Counter()
        suffixes = Counter()
        
        for tid in token_ids:
            if tid in self.token_info:
                info = self.token_info[tid]
                token_str = info['token_str']
                
                # Byte length
                byte_lengths.append(len(token_str.encode('utf-8')))
                
                # Check if incomplete word (doesn't start with space, not punctuation)
                if not token_str.startswith(' ') and info.get('is_alphabetic', False):
                    incomplete_count += 1
                
                # Extract prefix/suffix (first/last 2 chars)
                if len(token_str) >= 2:
                    prefixes[token_str[:2]] += 1
                    suffixes[token_str[-2:]] += 1
        
        if byte_lengths:
            characteristics['avg_byte_length'] = np.mean(byte_lengths)
            characteristics['incomplete_word_ratio'] = incomplete_count / len(token_ids)
            characteristics['common_prefixes'] = prefixes.most_common(3)
            characteristics['common_suffixes'] = suffixes.most_common(3)
        
        return characteristics
    
    def _identify_bpe_artifacts(self, trajectory_results: Dict) -> List[Dict]:
        """Identify specific BPE artifacts in the token organization."""
        artifacts = []
        
        # Look for paths dominated by incomplete words
        for window_name, window_data in trajectory_results['window_analysis'].items():
            for path_info in window_data['archetypal_paths'][:20]:
                # Check if path is dominated by BPE artifacts
                token_sample = path_info.get('token_ids', [])[:50]
                incomplete_ratio = sum(
                    1 for tid in token_sample 
                    if tid in self.token_info and 
                    not self.token_info[tid]['token_str'].startswith(' ') and
                    self.token_info[tid].get('is_alphabetic', False)
                ) / len(token_sample) if token_sample else 0
                
                if incomplete_ratio > 0.7:
                    artifacts.append({
                        "window": window_name,
                        "path": path_info['path'],
                        "incomplete_ratio": incomplete_ratio,
                        "examples": path_info['example_tokens'][:10],
                        "interpretation": "Path dominated by word fragments (BPE subwords)"
                    })
        
        return artifacts[:10]  # Top 10 artifacts
    
    def _calculate_semantic_coherence(self, hypernyms: List[Tuple[str, int]]) -> float:
        """Calculate semantic coherence based on hypernym distribution."""
        if not hypernyms:
            return 0.0
        
        # Higher concentration of few hypernyms = higher coherence
        total_count = sum(count for _, count in hypernyms)
        if total_count == 0:
            return 0.0
        
        # Calculate entropy-based coherence
        entropy = 0
        for _, count in hypernyms:
            p = count / total_count
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Normalize (lower entropy = higher coherence)
        max_entropy = np.log2(len(hypernyms))
        coherence = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
        
        return coherence
    
    def _analyze_cross_type_semantics(self, trajectory_results: Dict) -> List[Dict]:
        """Analyze semantic preservation across token types."""
        patterns = []
        
        # Look for paths that mix token types but maintain semantic coherence
        for window_name, window_data in trajectory_results['window_analysis'].items():
            for path_info in window_data['archetypal_paths']:
                type_dist = path_info.get('token_type_distribution', {})
                
                # Mixed type path
                if type_dist and 30 < max(type_dist.values()) < 70:
                    if 'wordnet_analysis' in path_info and path_info['wordnet_analysis'].get('common_hypernyms'):
                        patterns.append({
                            "window": window_name,
                            "path": path_info['path'],
                            "type_distribution": type_dist,
                            "semantic_theme": path_info['wordnet_analysis']['common_hypernyms'][0][0] if path_info['wordnet_analysis']['common_hypernyms'] else "unknown",
                            "examples": path_info['example_tokens'][:10]
                        })
        
        return patterns[:10]
    
    def _analyze_space_prefix_patterns(self, trajectory_results: Dict) -> Dict:
        """Analyze patterns specific to tokens with/without space prefixes."""
        analysis = {
            "space_prefix_separation": {},
            "interpretation": "How leading spaces affect token trajectories"
        }
        
        for window_name, window_data in trajectory_results['window_analysis'].items():
            space_patterns = {
                "with_space_dominant_paths": [],
                "without_space_dominant_paths": [],
                "mixed_space_paths": []
            }
            
            for path_info in window_data['archetypal_paths'][:20]:
                # Analyze space prefix distribution
                token_sample = path_info.get('token_ids', [])[:100]
                with_space = sum(
                    1 for tid in token_sample
                    if tid in self.token_info and 
                    self.token_info[tid]['token_str'].startswith(' ')
                )
                space_ratio = with_space / len(token_sample) if token_sample else 0
                
                path_summary = {
                    "path": path_info['path'],
                    "space_ratio": space_ratio,
                    "examples": path_info['example_tokens'][:5]
                }
                
                if space_ratio > 0.8:
                    space_patterns['with_space_dominant_paths'].append(path_summary)
                elif space_ratio < 0.2:
                    space_patterns['without_space_dominant_paths'].append(path_summary)
                elif 0.4 < space_ratio < 0.6:
                    space_patterns['mixed_space_paths'].append(path_summary)
            
            analysis['space_prefix_separation'][window_name] = space_patterns
        
        return analysis
    
    def prepare_complete_llm_analysis(self):
        """Prepare complete LLM analysis data package."""
        logging.info("Preparing complete LLM analysis data...")
        
        # Load trajectory results
        trajectory_results = self.load_trajectory_results()
        
        # Prepare all analyses
        llm_data = {
            "experiment_info": {
                "title": "GPT-2 Top 10k Tokens Trajectory Analysis",
                "description": "Analysis of how GPT-2 organizes its most common 10,000 tokens, focusing on frequent words and morphological patterns",
                "key_questions": [
                    "How do grammatical functions cluster and evolve?",
                    "How do subword tokens organize based on morphological structure?",
                    "Do tokens maintain semantic relationships within grammatical highways?",
                    "How do different token types segregate in representation space?"
                ],
                "timestamp": datetime.now().isoformat()
            },
            "analyses": {
                "contextual_roles": self.prepare_contextual_role_analysis(trajectory_results),
                "subword_composition": self.prepare_subword_composition_analysis(trajectory_results),
                "semantic_preservation": self.prepare_semantic_preservation_analysis(trajectory_results),
                "token_type_segregation": self.prepare_token_type_segregation_analysis(trajectory_results)
            },
            "summary_statistics": self._prepare_summary_statistics(trajectory_results)
        }
        
        # Save as JSON
        output_path = self.llm_dir / f"llm_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(llm_data, f, indent=2)
        
        logging.info(f"Saved LLM analysis data to {output_path}")
        
        # Also save as markdown for easy reading
        self._save_as_markdown(llm_data)
        
        return llm_data
    
    def _prepare_summary_statistics(self, trajectory_results: Dict) -> Dict:
        """Prepare summary statistics for the analysis."""
        stats = {
            "total_tokens": 10000,
            "windows": {}
        }
        
        for window_name, window_data in trajectory_results['window_analysis'].items():
            stats['windows'][window_name] = {
                "unique_paths": window_data['num_unique_paths'],
                "path_diversity": window_data['metrics']['diversity'],
                "top_path_concentration": window_data['metrics']['concentration'],
                "fragmentation": window_data['metrics']['fragmentation'],
                "entropy": window_data['metrics']['entropy']
            }
        
        # Add morphological statistics
        morph_analysis = trajectory_results.get('morphological_trajectory_analysis', {})
        if morph_analysis:
            high_coherence_patterns = [
                pattern for pattern, data in morph_analysis.items()
                if data.get('concentration', 0) > 0.5
            ]
            stats['morphological_patterns'] = {
                "total_patterns": len(morph_analysis),
                "high_coherence_patterns": len(high_coherence_patterns),
                "examples": high_coherence_patterns[:5]
            }
        
        return stats
    
    def _save_as_markdown(self, llm_data: Dict):
        """Save analysis as markdown for easy reading."""
        md_lines = [
            f"# {llm_data['experiment_info']['title']}",
            f"\n{llm_data['experiment_info']['description']}",
            f"\nGenerated: {llm_data['experiment_info']['timestamp']}",
            "\n## Key Research Questions\n"
        ]
        
        for q in llm_data['experiment_info']['key_questions']:
            md_lines.append(f"- {q}")
        
        # Add each analysis section
        for analysis_name, analysis_data in llm_data['analyses'].items():
            md_lines.append(f"\n## {analysis_name.replace('_', ' ').title()}")
            md_lines.append(f"\n**Question:** {analysis_data['question']}")
            
            # Format the data in a readable way
            if 'windows' in analysis_data:
                for window, window_data in analysis_data['windows'].items():
                    md_lines.append(f"\n### {window.capitalize()} Window")
                    md_lines.append(self._format_dict_as_markdown(window_data, indent=0))
            else:
                md_lines.append(self._format_dict_as_markdown(analysis_data, indent=0))
        
        # Add summary statistics
        md_lines.append("\n## Summary Statistics")
        md_lines.append(self._format_dict_as_markdown(llm_data['summary_statistics'], indent=0))
        
        output_path = self.llm_dir / f"llm_analysis_readable_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(output_path, 'w') as f:
            f.write('\n'.join(md_lines))
        
        logging.info(f"Saved readable markdown to {output_path}")
    
    def _format_dict_as_markdown(self, data: Any, indent: int = 0) -> str:
        """Recursively format dictionary as markdown."""
        if not isinstance(data, dict):
            return str(data)
        
        lines = []
        prefix = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}- **{key}:**")
                lines.append(self._format_dict_as_markdown(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}- **{key}:** ({len(value)} items)")
                for item in value[:3]:  # First 3 items
                    if isinstance(item, dict):
                        lines.append(self._format_dict_as_markdown(item, indent + 1))
                    else:
                        lines.append(f"{prefix}  - {item}")
                if len(value) > 3:
                    lines.append(f"{prefix}  - ... and {len(value) - 3} more")
            else:
                lines.append(f"{prefix}- **{key}:** {value}")
        
        return '\n'.join(lines)


def main():
    base_dir = Path(__file__).parent
    preparer = Top10kLLMDataPreparer(base_dir)
    
    # Prepare complete analysis
    llm_data = preparer.prepare_complete_llm_analysis()
    
    print("\nLLM Analysis Data Prepared!")
    print(f"Total analyses: {len(llm_data['analyses'])}")
    print("\nKey findings to investigate:")
    for analysis_name in llm_data['analyses']:
        print(f"- {analysis_name.replace('_', ' ').title()}")


if __name__ == "__main__":
    main()