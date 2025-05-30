#!/usr/bin/env python3
"""
Find optimal k for each layer of GPT-2 token representations using gap statistic
and morphological/semantic purity metrics.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime
import nltk
from nltk.corpus import wordnet
import warnings
warnings.filterwarnings('ignore')

# Download WordNet if not available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PerLayerClusterAnalyzer:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / "clustering_results_per_layer"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load token analysis
        logging.info("Loading token analysis...")
        with open(base_dir / "full_token_analysis.json", 'r', encoding='utf-8') as f:
            token_list = json.load(f)
        
        # Create token_id to info mapping
        self.token_info = {}
        self.morphological_patterns = defaultdict(list)  # Track morphological patterns
        
        for token_data in token_list:
            token_id = token_data['token_id']
            token_str = token_data['token_str']
            
            self.token_info[token_id] = {
                'token': token_str,
                'type': token_data['token_type'],
                'has_space': token_data.get('has_leading_space', False),
                'is_alphabetic': token_data.get('is_alphabetic', False),
                'is_punctuation': token_data.get('is_punctuation', False),
                'is_numeric': token_data.get('is_numeric', False),
                'is_subword': token_data.get('is_subword', False),
                'language': token_data.get('likely_language', 'unknown'),
                'morphological_type': self._get_morphological_type(token_str),
                'wordnet_features': None  # Will be populated for complete words
            }
            
            # Track morphological patterns
            morph_type = self.token_info[token_id]['morphological_type']
            if morph_type:
                self.morphological_patterns[morph_type].append(token_id)
            
        logging.info(f"Loaded info for {len(self.token_info)} tokens")
        logging.info(f"Found {len(self.morphological_patterns)} morphological patterns")
        
        # Extract WordNet features for complete words
        self._extract_wordnet_features()
    
    def _get_morphological_type(self, token: str) -> Optional[str]:
        """Identify morphological pattern of token."""
        # Remove leading space for analysis
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
        elif token_clean.startswith('dis'):
            return 'prefix_dis'
        elif token_clean.startswith('non'):
            return 'prefix_non'
        
        return None
    
    def _extract_wordnet_features(self):
        """Extract WordNet features for complete words."""
        logging.info("Extracting WordNet features for complete words...")
        
        wordnet_count = 0
        for token_id, info in tqdm(self.token_info.items(), desc="WordNet extraction"):
            token = info['token'].strip()
            
            # Only process alphabetic tokens that might be complete words
            if info['is_alphabetic'] and not info['is_subword'] and len(token) > 2:
                synsets = wordnet.synsets(token.lower())
                
                if synsets:
                    # Get primary synset
                    primary_synset = synsets[0]
                    
                    # Extract hypernyms
                    hypernyms = []
                    for hypernym in primary_synset.hypernyms()[:3]:  # Top 3 hypernyms
                        hypernyms.append(hypernym.lemmas()[0].name())
                    
                    info['wordnet_features'] = {
                        'pos': primary_synset.pos(),
                        'definition': primary_synset.definition(),
                        'hypernyms': hypernyms,
                        'synset_count': len(synsets)
                    }
                    wordnet_count += 1
        
        logging.info(f"Extracted WordNet features for {wordnet_count} tokens")
    
    def load_all_activations(self) -> np.ndarray:
        """Load all activation chunks and concatenate them."""
        logging.info("Loading all activation chunks...")
        
        all_activations = []
        chunk_files = sorted((self.base_dir / "activations").glob("activations_chunk_*.npy"))
        
        for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
            chunk = np.load(chunk_file)
            all_activations.append(chunk)
        
        # Concatenate all chunks
        activations = np.vstack(all_activations)
        logging.info(f"Loaded activations shape: {activations.shape}")
        
        return activations
    
    def calculate_gap_statistic(self, data: np.ndarray, k: int, n_refs: int = 5) -> Tuple[float, float]:
        """Calculate gap statistic for given k."""
        import time
        
        n_samples = len(data)
        start_time = time.time()
        
        # Log clustering start
        logging.info(f"    Clustering {n_samples:,} tokens with k={k} clusters...")
        
        # For large k values, we can reduce n_init to speed up
        n_init = 3 if k > 100 else 5
        
        # Fit clustering on actual data
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init)
        labels = kmeans.fit_predict(data)
        
        cluster_time = time.time() - start_time
        logging.info(f"    Clustering completed in {cluster_time:.1f}s ({n_samples/cluster_time:.0f} tokens/sec)")
        
        # Calculate within-cluster sum of squares
        W_k = 0
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                center = cluster_points.mean(axis=0)
                W_k += np.sum((cluster_points - center) ** 2)
        
        # Generate reference datasets
        n_samples, n_features = data.shape
        ref_W_ks = []
        
        # Get data range for uniform distribution
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        
        np.random.seed(42)
        for ref_idx in range(n_refs):
            # Generate uniform random data in same range
            ref_data = np.random.uniform(
                data_min, data_max, 
                size=(n_samples, n_features)
            )
            
            # Cluster reference data
            ref_kmeans = KMeans(n_clusters=k, random_state=42 + ref_idx, n_init=3)
            ref_labels = ref_kmeans.fit_predict(ref_data)
            
            # Calculate W_k for reference
            ref_W_k = 0
            for i in range(k):
                ref_cluster = ref_data[ref_labels == i]
                if len(ref_cluster) > 0:
                    ref_center = ref_cluster.mean(axis=0)
                    ref_W_k += np.sum((ref_cluster - ref_center) ** 2)
            
            ref_W_ks.append(ref_W_k)
        
        # Calculate gap statistic
        ref_W_ks = np.array(ref_W_ks)
        gap = np.mean(np.log(ref_W_ks + 1)) - np.log(W_k + 1)  # Add 1 to avoid log(0)
        gap_std = np.std(np.log(ref_W_ks + 1))
        
        return gap, gap_std
    
    def calculate_morphological_purity(self, labels: np.ndarray) -> Dict[str, float]:
        """Calculate purity scores based on morphological patterns."""
        cluster_morphs = defaultdict(lambda: defaultdict(int))
        
        # Count morphological patterns per cluster
        for token_id, cluster_id in enumerate(labels):
            if token_id in self.token_info:
                morph_type = self.token_info[token_id]['morphological_type']
                if morph_type:
                    cluster_morphs[cluster_id][morph_type] += 1
        
        # Calculate purity scores
        purity_scores = {}
        total_morphological_tokens = 0
        pure_morphological_tokens = 0
        
        for cluster_id, morph_counts in cluster_morphs.items():
            if morph_counts:
                total = sum(morph_counts.values())
                dominant = max(morph_counts.values())
                purity = dominant / total
                purity_scores[f'cluster_{cluster_id}'] = purity
                
                total_morphological_tokens += total
                pure_morphological_tokens += dominant
        
        # Overall morphological purity
        overall_purity = pure_morphological_tokens / total_morphological_tokens if total_morphological_tokens > 0 else 0
        
        return {
            'overall_morphological_purity': overall_purity,
            'cluster_purities': purity_scores,
            'total_morphological_tokens': total_morphological_tokens
        }
    
    def calculate_semantic_purity(self, labels: np.ndarray) -> Dict[str, float]:
        """Calculate semantic purity for tokens with WordNet features."""
        cluster_pos = defaultdict(lambda: defaultdict(int))
        cluster_hypernyms = defaultdict(lambda: defaultdict(int))
        
        # Count POS tags and hypernyms per cluster
        for token_id, cluster_id in enumerate(labels):
            if token_id in self.token_info:
                info = self.token_info[token_id]
                if info['wordnet_features']:
                    pos = info['wordnet_features']['pos']
                    cluster_pos[cluster_id][pos] += 1
                    
                    for hypernym in info['wordnet_features']['hypernyms']:
                        cluster_hypernyms[cluster_id][hypernym] += 1
        
        # Calculate POS purity
        pos_purity_scores = {}
        total_wordnet_tokens = 0
        pure_pos_tokens = 0
        
        for cluster_id, pos_counts in cluster_pos.items():
            if pos_counts:
                total = sum(pos_counts.values())
                dominant = max(pos_counts.values())
                purity = dominant / total
                pos_purity_scores[f'cluster_{cluster_id}'] = purity
                
                total_wordnet_tokens += total
                pure_pos_tokens += dominant
        
        overall_pos_purity = pure_pos_tokens / total_wordnet_tokens if total_wordnet_tokens > 0 else 0
        
        return {
            'overall_pos_purity': overall_pos_purity,
            'cluster_pos_purities': pos_purity_scores,
            'total_wordnet_tokens': total_wordnet_tokens
        }

    def find_optimal_k_adaptive(self, layer_activations: np.ndarray, layer_idx: int, 
                               k_min: int = 2, k_max: int = 1000) -> Tuple[int, Dict]:
        """
        Three-phase adaptive search for optimal k using gap statistic elbow method.
        
        Strategy:
        1. Coarse sampling to find general regions of interest
        2. Refined search in promising regions
        3. Focused search around the optimal candidate
        """
        import time
        logging.info(f"\nFinding optimal k for layer {layer_idx} using three-phase adaptive search...")
        
        # Phase 1: Coarse exponential-like sampling
        # Start with strategic k values that cover the range well
        k_coarse = [2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
        k_coarse = [k for k in k_coarse if k_min <= k <= k_max]
        
        logging.info(f"Phase 1: Coarse sampling with {len(k_coarse)} k values: {k_coarse}")
        
        # Calculate metrics for coarse grid
        gap_scores = {}
        gap_stds = {}
        silhouette_scores = {}
        all_labels = {}
        elbow_candidates = []
        
        phase1_start = time.time()
        
        for i, k in enumerate(tqdm(k_coarse, desc="Phase 1: Coarse search")):
            k_start = time.time()
            
            # Log progress with time estimates
            if i > 0:
                elapsed = time.time() - phase1_start
                avg_time_per_k = elapsed / i
                remaining_ks = len(k_coarse) - i
                eta = remaining_ks * avg_time_per_k
                logging.info(f"  Testing k={k} ({i+1}/{len(k_coarse)}) - ETA: {eta/60:.1f} min")
            
            # Calculate gap statistic with moderate n_refs for speed
            gap, std = self.calculate_gap_statistic(layer_activations, k, n_refs=5)
            gap_scores[k] = gap
            gap_stds[k] = std
            
            # Get labels and calculate other metrics
            kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
            labels = kmeans.fit_predict(layer_activations)
            all_labels[k] = labels
            
            # Silhouette on sample for speed
            sample_size = min(5000, len(layer_activations))
            sample_idx = np.random.choice(len(layer_activations), sample_size, replace=False)
            silhouette_scores[k] = silhouette_score(
                layer_activations[sample_idx], labels[sample_idx]
            )
            
            k_time = time.time() - k_start
            logging.info(f"  k={k}: gap={gap:.4f}, std={std:.4f}, silhouette={silhouette_scores[k]:.4f} (took {k_time:.1f}s)")
            
            # Check for potential elbow points
            if i > 0:
                k_prev = k_coarse[i-1]
                gap_prev = gap_scores[k_prev]
                # Look for significant drops in gap statistic
                if gap_prev >= gap - std:
                    elbow_candidates.append(k_prev)
                    logging.info(f"  Potential elbow detected at k={k_prev}")
        
        # Find primary region of interest based on gap statistic patterns
        if elbow_candidates:
            # Use the first significant elbow
            primary_elbow = elbow_candidates[0]
        else:
            # If no clear elbow, find k with highest gap score
            primary_elbow = max(gap_scores.keys(), key=lambda k: gap_scores[k])
            logging.info(f"No clear elbow in coarse search, using k with highest gap: {primary_elbow}")
        
        # Phase 2: Refine around regions of interest
        # Define search regions based on coarse results
        regions_to_refine = []
        
        # Region around primary elbow
        regions_to_refine.append((
            max(k_min, int(primary_elbow * 0.7)),
            min(k_max, int(primary_elbow * 1.3))
        ))
        
        # If gap scores suggest multiple plateaus, add those regions
        gap_values = [gap_scores[k] for k in k_coarse]
        if len(gap_values) > 3:
            # Look for secondary peaks/plateaus
            for i in range(1, len(gap_values) - 1):
                if gap_values[i] > gap_values[i-1] and gap_values[i] > gap_values[i+1]:
                    secondary_k = k_coarse[i]
                    if abs(secondary_k - primary_elbow) > primary_elbow * 0.5:
                        regions_to_refine.append((
                            max(k_min, int(secondary_k * 0.8)),
                            min(k_max, int(secondary_k * 1.2))
                        ))
        
        logging.info(f"\nPhase 2: Refining search in {len(regions_to_refine)} regions")
        
        # Refine each region
        refined_results = {}
        phase2_start = time.time()
        
        for region_idx, (region_min, region_max) in enumerate(regions_to_refine):
            logging.info(f"  Region {region_idx+1}: [{region_min}, {region_max}]")
            
            # Determine step size for this region
            region_size = region_max - region_min
            n_points = min(15, max(5, region_size // 10))
            step = max(1, region_size // n_points)
            
            k_values_region = list(range(region_min, region_max + 1, step))
            
            for k in tqdm(k_values_region, desc=f"Phase 2: Region {region_idx+1}"):
                if k not in gap_scores:
                    # Calculate with more references for better accuracy
                    gap, std = self.calculate_gap_statistic(layer_activations, k, n_refs=7)
                    gap_scores[k] = gap
                    gap_stds[k] = std
                    
                    # Get clustering
                    kmeans = KMeans(n_clusters=k, n_init=7, random_state=42)
                    labels = kmeans.fit_predict(layer_activations)
                    all_labels[k] = labels
                    
                    # Better silhouette sample
                    sample_size = min(7500, len(layer_activations))
                    sample_idx = np.random.choice(len(layer_activations), sample_size, replace=False)
                    silhouette_scores[k] = silhouette_score(
                        layer_activations[sample_idx], labels[sample_idx]
                    )
                
                # Calculate purity metrics
                morph_purity = self.calculate_morphological_purity(all_labels[k])
                sem_purity = self.calculate_semantic_purity(all_labels[k])
                
                # Store refined results
                refined_results[k] = {
                    'gap': gap_scores[k],
                    'gap_std': gap_stds[k],
                    'silhouette': silhouette_scores[k],
                    'morphological_purity': morph_purity['overall_morphological_purity'],
                    'semantic_purity': sem_purity['overall_pos_purity']
                }
        
        # Phase 3: Final optimization around best candidates
        logging.info("\nPhase 3: Final optimization around best candidates")
        
        # Identify top candidates based on multiple criteria
        candidates = []
        
        # 1. Best gap statistic
        best_gap_k = max(refined_results.keys(), key=lambda k: refined_results[k]['gap'])
        candidates.append(best_gap_k)
        
        # 2. Best silhouette score
        best_sil_k = max(refined_results.keys(), key=lambda k: refined_results[k]['silhouette'])
        if best_sil_k not in candidates:
            candidates.append(best_sil_k)
        
        # 3. Best morphological purity (if meaningful)
        morph_ks = [k for k in refined_results.keys() if refined_results[k]['morphological_purity'] > 0.1]
        if morph_ks:
            best_morph_k = max(morph_ks, key=lambda k: refined_results[k]['morphological_purity'])
            if best_morph_k not in candidates:
                candidates.append(best_morph_k)
        
        # 4. Elbow points from refined search
        sorted_ks = sorted(refined_results.keys())
        for i in range(len(sorted_ks) - 1):
            k_curr = sorted_ks[i]
            k_next = sorted_ks[i + 1]
            gap_curr = refined_results[k_curr]['gap']
            gap_next = refined_results[k_next]['gap']
            std_next = refined_results[k_next]['gap_std']
            
            if gap_curr >= gap_next - std_next and k_curr not in candidates:
                candidates.append(k_curr)
        
        logging.info(f"  Final candidates: {candidates}")
        
        # Detailed analysis of candidates
        best_k = None
        best_combined_score = -float('inf')
        detailed_results = {}
        
        for k in tqdm(candidates, desc="Phase 3: Final analysis"):
            # Ensure we have full analysis for each candidate
            if k not in all_labels:
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = kmeans.fit_predict(layer_activations)
                all_labels[k] = labels
            else:
                labels = all_labels[k]
            
            # Recalculate with higher accuracy if needed
            if k not in gap_scores or gap_stds[k] < 0.01:
                gap, std = self.calculate_gap_statistic(layer_activations, k, n_refs=10)
                gap_scores[k] = gap
                gap_stds[k] = std
            
            # Full silhouette calculation
            sample_size = min(10000, len(layer_activations))
            sample_idx = np.random.choice(len(layer_activations), sample_size, replace=False)
            full_silhouette = silhouette_score(
                layer_activations[sample_idx], labels[sample_idx]
            )
            
            # Complete purity analysis
            morph_purity = self.calculate_morphological_purity(labels)
            sem_purity = self.calculate_semantic_purity(labels)
            
            # Calculate combined score with balanced weights
            # Normalize scores to [0, 1] range for fair comparison
            norm_gap = gap_scores[k] / max(gap_scores.values()) if max(gap_scores.values()) > 0 else 0
            norm_sil = (full_silhouette + 1) / 2  # Silhouette is in [-1, 1]
            norm_morph = morph_purity['overall_morphological_purity']
            norm_sem = sem_purity['overall_pos_purity']
            
            # Adaptive weighting based on layer
            if layer_idx <= 3:
                # Early layers: emphasize morphological organization
                combined = (0.3 * norm_gap + 0.2 * norm_sil + 0.4 * norm_morph + 0.1 * norm_sem)
            elif layer_idx <= 7:
                # Middle layers: balanced approach
                combined = (0.35 * norm_gap + 0.25 * norm_sil + 0.25 * norm_morph + 0.15 * norm_sem)
            else:
                # Late layers: emphasize clustering quality
                combined = (0.4 * norm_gap + 0.3 * norm_sil + 0.15 * norm_morph + 0.15 * norm_sem)
            
            detailed_results[k] = {
                'gap': gap_scores[k],
                'gap_std': gap_stds[k],
                'silhouette': full_silhouette,
                'morphological_purity': morph_purity,
                'semantic_purity': sem_purity,
                'combined_score': combined,
                'normalized_scores': {
                    'gap': norm_gap,
                    'silhouette': norm_sil,
                    'morphological': norm_morph,
                    'semantic': norm_sem
                }
            }
            
            logging.info(f"  k={k}: combined_score={combined:.4f} "
                        f"(gap={norm_gap:.3f}, sil={norm_sil:.3f}, "
                        f"morph={norm_morph:.3f}, sem={norm_sem:.3f})")
            
            if combined > best_combined_score:
                best_combined_score = combined
                best_k = k
        
        # Final validation: ensure the chosen k makes sense
        if best_k == k_max and k_max > 500:
            logging.warning(f"Selected k={best_k} is at the upper bound. Checking if lower k is acceptable...")
            # Check if a lower k has nearly as good a score
            for k in sorted(detailed_results.keys(), reverse=True):
                if k < best_k * 0.8:
                    score_diff = best_combined_score - detailed_results[k]['combined_score']
                    if score_diff < 0.05:  # Within 5% of best score
                        logging.info(f"Using k={k} instead (score difference only {score_diff:.3f})")
                        best_k = k
                        best_combined_score = detailed_results[k]['combined_score']
                        break
        
        total_time = time.time() - phase1_start
        logging.info(f"\nOptimal k for layer {layer_idx}: {best_k} "
                    f"(combined score: {best_combined_score:.4f}, total time: {total_time/60:.1f} min)")
        
        # Prepare comprehensive results
        results = {
            'layer_idx': layer_idx,
            'optimal_k': best_k,
            'search_range': [k_min, k_max],
            'primary_elbow': primary_elbow,
            'candidates_analyzed': candidates,
            'best_combined_score': best_combined_score,
            'total_search_time': total_time,
            'coarse_search_results': {
                k: {
                    'gap': gap_scores.get(k),
                    'gap_std': gap_stds.get(k),
                    'silhouette': silhouette_scores.get(k)
                } for k in k_coarse
            },
            'refined_results': refined_results,
            'detailed_results': detailed_results,
            'optimal_labels': all_labels[best_k].tolist()
        }
        
        return best_k, results

    def find_optimal_k_for_layer(self, layer_activations: np.ndarray, k_values: List[int], layer_idx: int) -> Tuple[int, float, Dict]:
        """Find optimal k for a single layer using gap statistic and purity metrics."""
        logging.info(f"\nAnalyzing layer {layer_idx}...")
        
        results = {
            'layer_idx': layer_idx,
            'k_values': k_values,
            'silhouette_scores': {},
            'gap_statistics': {},
            'morphological_purities': {},
            'semantic_purities': {},
            'cluster_summaries': {},
            'cluster_labels': {}  # Store labels for trajectory analysis
        }
        
        best_k = None
        best_score = -float('inf')
        gap_values = []
        gap_stds = []
        
        for k in k_values:
            logging.info(f"  Testing k={k}...")
            
            # Use regular KMeans for gap statistic calculation
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(layer_activations)
            
            # Calculate metrics
            # 1. Silhouette score
            sample_size = min(5000, len(layer_activations))
            sample_idx = np.random.choice(len(layer_activations), sample_size, replace=False)
            silhouette = silhouette_score(layer_activations[sample_idx], labels[sample_idx])
            
            # 2. Gap statistic
            gap, gap_std = self.calculate_gap_statistic(layer_activations, k)
            gap_values.append(gap)
            gap_stds.append(gap_std)
            
            # 3. Morphological purity
            morph_purity = self.calculate_morphological_purity(labels)
            
            # 4. Semantic purity (for complete words)
            sem_purity = self.calculate_semantic_purity(labels)
            
            # Store results
            results['silhouette_scores'][k] = float(silhouette)
            results['gap_statistics'][k] = {'gap': float(gap), 'std': float(gap_std)}
            results['morphological_purities'][k] = morph_purity
            results['semantic_purities'][k] = sem_purity
            
            # Quick cluster analysis
            cluster_stats = self.analyze_clusters_quick(labels)
            results['cluster_summaries'][k] = cluster_stats
            
            # Combined score (weighted combination of metrics)
            combined_score = (
                0.3 * silhouette + 
                0.3 * gap + 
                0.2 * morph_purity['overall_morphological_purity'] +
                0.2 * sem_purity['overall_pos_purity']
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_k = k
                # Save best labels for trajectory analysis
                results['cluster_labels'][k] = labels.tolist()
            
            logging.info(f"    Silhouette: {silhouette:.4f}, Gap: {gap:.4f}, "
                        f"Morph purity: {morph_purity['overall_morphological_purity']:.4f}, "
                        f"Semantic purity: {sem_purity['overall_pos_purity']:.4f}")
        
        # Apply elbow method to gap statistic
        if len(gap_values) > 1:
            for i in range(len(gap_values) - 1):
                if gap_values[i] >= gap_values[i + 1] - gap_stds[i + 1]:
                    optimal_k_gap = k_values[i]
                    logging.info(f"  Gap statistic suggests k={optimal_k_gap}")
                    break
        
        results['optimal_k'] = best_k
        results['best_combined_score'] = float(best_score)
        
        logging.info(f"  Layer {layer_idx} optimal: k={best_k} (combined score={best_score:.4f})")
        
        return best_k, best_score, results
    
    def analyze_clusters_quick(self, labels: np.ndarray) -> Dict:
        """Quick cluster analysis for summary statistics."""
        cluster_types = defaultdict(lambda: Counter())
        
        for token_id, cluster_id in enumerate(labels):
            if token_id not in self.token_info:
                continue
            info = self.token_info[token_id]
            cluster_types[cluster_id][info['type']] += 1
        
        # Count specialized clusters
        specialized = 0
        total_clusters = len(cluster_types)
        
        for cluster_id, types in cluster_types.items():
            total = sum(types.values())
            if total > 0:
                dominant_count = types.most_common(1)[0][1]
                if dominant_count / total > 0.8:
                    specialized += 1
        
        return {
            'total_clusters': total_clusters,
            'specialized_clusters': specialized
        }
    
    def run_per_layer_analysis_adaptive(self, layers_to_analyze: List[int] = None, 
                                       k_min: int = 10, k_max: int = 1000):
        """Find optimal k for each layer using adaptive search."""
        import time
        
        # Load all activations
        all_activations = self.load_all_activations()
        num_layers = all_activations.shape[1]
        
        if layers_to_analyze is None:
            layers_to_analyze = list(range(num_layers))
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_tokens': all_activations.shape[0],
            'num_layers': num_layers,
            'search_method': 'adaptive',
            'k_range': [k_min, k_max],
            'layer_results': {},
            'optimal_k_per_layer': {},
            'best_combined_score_per_layer': {}
        }
        
        # Store labels for each layer's optimal clustering
        all_labels = {}
        
        # Track timing for ETA
        analysis_start = time.time()
        
        # Analyze each layer
        for i, layer_idx in enumerate(layers_to_analyze):
            layer_start = time.time()
            
            # Log layer progress with ETA
            if i > 0:
                elapsed = time.time() - analysis_start
                avg_time_per_layer = elapsed / i
                remaining_layers = len(layers_to_analyze) - i
                eta_minutes = (remaining_layers * avg_time_per_layer) / 60
                logging.info(f"\n{'='*60}")
                logging.info(f"LAYER {layer_idx} of {num_layers-1} | Progress: {i+1}/{len(layers_to_analyze)} | ETA: {eta_minutes:.1f} minutes")
                logging.info(f"{'='*60}")
            else:
                logging.info(f"\n{'='*60}")
                logging.info(f"LAYER {layer_idx} of {num_layers-1} | Starting analysis...")
                logging.info(f"{'='*60}")
            
            layer_activations = all_activations[:, layer_idx, :]
            
            # Use adaptive search
            optimal_k, layer_results = self.find_optimal_k_adaptive(
                layer_activations, layer_idx, k_min, k_max
            )
            
            layer_time = time.time() - layer_start
            logging.info(f"\nLayer {layer_idx} completed in {layer_time/60:.1f} minutes. Optimal k={optimal_k}")
            
            results['layer_results'][layer_idx] = layer_results
            results['optimal_k_per_layer'][layer_idx] = optimal_k
            results['best_combined_score_per_layer'][layer_idx] = layer_results['best_combined_score']
            
            # Save optimal labels for trajectory analysis
            optimal_labels = layer_results['optimal_labels']
            all_labels[layer_idx] = optimal_labels
            
            # Save labels to file for trajectory analysis
            np.save(self.results_dir / f"labels_layer{layer_idx}_k{optimal_k}.npy", 
                    np.array(optimal_labels))
        
        # Save all optimal labels together
        with open(self.results_dir / "optimal_labels_all_layers.json", 'w') as f:
            json.dump(all_labels, f)
        
        # Save results
        with open(self.results_dir / "per_layer_adaptive_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualizations
        self.plot_adaptive_results(results)
        
        # Generate report
        self.generate_adaptive_report(results)
        
        logging.info(f"\nSaved cluster labels for trajectory analysis in {self.results_dir}")
        
        return results
    
    def plot_per_layer_results(self, results: Dict):
        """Generate visualization plots for per-layer analysis."""
        layers = sorted(results['optimal_k_per_layer'].keys(), key=int)
        optimal_ks = [results['optimal_k_per_layer'][str(l)] for l in layers]
        best_scores = [results['best_silhouette_per_layer'][str(l)] for l in layers]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot 1: Optimal k per layer
        ax1.bar(layers, optimal_ks, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Optimal k')
        ax1.set_title('Optimal Number of Clusters per Layer')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (layer, k) in enumerate(zip(layers, optimal_ks)):
            ax1.text(i, k + 0.5, str(k), ha='center', va='bottom')
        
        # Plot 2: Best silhouette score per layer
        ax2.plot(layers, best_scores, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Best Silhouette Score')
        ax2.set_title('Clustering Quality per Layer')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Highlight early/middle/late layers
        ax2.axvspan(-0.5, 1.5, alpha=0.1, color='blue', label='Early layers')
        ax2.axvspan(4.5, 7.5, alpha=0.1, color='green', label='Middle layers')
        ax2.axvspan(9.5, 11.5, alpha=0.1, color='red', label='Late layers')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'optimal_k_per_layer.png', dpi=150)
        plt.close()
        
        # Additional plot: Silhouette scores heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap data
        k_values = results['k_values_tested']
        heatmap_data = []
        
        for layer in layers:
            layer_scores = []
            layer_results = results['layer_results'][str(layer)]
            for k in k_values:
                score = layer_results['silhouette_scores'].get(k, 0)
                layer_scores.append(score)
            heatmap_data.append(layer_scores)
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=-0.1, vmax=0.2)
        
        # Set ticks
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels(k_values)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f'Layer {l}' for l in layers])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Silhouette Score')
        
        # Add optimal k markers
        for i, (layer, opt_k) in enumerate(zip(layers, optimal_ks)):
            k_idx = k_values.index(opt_k)
            ax.plot(k_idx, i, 'r*', markersize=15)
        
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Layer')
        ax.set_title('Silhouette Scores Across Layers and k Values\n(Red stars indicate optimal k)')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'silhouette_heatmap.png', dpi=150)
        plt.close()
    
    def generate_report(self, results: Dict):
        """Generate comprehensive report of per-layer analysis."""
        report_lines = [
            "# GPT-2 Per-Layer Optimal Clustering Analysis with Morphological and Semantic Purity",
            f"\nAnalyzed {results['total_tokens']:,} tokens across {results['num_layers']} layers",
            f"K values tested: {results['k_values_tested']}",
            "\n## Methodology",
            "- Gap Statistic: Determines optimal number of clusters",
            "- Morphological Purity: Measures clustering of tokens with same morphological patterns (suffixes/prefixes)",
            "- Semantic Purity: Measures clustering of tokens with same WordNet POS tags and hypernyms",
            "- Combined Score: 0.3*silhouette + 0.3*gap + 0.2*morph_purity + 0.2*semantic_purity",
            "\n## Optimal k per Layer\n"
        ]
        
        # Create detailed table
        report_lines.append("| Layer | Optimal k | Silhouette | Gap Stat | Morph Purity | Semantic Purity | Interpretation |")
        report_lines.append("|-------|-----------|------------|----------|--------------|-----------------|----------------|")
        
        layers = sorted(results['optimal_k_per_layer'].keys(), key=int)
        
        for layer in layers:
            layer_str = str(layer)
            opt_k = results['optimal_k_per_layer'][layer_str]
            layer_results = results['layer_results'][layer_str]
            
            # Get metrics for optimal k
            silhouette = layer_results['silhouette_scores'][opt_k]
            gap = layer_results['gap_statistics'][opt_k]['gap']
            morph_purity = layer_results['morphological_purities'][opt_k]['overall_morphological_purity']
            sem_purity = layer_results['semantic_purities'][opt_k]['overall_pos_purity']
            
            # Add interpretation
            if int(layer) <= 2:
                interpretation = "Early: Token types & morphology"
            elif int(layer) <= 7:
                interpretation = "Middle: Syntactic patterns"
            else:
                interpretation = "Late: Complex linguistic features"
            
            report_lines.append(f"| {layer} | {opt_k} | {silhouette:.3f} | {gap:.3f} | "
                              f"{morph_purity:.3f} | {sem_purity:.3f} | {interpretation} |")
        
        # Add morphological pattern analysis
        report_lines.extend([
            "\n## Morphological Pattern Distribution",
            "\nTop morphological patterns found:"
        ])
        
        # Count total tokens per morphological pattern
        morph_counts = Counter()
        for pattern, token_ids in self.morphological_patterns.items():
            morph_counts[pattern] = len(token_ids)
        
        for pattern, count in morph_counts.most_common(10):
            report_lines.append(f"- {pattern}: {count:,} tokens")
        
        # Add summary statistics
        k_values = [results['optimal_k_per_layer'][str(l)] for l in layers]
        avg_k = np.mean(k_values)
        
        report_lines.extend([
            f"\n## Summary Statistics",
            f"- Average optimal k: {avg_k:.1f}",
            f"- Range of optimal k: {min(k_values)} - {max(k_values)}",
            f"- Total morphological tokens: {sum(morph_counts.values()):,}",
            f"- Total WordNet-annotated tokens: {sum(1 for info in self.token_info.values() if info['wordnet_features'])}",
            "\n## Key Findings\n"
        ])
        
        # Analyze trends across layer groups
        early_ks = [results['optimal_k_per_layer'][str(l)] for l in layers[:3]]
        middle_ks = [results['optimal_k_per_layer'][str(l)] for l in layers[3:8]]
        late_ks = [results['optimal_k_per_layer'][str(l)] for l in layers[8:]]
        
        # Calculate average purities per layer group
        early_morph = []
        middle_morph = []
        late_morph = []
        
        for layer in layers[:3]:
            opt_k = results['optimal_k_per_layer'][str(layer)]
            early_morph.append(results['layer_results'][str(layer)]['morphological_purities'][opt_k]['overall_morphological_purity'])
        
        for layer in layers[3:8]:
            opt_k = results['optimal_k_per_layer'][str(layer)]
            middle_morph.append(results['layer_results'][str(layer)]['morphological_purities'][opt_k]['overall_morphological_purity'])
        
        for layer in layers[8:]:
            opt_k = results['optimal_k_per_layer'][str(layer)]
            late_morph.append(results['layer_results'][str(layer)]['morphological_purities'][opt_k]['overall_morphological_purity'])
        
        report_lines.extend([
            f"1. **Early layers (0-2)**: Average k = {np.mean(early_ks):.1f}, Avg morph purity = {np.mean(early_morph):.3f}",
            f"   - Focus on basic token categorization and morphological patterns",
            f"   - High morphological purity suggests tokens cluster by suffix/prefix patterns",
            f"2. **Middle layers (3-7)**: Average k = {np.mean(middle_ks):.1f}, Avg morph purity = {np.mean(middle_morph):.3f}",
            f"   - Develop syntactic and grammatical patterns",
            f"   - Morphological patterns may merge into broader grammatical categories",
            f"3. **Late layers (8-11)**: Average k = {np.mean(late_ks):.1f}, Avg morph purity = {np.mean(late_morph):.3f}",
            f"   - Refine complex linguistic features",
            f"   - Lower morphological purity may indicate semantic/contextual organization",
            "\n## Implications for Full Vocabulary Analysis",
            "",
            "1. **Subword Organization**: The morphological purity scores reveal how GPT-2 groups",
            "   tokens with similar morphological patterns (e.g., all '-ing' endings, all '-ed' endings).",
            "",
            "2. **BPE Artifacts**: The clustering patterns show how byte-pair encoding influences",
            "   the organization of the vocabulary in representation space.",
            "",
            "3. **Semantic vs Morphological**: The balance between semantic and morphological purity",
            "   across layers reveals the transition from surface-level patterns to deeper linguistic understanding.",
            "",
            "4. **Full Vocabulary Insights**: Unlike the single-token word analysis in the paper,",
            "   this analysis reveals how GPT-2 organizes its entire 50,257 token vocabulary,",
            "   including subwords, punctuation, numbers, and special tokens."
        ])
        
        with open(self.results_dir / "per_layer_report.md", 'w') as f:
            f.write('\n'.join(report_lines))
    
    def plot_adaptive_results(self, results: Dict):
        """Generate visualization plots for adaptive analysis results."""
        # For now, use the same plotting as regular results
        self.plot_per_layer_results(results)
    
    def generate_adaptive_report(self, results: Dict):
        """Generate report for adaptive analysis results."""
        # For now, use the same report generation
        self.generate_report(results)


def main():
    base_dir = Path(__file__).parent
    analyzer = PerLayerClusterAnalyzer(base_dir)
    
    # Analyze all 12 layers with adaptive k selection
    # k_min and k_max define the search range for 50k tokens
    # Start from k=2 to allow for simple binary splits
    results = analyzer.run_per_layer_analysis_adaptive(k_min=2, k_max=1000)
    
    logging.info(f"\nAnalysis complete! Results saved to {analyzer.results_dir}")
    
    # Print summary
    print("\nOptimal k per layer (using gap statistic + purity metrics):")
    for layer in sorted(results['optimal_k_per_layer'].keys(), key=int):
        layer_results = results['layer_results'][str(layer)]
        opt_k = results['optimal_k_per_layer'][layer]
        
        # Get metrics for optimal k
        if 'detailed_results' in layer_results and opt_k in layer_results['detailed_results']:
            morph_purity = layer_results['detailed_results'][opt_k]['morphological_purity']['overall_morphological_purity']
            sem_purity = layer_results['detailed_results'][opt_k]['semantic_purity']['overall_pos_purity']
            print(f"  Layer {layer}: k={opt_k} "
                  f"(morph_purity={morph_purity:.3f}, semantic_purity={sem_purity:.3f})")
        else:
            print(f"  Layer {layer}: k={opt_k}")


if __name__ == "__main__":
    main()