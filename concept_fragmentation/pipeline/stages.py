"""
Common pipeline stages for neural network analysis.

This module provides reusable pipeline stages for common operations in
neural network analysis workflows, including activation processing,
clustering, path tracking, and visualization preparation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Generator
import logging
from dataclasses import dataclass
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from ..activation.collector import ActivationCollector, ActivationFormat
from ..activation.processor import ActivationProcessor
from ..activation.storage import ActivationStorage, StorageBackend
from .pipeline import PipelineStageBase, StreamingStage

# Setup logger
logger = logging.getLogger(__name__)


class ActivationCollectionStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """Pipeline stage for collecting activations from a model."""
    
    def __init__(
        self,
        collector: Optional[ActivationCollector] = None,
        streaming: bool = False,
        layer_names: Optional[List[str]] = None,
        store_to_disk: bool = False,
        output_path: Optional[str] = None,
        name: str = "ActivationCollection"
    ):
        """
        Initialize the activation collection stage.
        
        Args:
            collector: Activation collector to use
            streaming: Whether to use streaming mode
            layer_names: Specific layers to collect
            store_to_disk: Whether to store activations to disk
            output_path: Path to store activations to
            name: Name for this stage
        """
        super().__init__(name=name)
        self.collector = collector or ActivationCollector()
        self.streaming = streaming
        self.layer_names = layer_names
        self.store_to_disk = store_to_disk
        self.output_path = output_path
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect activations from a model.
        
        The input should be a dictionary with at least:
        - 'model': The PyTorch model to collect from
        - 'inputs': The input data for the model
        
        Optional keys:
        - 'model_id': Identifier for the model
        - 'split_name': Name of the data split
        - 'metadata': Additional metadata to store
        
        Returns a dictionary with the collected activations and metadata.
        """
        model = data.get('model')
        inputs = data.get('inputs')
        
        if model is None or inputs is None:
            raise ValueError("Input must contain 'model' and 'inputs' keys")
        
        model_id = data.get('model_id', 'default')
        split_name = data.get('split_name', 'unknown')
        metadata = data.get('metadata', {})
        
        # Register the model if not already registered
        if model_id not in self.collector.hooks:
            self.collector.register_model(
                model,
                model_id=model_id,
                activation_points=self.layer_names
            )
        
        # Collect activations
        if self.store_to_disk:
            output_path = self.collector.collect_and_store(
                model=model,
                inputs=inputs,
                output_path=self.output_path,
                model_id=model_id,
                split_name=split_name,
                streaming=self.streaming,
                metadata=metadata
            )
            
            return {
                'model': model,
                'inputs': inputs,
                'model_id': model_id,
                'split_name': split_name,
                'metadata': metadata,
                'activations_path': output_path
            }
        else:
            activations = self.collector.collect(
                model=model,
                inputs=inputs,
                model_id=model_id,
                streaming=self.streaming
            )
            
            return {
                'model': model,
                'inputs': inputs,
                'model_id': model_id,
                'split_name': split_name,
                'metadata': metadata,
                'activations': activations
            }
    
    def can_stream(self) -> bool:
        """Whether this stage can process data in a streaming fashion."""
        return self.streaming


class ActivationProcessingStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """Pipeline stage for processing activations."""
    
    def __init__(
        self,
        processor: Optional[ActivationProcessor] = None,
        dimensionality_reduction: bool = False,
        n_components: int = 50,
        normalization: bool = False,
        normalization_method: str = 'standard',
        name: str = "ActivationProcessing"
    ):
        """
        Initialize the activation processing stage.
        
        Args:
            processor: Activation processor to use
            dimensionality_reduction: Whether to perform dimensionality reduction
            n_components: Number of components for dimensionality reduction
            normalization: Whether to normalize activations
            normalization_method: Method to use for normalization
            name: Name for this stage
        """
        super().__init__(name=name)
        self.processor = processor or ActivationProcessor()
        
        # Add operations to the processor
        if dimensionality_reduction:
            self.processor.dimensionality_reduction(
                method='pca',
                n_components=n_components
            )
        
        if normalization:
            self.processor.normalize(
                method=normalization_method
            )
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process activations.
        
        The input should be a dictionary with either:
        - 'activations': Dictionary of activations
        - 'activations_path': Path to stored activations
        
        Returns a dictionary with the processed activations and other input data.
        """
        # Check if we need to load activations from disk
        if 'activations' not in data and 'activations_path' in data:
            storage = ActivationStorage()
            loaded_data = storage.load(data['activations_path'])
            activations = loaded_data.get('activations', {})
            data['activations'] = activations
        
        if 'activations' not in data:
            raise ValueError("Input must contain either 'activations' or 'activations_path' key")
        
        activations = data['activations']
        
        # Process activations
        processed = self.processor.process(activations)
        
        # Update the data dictionary
        result = data.copy()
        result['activations'] = processed
        
        return result


class ClusteringStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """Pipeline stage for clustering activations."""
    
    def __init__(
        self,
        max_k: int = 10,
        random_state: int = 42,
        compute_silhouette: bool = True,
        name: str = "Clustering"
    ):
        """
        Initialize the clustering stage.
        
        Args:
            max_k: Maximum number of clusters to try
            random_state: Random seed for reproducibility
            compute_silhouette: Whether to compute silhouette scores
            name: Name for this stage
        """
        super().__init__(name=name)
        self.max_k = max_k
        self.random_state = random_state
        self.compute_silhouette = compute_silhouette
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cluster activations.
        
        The input should be a dictionary with:
        - 'activations': Dictionary mapping layer names to activation arrays
        
        Returns a dictionary with the input data and added 'layer_clusters'.
        """
        if 'activations' not in data:
            raise ValueError("Input must contain 'activations' key")
        
        activations = data['activations']
        
        # Cluster each layer
        layer_clusters = {}
        for layer_name, layer_activations in activations.items():
            logger.info(f"Clustering layer {layer_name}...")
            k, centers, labels = self._compute_clusters_for_layer(
                layer_activations,
                max_k=self.max_k,
                random_state=self.random_state
            )
            
            logger.info(f"Found {k} clusters for layer {layer_name}")
            
            # Store cluster information
            layer_clusters[layer_name] = {
                "k": k,
                "centers": centers,
                "labels": labels,
                "activations": layer_activations
            }
            
            # Compute silhouette score if requested
            if self.compute_silhouette and k > 1 and np.min(np.bincount(labels)) > 1:
                try:
                    score = silhouette_score(layer_activations, labels)
                    layer_clusters[layer_name]["silhouette_score"] = score
                    logger.info(f"Silhouette score for layer {layer_name}: {score:.4f}")
                except Exception as e:
                    logger.warning(f"Error computing silhouette score for layer {layer_name}: {e}")
        
        # Update the data dictionary
        result = data.copy()
        result['layer_clusters'] = layer_clusters
        
        return result
    
    def _compute_clusters_for_layer(
        self,
        activations: np.ndarray, 
        max_k: int = 10, 
        random_state: int = 42
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Find optimal number of clusters for a layer's activations using silhouette score.
        
        Args:
            activations: Numpy array of activations (n_samples, n_features)
            max_k: Maximum number of clusters to try
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (optimal_k, cluster_centers, cluster_labels)
        """
        # Same logic as in cluster_paths.compute_clusters_for_layer
        best_k = 2
        best_score = -1.0
        best_labels = None
        best_centers = None
        
        # Try different k values
        for k in range(2, min(max_k, activations.shape[0]//2) + 1):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels = kmeans.fit_predict(activations)
            
            # Skip if we have clusters with only one point (silhouette undefined)
            # or if only one cluster is formed
            if len(set(labels)) < 2 or np.min(np.bincount(labels)) < 2:
                continue
                
            try:
                score = silhouette_score(activations, labels)
                if score > best_score:
                    best_k = k
                    best_score = score
                    best_labels = labels
                    best_centers = kmeans.cluster_centers_
            except Exception as e:
                logger.warning(f"Error computing silhouette for k={k}: {e}")
                continue
        
        # Fall back to k=2 if no valid clustering found
        if best_centers is None:
            # Attempt to force k=2 if possible
            if activations.shape[0] >= 4:
                num_clusters_fallback = 2
            elif activations.shape[0] >=2:
                num_clusters_fallback = 1
            else:
                return 0, np.array([]), np.array([])

            if num_clusters_fallback == 1 and activations.shape[0] > 0:
                best_labels = np.zeros(activations.shape[0], dtype=int)
                best_centers = np.mean(activations, axis=0, keepdims=True)
                best_k = 1
            elif num_clusters_fallback == 2:
                kmeans = KMeans(n_clusters=2, n_init=10, random_state=random_state)
                try:
                    best_labels = kmeans.fit_predict(activations)
                    # Check if k-means actually produced 2 clusters
                    if len(np.unique(best_labels)) < 2:
                        # If k-means collapsed to 1 cluster, set labels to 0s and center to mean
                        best_labels = np.zeros(activations.shape[0], dtype=int)
                        best_centers = np.mean(activations, axis=0, keepdims=True)
                        best_k = 1
                    else:
                        best_centers = kmeans.cluster_centers_
                        best_k = 2
                except Exception as e:
                    logger.warning(f"Error during fallback KMeans (k=2): {e}")
                    return 0, np.array([]), np.array([])
            else:
                return 0, np.array([]), np.array([])
        
        return best_k, best_centers, best_labels


class ClusterPathStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """Pipeline stage for computing cluster paths and related information."""
    
    def __init__(
        self,
        compute_similarity: bool = True,
        similarity_metric: str = 'cosine',
        min_similarity: float = 0.3,
        name: str = "ClusterPaths"
    ):
        """
        Initialize the cluster path stage.
        
        Args:
            compute_similarity: Whether to compute similarity matrix
            similarity_metric: Similarity metric to use
            min_similarity: Minimum similarity threshold
            name: Name for this stage
        """
        super().__init__(name=name)
        self.compute_similarity = compute_similarity
        self.similarity_metric = similarity_metric
        self.min_similarity = min_similarity
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute cluster paths and related information.
        
        The input should be a dictionary with:
        - 'layer_clusters': Dictionary of cluster information per layer
        
        Returns a dictionary with the input data and added path information.
        """
        if 'layer_clusters' not in data:
            raise ValueError("Input must contain 'layer_clusters' key")
        
        layer_clusters = data['layer_clusters']
        
        # Import necessary functions here to avoid circular imports
        from ..analysis.cluster_paths import (
            assign_unique_cluster_ids, compute_cluster_paths,
            compute_centroid_similarity, normalize_similarity_matrix,
            compute_layer_similarity_matrix, get_top_similar_clusters,
            identify_similarity_convergent_paths, compute_fragmentation_score
        )
        
        # Assign unique cluster IDs if not already done
        if not all("unique_labels" in layer_info for layer_info in layer_clusters.values()):
            layer_clusters, id_to_layer_cluster, cluster_to_unique_id = assign_unique_cluster_ids(layer_clusters)
        
        # Compute cluster paths
        unique_paths, layer_names, id_to_layer_cluster, original_paths, human_readable_paths = compute_cluster_paths(layer_clusters)
        
        # Compute similarity matrix if requested
        similarity_data = {}
        
        if self.compute_similarity:
            logger.info(f"Computing centroid similarity matrix using {self.similarity_metric} similarity...")
            similarity_matrix = compute_centroid_similarity(
                layer_clusters,
                id_to_layer_cluster,
                metric=self.similarity_metric,
                min_similarity=0.0  # Capture all similarities
            )
            
            normalized_matrix = normalize_similarity_matrix(similarity_matrix, metric=self.similarity_metric)
            
            layer_similarity = compute_layer_similarity_matrix(
                normalized_matrix,
                id_to_layer_cluster
            )
            
            top_similar_clusters = get_top_similar_clusters(
                normalized_matrix, 
                id_to_layer_cluster,
                top_k=5,
                min_similarity=self.min_similarity
            )
            
            convergent_paths = identify_similarity_convergent_paths(
                unique_paths,
                normalized_matrix,
                id_to_layer_cluster,
                min_similarity=self.min_similarity,
                max_layer_distance=None
            )
            
            logger.info("Computing fragmentation scores for all paths...")
            fragmentation_scores = np.zeros(len(unique_paths))
            for path_idx, path in enumerate(unique_paths):
                fragmentation_scores[path_idx] = compute_fragmentation_score(
                    path, normalized_matrix, id_to_layer_cluster
                )
            
            # Store similarity data
            similarity_data = {
                'raw_similarity': similarity_matrix,
                'normalized_similarity': normalized_matrix,
                'layer_similarity': layer_similarity,
                'top_similar_clusters': top_similar_clusters,
                'convergent_paths': convergent_paths,
                'fragmentation_scores': fragmentation_scores
            }
        
        # Update the data dictionary
        result = data.copy()
        result['paths'] = {
            'unique_paths': unique_paths,
            'layer_names': layer_names,
            'id_to_layer_cluster': id_to_layer_cluster,
            'original_paths': original_paths,
            'human_readable_paths': human_readable_paths
        }
        
        if similarity_data:
            result['similarity'] = similarity_data
        
        return result


class PathArchetypeStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """Pipeline stage for computing path archetypes."""
    
    def __init__(
        self,
        target_column: Optional[str] = None,
        demographic_columns: Optional[List[str]] = None,
        top_k: int = 5,
        max_members: int = 50,
        name: str = "PathArchetypes"
    ):
        """
        Initialize the path archetype stage.
        
        Args:
            target_column: Name of the target column
            demographic_columns: Columns to include in demographic statistics
            top_k: Number of most frequent paths to analyze
            max_members: Maximum number of member indices to include
            name: Name for this stage
        """
        super().__init__(name=name)
        self.target_column = target_column
        self.demographic_columns = demographic_columns
        self.top_k = top_k
        self.max_members = max_members
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute path archetypes.
        
        The input should be a dictionary with:
        - 'paths': Dictionary of path information
        - 'df': DataFrame with demographic information
        
        Returns a dictionary with the input data and added archetypes.
        """
        if 'paths' not in data:
            raise ValueError("Input must contain 'paths' key")
        
        paths_info = data['paths']
        df = data.get('df')
        
        # Can't compute archetypes without a dataframe
        if df is None:
            logger.warning("No dataframe provided, skipping archetype computation")
            return data
        
        # Import necessary functions here to avoid circular imports
        from ..analysis.cluster_paths import compute_path_archetypes
        
        # Extract path information
        unique_paths = paths_info['unique_paths']
        layer_names = paths_info['layer_names']
        id_to_layer_cluster = paths_info['id_to_layer_cluster']
        human_readable_paths = paths_info['human_readable_paths']
        
        # Determine dataset name
        dataset_name = data.get('dataset_name', 'unknown')
        
        # Compute archetypes
        archetypes = compute_path_archetypes(
            unique_paths,
            layer_names,
            df,
            dataset_name,
            id_to_layer_cluster=id_to_layer_cluster,
            human_readable_paths=human_readable_paths,
            target_column=self.target_column,
            demographic_columns=self.demographic_columns,
            top_k=self.top_k,
            max_members=self.max_members
        )
        
        # Update the data dictionary
        result = data.copy()
        result['path_archetypes'] = archetypes
        
        return result


class PersistenceStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """Pipeline stage for saving results to disk."""
    
    def __init__(
        self,
        output_path: str,
        format: str = 'json',
        save_activations: bool = False,
        name: str = "Persistence"
    ):
        """
        Initialize the persistence stage.
        
        Args:
            output_path: Path to save results to
            format: Format to save in ('json', 'pickle')
            save_activations: Whether to save activations
            name: Name for this stage
        """
        super().__init__(name=name)
        self.output_path = output_path
        self.format = format
        self.save_activations = save_activations
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save results to disk.
        
        Args:
            data: Dictionary of data to save
            
        Returns:
            The input data with an added 'saved_path' key
        """
        import os
        import json
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        
        # Remove activations if requested
        result_data = data.copy()
        if not self.save_activations and 'activations' in result_data:
            del result_data['activations']
        
        # Save the result
        if self.format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                else:
                    return obj
            
            # Convert to JSON-compatible types
            json_data = convert_numpy(result_data)
            
            with open(self.output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        elif self.format == 'pickle':
            with open(self.output_path, 'wb') as f:
                pickle.dump(result_data, f)
        
        else:
            raise ValueError(f"Unsupported format: {self.format}")
        
        logger.info(f"Saved results to {self.output_path}")
        
        # Add the saved path to the output
        data['saved_path'] = self.output_path
        
        return data


class LLMAnalysisStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """Pipeline stage for performing LLM-based analysis of clusters and paths."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        label_clusters: bool = True,
        generate_narratives: bool = True,
        top_k_paths: int = 5,
        name: str = "LLMAnalysis"
    ):
        """
        Initialize the LLM analysis stage.
        
        Args:
            provider: The LLM provider to use
            model: The model to use
            api_key: API key for the provider
            use_cache: Whether to cache LLM responses
            cache_dir: Directory to store cache files
            label_clusters: Whether to label clusters
            generate_narratives: Whether to generate path narratives
            top_k_paths: Number of most frequent paths to analyze
            name: Name for this stage
        """
        super().__init__(name=name)
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.label_clusters = label_clusters
        self.generate_narratives = generate_narratives
        self.top_k_paths = top_k_paths
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform LLM-based analysis.
        
        The input should be a dictionary with:
        - 'layer_clusters': Dictionary of cluster information
        - 'paths': Dictionary of path information
        - 'path_archetypes': List of path archetypes (optional)
        
        Returns a dictionary with the input data and added LLM analysis.
        """
        # Import here to avoid circular imports
        from ..llm.analysis import ClusterAnalysis
        
        # Create LLM client
        llm = ClusterAnalysis(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir
        )
        
        result = data.copy()
        
        # Label clusters if requested
        if self.label_clusters and 'layer_clusters' in data:
            logger.info("Generating cluster labels...")
            
            # Prepare cluster profiles
            cluster_profiles = {}
            
            if 'paths' in data and 'id_to_layer_cluster' in data['paths']:
                id_to_layer_cluster = data['paths']['id_to_layer_cluster']
                
                # For each unique cluster ID
                for unique_id, (layer_name, original_id, _) in id_to_layer_cluster.items():
                    cluster_id = str(unique_id)
                    
                    # Get cluster summary from layer clusters
                    if layer_name in data['layer_clusters']:
                        layer_info = data['layer_clusters'][layer_name]
                        
                        # Generate a textual profile for this cluster
                        profile_lines = []
                        profile_lines.append(f"Layer: {layer_name}")
                        
                        if 'centers' in layer_info and original_id < len(layer_info['centers']):
                            center = layer_info['centers'][original_id]
                            profile_lines.append(f"Center stats: min={np.min(center):.2f}, max={np.max(center):.2f}, mean={np.mean(center):.2f}")
                        
                        if 'labels' in layer_info:
                            labels = layer_info['labels']
                            member_count = np.sum(labels == original_id)
                            profile_lines.append(f"Members: {member_count} samples ({100 * member_count / len(labels):.1f}% of total)")
                        
                        # Add demographic statistics if available in archetypes
                        if 'path_archetypes' in data:
                            for archetype in data['path_archetypes']:
                                if 'numeric_path' in archetype and unique_id in archetype['numeric_path']:
                                    profile_lines.append(f"Demographic info from archetype {archetype.get('path', 'unknown')}:")
                                    
                                    for key, value in archetype.items():
                                        if key not in ['path', 'numeric_path', 'count', 'percentage', 'member_indices']:
                                            if isinstance(value, float):
                                                profile_lines.append(f"  {key}: {value:.2f}")
                                            elif isinstance(value, dict):
                                                profile_lines.append(f"  {key}: {value}")
                                            else:
                                                profile_lines.append(f"  {key}: {value}")
                                    
                                    break
                        
                        cluster_profiles[cluster_id] = "\n".join(profile_lines)
            
            # Generate labels if we have profiles
            if cluster_profiles:
                labels = llm.label_clusters_sync(cluster_profiles)
                result['cluster_labels'] = labels
                logger.info(f"Generated {len(labels)} cluster labels")
        
        # Generate path narratives if requested
        if self.generate_narratives and 'path_archetypes' in data and 'paths' in data:
            logger.info("Generating path narratives...")
            
            # Collect paths to analyze
            paths_to_analyze = {}
            archetypes = data['path_archetypes']
            
            # Sort archetypes by count
            sorted_archetypes = sorted(archetypes, key=lambda x: x.get('count', 0), reverse=True)
            
            # Take top-k archetypes
            for i, archetype in enumerate(sorted_archetypes[:self.top_k_paths]):
                if 'numeric_path' in archetype:
                    path_id = i
                    paths_to_analyze[path_id] = archetype['numeric_path']
            
            # Use existing cluster labels or dummy labels
            cluster_labels = result.get('cluster_labels', {})
            if not cluster_labels and 'paths' in data and 'id_to_layer_cluster' in data['paths']:
                # Create simple labels as fallback
                id_to_layer_cluster = data['paths']['id_to_layer_cluster']
                cluster_labels = {
                    str(unique_id): f"Cluster {original_id} in {layer_name}"
                    for unique_id, (layer_name, original_id, _) in id_to_layer_cluster.items()
                }
            
            # Extract demographic information for each path
            path_demographic_info = {}
            for path_id, path in paths_to_analyze.items():
                if path_id < len(sorted_archetypes):
                    archetype = sorted_archetypes[path_id]
                    demographics = {}
                    
                    for key, value in archetype.items():
                        if key not in ['path', 'numeric_path', 'count', 'percentage', 'member_indices']:
                            demographics[key] = value
                    
                    path_demographic_info[path_id] = demographics
            
            # Extract fragmentation scores if available
            fragmentation_scores = {}
            if 'similarity' in data and 'fragmentation_scores' in data['similarity']:
                scores = data['similarity']['fragmentation_scores']
                
                # Extract scores for the paths we're analyzing
                for path_id, path in paths_to_analyze.items():
                    # Find the index of this path in the unique paths
                    if 'paths' in data and 'unique_paths' in data['paths']:
                        unique_paths = data['paths']['unique_paths']
                        for i, unique_path in enumerate(unique_paths):
                            if np.array_equal(path, unique_path):
                                if i < len(scores):
                                    fragmentation_scores[path_id] = scores[i]
                                break
            
            # Extract convergent points if available
            convergent_points = {}
            if 'similarity' in data and 'convergent_paths' in data['similarity']:
                convergent_path_data = data['similarity']['convergent_paths']
                
                # Convert to our path IDs
                for path_id, path in paths_to_analyze.items():
                    # Find the index of this path in the unique paths
                    if 'paths' in data and 'unique_paths' in data['paths']:
                        unique_paths = data['paths']['unique_paths']
                        for i, unique_path in enumerate(unique_paths):
                            if np.array_equal(path, unique_path):
                                if str(i) in convergent_path_data:
                                    # Convert convergent points to the format expected by the LLM
                                    conv_points = []
                                    for conv in convergent_path_data[str(i)]:
                                        early_cluster = conv.get('early_cluster')
                                        late_cluster = conv.get('late_cluster')
                                        similarity = conv.get('similarity', 0.0)
                                        
                                        if early_cluster is not None and late_cluster is not None:
                                            conv_points.append((str(early_cluster), str(late_cluster), similarity))
                                    
                                    if conv_points:
                                        convergent_points[path_id] = conv_points
                                break
            
            # Generate narratives
            if paths_to_analyze and cluster_labels:
                narratives = llm.generate_path_narratives_sync(
                    paths_to_analyze,
                    cluster_labels,
                    convergent_points=convergent_points,
                    fragmentation_scores=fragmentation_scores,
                    path_demographic_info=path_demographic_info
                )
                
                result['path_narratives'] = narratives
                logger.info(f"Generated {len(narratives)} path narratives")
        
        return result