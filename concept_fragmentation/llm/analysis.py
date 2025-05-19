"""
High-level API for LLM-based analysis of cluster paths.

This module provides functions for using LLMs to label clusters and generate
narratives for paths through the activation space.
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

from .factory import LLMClientFactory
from .responses import LLMResponse


class ClusterAnalysis:
    """
    High-level API for LLM-based analysis of clusters and paths.
    """
    
    def __init__(
        self,
        provider: str = "grok",
        model: str = "default",
        api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the ClusterAnalysis.
        
        Args:
            provider: The LLM provider to use (e.g., "openai", "claude", "grok")
            model: The model to use (default: provider's default model)
            api_key: The API key to use (if None, will try to get from environment)
            use_cache: Whether to cache LLM responses
            cache_dir: Directory to store cache files (default: ./cache/llm)
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        
        # Create the LLM client
        self.client = LLMClientFactory.create_client(
            provider=provider,
            model=model,
            api_key=api_key
        )
        
        # Set up caching
        self.use_cache = use_cache
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "cache", "llm")
        self.cache = {}
        
        # Create cache directory if it doesn't exist
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing cache if available
        if self.use_cache:
            self._load_cache()
    
    def _load_cache(self):
        """Load the response cache from disk."""
        cache_file = self._get_cache_file()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load cache file: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save the response cache to disk."""
        if not self.use_cache:
            return
        
        cache_file = self._get_cache_file()
        try:
            with open(cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to save cache file: {e}")
    
    def _get_cache_file(self) -> str:
        """Get the path to the cache file for the current provider and model."""
        return os.path.join(self.cache_dir, f"{self.provider}_{self.model}_cache.json")
    
    def _cache_key(self, prompt: str, **kwargs) -> str:
        """Generate a cache key for a prompt and parameters."""
        # Create a sorted string representation of kwargs
        kwargs_str = ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{prompt}|{kwargs_str}"
    
    async def _generate_with_cache(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text with caching.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Controls randomness (lower values are more deterministic)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            An LLMResponse object containing the generated text and metadata
        """
        # Check cache if enabled
        if self.use_cache:
            cache_key = self._cache_key(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                return LLMResponse.from_dict(cached_data)
        
        # Generate response from LLM
        response = await self.client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Cache the response if enabled
        if self.use_cache:
            cache_key = self._cache_key(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)
            self.cache[cache_key] = response.to_dict()
            self._save_cache()
        
        return response
    
    def generate_with_cache(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Synchronous wrapper for _generate_with_cache.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Controls randomness (lower values are more deterministic)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            An LLMResponse object containing the generated text and metadata
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._generate_with_cache(prompt, temperature, max_tokens, **kwargs)
        )
    
    async def label_cluster(
        self,
        cluster_centroid: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k_features: int = 10,
        other_clusters: Optional[List[Tuple[np.ndarray, str]]] = None
    ) -> str:
        """
        Generate a human-readable label for a cluster.
        
        Args:
            cluster_centroid: The centroid of the cluster
            feature_names: Optional names for the features
            top_k_features: Number of top features to include in the prompt
            other_clusters: Optional list of other cluster centroids and their labels
            
        Returns:
            A human-readable label for the cluster
        """
        # Get the top features for this cluster
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(len(cluster_centroid))]
        
        # Get indices of the top k features by absolute value
        top_indices = np.argsort(np.abs(cluster_centroid))[-top_k_features:][::-1]
        
        # Create a description of the top features
        feature_descriptions = []
        for idx in top_indices:
            feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            feature_value = cluster_centroid[idx]
            feature_descriptions.append(f"{feature_name}: {feature_value:.4f}")
        
        # Create the prompt
        prompt = f"""You are an AI expert analyzing neural network activations. 
        
Given the centroid of a cluster in activation space, provide a concise, meaningful label that captures the concept this cluster might represent.

Cluster centroid top {top_k_features} features:
{chr(10).join(feature_descriptions)}

"""
        
        # Add information about other clusters if provided
        if other_clusters:
            prompt += "\nOther clusters in the system with their labels:\n"
            for i, (_, label) in enumerate(other_clusters[:5]):  # Limit to 5 examples
                prompt += f"Cluster {i}: {label}\n"
            
            prompt += "\nProvide a label that is distinct from these existing labels but follows a similar naming convention.\n"
        
        prompt += "\nYour label should be concise (1-5 words) and interpretable. Focus on potential semantic meaning rather than technical details.\n\nCluster label:"
        
        # Generate the label
        response = await self._generate_with_cache(
            prompt=prompt,
            temperature=0.3,
            max_tokens=20
        )
        
        # Clean up the response
        label = response.text.strip().strip('"\'')
        
        # Truncate if too long
        if len(label) > 50:
            label = label[:47] + "..."
        
        return label
    
    async def label_clusters(
        self,
        cluster_centroids: Dict[str, np.ndarray],
        feature_names: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, str]:
        """
        Generate human-readable labels for multiple clusters.
        
        Args:
            cluster_centroids: Dictionary mapping cluster IDs to centroids
            feature_names: Optional dictionary mapping layer names to feature name lists
            
        Returns:
            Dictionary mapping cluster IDs to labels
        """
        labels = {}
        existing_labels = []
        
        # Process clusters layer by layer
        for cluster_id, centroid in cluster_centroids.items():
            # Determine the layer (extract from cluster ID format like "L1C0")
            layer = None
            if "L" in cluster_id and "C" in cluster_id:
                layer = cluster_id.split("C")[0]
            
            # Get feature names for this layer if available
            layer_feature_names = None
            if feature_names and layer and layer in feature_names:
                layer_feature_names = feature_names[layer]
            
            # Generate label for this cluster
            other_clusters_info = [(centroid, label) for cid, label in existing_labels]
            
            label = await self.label_cluster(
                centroid,
                feature_names=layer_feature_names,
                other_clusters=other_clusters_info
            )
            
            labels[cluster_id] = label
            existing_labels.append((centroid, label))
        
        return labels
    
    async def generate_path_narrative(
        self,
        path: List[str],
        cluster_labels: Dict[str, str],
        cluster_centroids: Dict[str, np.ndarray],
        convergent_points: Optional[List[Tuple[str, str, float]]] = None,
        fragmentation_score: Optional[float] = None,
        demographic_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a human-readable narrative for a path through activation space.
        
        Args:
            path: List of cluster IDs representing the path
            cluster_labels: Dictionary mapping cluster IDs to human-readable labels
            cluster_centroids: Dictionary mapping cluster IDs to centroids
            convergent_points: Optional list of convergent points in the path
                Format: [(early_cluster_id, late_cluster_id, similarity)]
            fragmentation_score: Optional fragmentation score for the path
            demographic_info: Optional demographic information about datapoints in this path
                
        Returns:
            A human-readable narrative for the path
        """
        # Create a description of the path
        path_description = []
        for cluster_id in path:
            label = cluster_labels.get(cluster_id, f"Unlabeled cluster {cluster_id}")
            path_description.append(f"{cluster_id} ({label})")
        
        # Create the prompt
        prompt = f"""You are an AI expert analyzing neural network activation patterns.
        
Generate a clear, insightful narrative that explains the following path through activation clusters in a neural network. 
Focus on the conceptual meaning and the potential decision process represented by this path.

Path: {" → ".join(path_description)}

"""
        
        # Add information about convergent points if provided
        if convergent_points:
            prompt += "\nConceptual convergence points in this path:\n"
            for early_id, late_id, similarity in convergent_points:
                early_label = cluster_labels.get(early_id, f"Unlabeled cluster {early_id}")
                late_label = cluster_labels.get(late_id, f"Unlabeled cluster {late_id}")
                prompt += f"- {early_id} ({early_label}) and {late_id} ({late_label}): {similarity:.2f} similarity\n"
        
        # Add fragmentation score if provided
        if fragmentation_score is not None:
            if fragmentation_score > 0.7:
                prompt += f"\nThis path has a high fragmentation score of {fragmentation_score:.2f}, indicating significant concept drift or fragmentation along the path."
            elif fragmentation_score < 0.3:
                prompt += f"\nThis path has a low fragmentation score of {fragmentation_score:.2f}, suggesting a relatively stable concept throughout the layers."
            else:
                prompt += f"\nThis path has a moderate fragmentation score of {fragmentation_score:.2f}."
        
        # Add demographic information if provided
        if demographic_info:
            prompt += "\n\nDemographic information about datapoints following this path:\n"
            for key, value in demographic_info.items():
                if isinstance(value, dict):
                    prompt += f"- {key}:\n"
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, float):
                            prompt += f"  - {subkey}: {subvalue:.1%}\n"
                        else:
                            prompt += f"  - {subkey}: {subvalue}\n"
                else:
                    prompt += f"- {key}: {value}\n"
        
        prompt += """
Based on this information, write a concise narrative (2-4 sentences) that explains:
1. What concepts or features this path might represent
2. How the concept evolves or transforms across layers (especially if there are convergence points)
3. Any potential insights about the model's decision-making process
4. If relevant, how demographic factors might relate to this path

Your explanation should be clear and insightful without being overly technical.

Path narrative:"""
        
        # Generate the narrative
        response = await self._generate_with_cache(
            prompt=prompt,
            temperature=0.4,
            max_tokens=250,
            system_prompt="You are an AI assistant that provides insightful, concise explanations of neural network behavior patterns."
        )
        
        # Return the narrative
        return response.text.strip()
    
    async def generate_path_narratives(
        self,
        paths: Dict[int, List[str]],
        cluster_labels: Dict[str, str],
        cluster_centroids: Dict[str, np.ndarray],
        convergent_points: Optional[Dict[int, List[Tuple[str, str, float]]]] = None,
        fragmentation_scores: Optional[Dict[int, float]] = None,
        demographic_info: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Dict[int, str]:
        """
        Generate human-readable narratives for multiple paths.
        
        Args:
            paths: Dictionary mapping path IDs to lists of cluster IDs
            cluster_labels: Dictionary mapping cluster IDs to human-readable labels
            cluster_centroids: Dictionary mapping cluster IDs to centroids
            convergent_points: Optional dictionary mapping path IDs to lists of convergent points
            fragmentation_scores: Optional dictionary mapping path IDs to fragmentation scores
            demographic_info: Optional dictionary mapping path IDs to demographic information
                
        Returns:
            Dictionary mapping path IDs to narratives
        """
        narratives = {}
        
        # Create a list of tasks for each path
        tasks = []
        for path_id, path in paths.items():
            # Get the convergent points for this path if available
            path_convergent_points = None
            if convergent_points and path_id in convergent_points:
                path_convergent_points = convergent_points[path_id]
            
            # Get the fragmentation score for this path if available
            path_fragmentation_score = None
            if fragmentation_scores and path_id in fragmentation_scores:
                path_fragmentation_score = fragmentation_scores[path_id]
            
            # Get the demographic info for this path if available
            path_demographic_info = None
            if demographic_info and path_id in demographic_info:
                path_demographic_info = demographic_info[path_id]
            
            # Create a task for generating the narrative
            task = self.generate_path_narrative(
                path=path,
                cluster_labels=cluster_labels,
                cluster_centroids=cluster_centroids,
                convergent_points=path_convergent_points,
                fragmentation_score=path_fragmentation_score,
                demographic_info=path_demographic_info
            )
            
            tasks.append((path_id, task))
        
        # Run all tasks in parallel
        for path_id, task in tasks:
            try:
                narrative = await task
                narratives[path_id] = narrative
            except Exception as e:
                print(f"Error generating narrative for path {path_id}: {e}")
                narratives[path_id] = f"Error generating narrative: {str(e)}"
        
        return narratives
    
    def label_clusters_sync(
        self,
        cluster_centroids: Dict[str, np.ndarray],
        feature_names: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, str]:
        """
        Synchronous wrapper for label_clusters.
        
        Args:
            cluster_centroids: Dictionary mapping cluster IDs to centroids
            feature_names: Optional dictionary mapping layer names to feature name lists
            
        Returns:
            Dictionary mapping cluster IDs to labels
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.label_clusters(cluster_centroids, feature_names)
        )
    
    def generate_path_narratives_sync(
        self,
        paths: Dict[int, List[str]],
        cluster_labels: Dict[str, str],
        cluster_centroids: Dict[str, np.ndarray],
        convergent_points: Optional[Dict[int, List[Tuple[str, str, float]]]] = None,
        fragmentation_scores: Optional[Dict[int, float]] = None,
        demographic_info: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Dict[int, str]:
        """
        Synchronous wrapper for generate_path_narratives.
        
        Args:
            paths: Dictionary mapping path IDs to lists of cluster IDs
            cluster_labels: Dictionary mapping cluster IDs to human-readable labels
            cluster_centroids: Dictionary mapping cluster IDs to centroids
            convergent_points: Optional dictionary mapping path IDs to lists of convergent points
            fragmentation_scores: Optional dictionary mapping path IDs to fragmentation scores
            demographic_info: Optional dictionary mapping path IDs to demographic information
                
        Returns:
            Dictionary mapping path IDs to narratives
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.generate_path_narratives(
                paths, cluster_labels, cluster_centroids,
                convergent_points, fragmentation_scores, demographic_info
            )
        )