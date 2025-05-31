"""
LLM adapter for generic feedforward network analysis.
Adapts existing LLM infrastructure for non-demographic models.
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import json

# Add parent directory to path to import concept_fragmentation
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concept_fragmentation.llm.factory import LLMClientFactory
from concept_fragmentation.llm.cache_manager import CacheManager
from concept_fragmentation.llm.responses import LLMResponse

class GenericLLMAdapter:
    """
    Adapts LLM analysis for generic feedforward networks.
    Works with feature names and activation patterns rather than demographics.
    """
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize the LLM adapter.
        
        Args:
            provider: LLM provider (openai, anthropic, etc.)
            api_key: API key for the provider
        """
        self.provider = provider
        self.client = LLMClientFactory.create_client(
            provider=provider,
            api_key=api_key
        )
        self.cache_manager = CacheManager(
            provider=provider,
            model=self.client.model if hasattr(self.client, 'model') else 'default'
        )
    
    async def generate_cluster_label(
        self, 
        cluster_id: str,
        cluster_stats: Dict[str, Any],
        feature_names: List[str],
        layer_info: Dict[str, Any]
    ) -> str:
        """
        Generate a label for a cluster based on activation patterns.
        
        Args:
            cluster_id: Identifier for the cluster (e.g., "L4_C2")
            cluster_stats: Statistics about the cluster (size, activation patterns)
            feature_names: Names of input features
            layer_info: Information about the layer
            
        Returns:
            Human-readable cluster label
        """
        # Build prompt for cluster labeling
        prompt = self._build_cluster_label_prompt(
            cluster_id, cluster_stats, feature_names, layer_info
        )
        
        # Check cache
        cache_key = self.cache_manager._generate_cache_key(prompt)
        cached_response = self.cache_manager.get(prompt)
        
        if cached_response:
            return cached_response.text.strip()
        
        # Generate response
        response = await self.client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=50
        )
        
        # Cache the response
        self.cache_manager.store(prompt, response)
        
        label = response.text.strip().strip('"\'')
        return label[:50]  # Limit length
    
    def _build_cluster_label_prompt(
        self,
        cluster_id: str,
        cluster_stats: Dict[str, Any],
        feature_names: List[str],
        layer_info: Dict[str, Any]
    ) -> str:
        """Build prompt for cluster labeling."""
        # Extract key information
        layer_name = layer_info.get('name', cluster_id.split('_')[0])
        layer_type = layer_info.get('type', 'hidden')
        cluster_size = cluster_stats.get('size', 0)
        
        # Get top activated features (if available)
        top_features = []
        if 'feature_importance' in cluster_stats:
            importance = cluster_stats['feature_importance']
            sorted_features = sorted(
                enumerate(importance), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            top_features = [
                f"{feature_names[idx]} ({imp:.2f})"
                for idx, imp in sorted_features
                if idx < len(feature_names)
            ]
        
        prompt = f"""Analyze this neural network cluster and provide a concise semantic label (3-5 words).

Cluster: {cluster_id}
Layer: {layer_name} ({layer_type})
Size: {cluster_size} samples

Activation Pattern:
- Mean activation: {cluster_stats.get('mean_activation', 0):.3f}
- Std deviation: {cluster_stats.get('std_activation', 0):.3f}
"""
        
        if top_features:
            prompt += f"\nTop activated features:\n"
            for feat in top_features:
                prompt += f"- {feat}\n"
        
        prompt += "\nBased on the activation patterns and features, provide a semantic label that captures what this cluster represents:"
        
        return prompt
    
    async def generate_path_narrative(
        self,
        path: List[str],
        cluster_labels: Dict[str, str],
        path_stats: Dict[str, Any],
        feature_names: List[str]
    ) -> str:
        """
        Generate a narrative for a path through the network.
        
        Args:
            path: List of cluster IDs representing the path
            cluster_labels: Mapping of cluster IDs to labels
            path_stats: Statistics about the path
            feature_names: Names of input features
            
        Returns:
            Human-readable narrative
        """
        prompt = self._build_path_narrative_prompt(
            path, cluster_labels, path_stats, feature_names
        )
        
        # Check cache
        cached_response = self.cache_manager.get(prompt)
        if cached_response:
            return cached_response.text.strip()
        
        # Generate response
        response = await self.client.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=200
        )
        
        # Cache the response
        self.cache_manager.store(prompt, response)
        
        return response.text.strip()
    
    def _build_path_narrative_prompt(
        self,
        path: List[str],
        cluster_labels: Dict[str, str],
        path_stats: Dict[str, Any],
        feature_names: List[str]
    ) -> str:
        """Build prompt for path narrative generation."""
        # Create path description
        path_description = " → ".join([
            f"{cluster_id} ({cluster_labels.get(cluster_id, 'unlabeled')})"
            for cluster_id in path
        ])
        
        prompt = f"""Analyze this neural network activation path and provide an insightful narrative.

Path: {path_description}

Path Statistics:
- Frequency: {path_stats.get('frequency', 1)}
- Stability: {path_stats.get('stability', 0):.2f}
- Fragmentation: {path_stats.get('fragmentation', 0):.2f}
"""
        
        if 'dominant_features' in path_stats:
            prompt += f"\nDominant features for this path:\n"
            for feat_idx, importance in path_stats['dominant_features'][:5]:
                if feat_idx < len(feature_names):
                    prompt += f"- {feature_names[feat_idx]}: {importance:.2f}\n"
        
        prompt += """
Write a concise narrative (2-3 sentences) that explains:
1. What this path represents in terms of the neural network's processing
2. How the concept evolves through the layers
3. Any insights about the model's decision-making

Narrative:"""
        
        return prompt
    
    async def generate_cluster_card_content(
        self,
        cluster_id: str,
        cluster_data: Dict[str, Any],
        feature_names: List[str],
        layer_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate complete content for a cluster card.
        
        Args:
            cluster_id: Cluster identifier
            cluster_data: All data about the cluster
            feature_names: Names of features
            layer_info: Layer information
            
        Returns:
            Dictionary with card content
        """
        # Generate label
        label = await self.generate_cluster_label(
            cluster_id,
            cluster_data.get('stats', {}),
            feature_names,
            layer_info
        )
        
        # Calculate feature importance if not provided
        feature_importance = cluster_data.get('feature_importance', [])
        if not feature_importance and 'centroid' in cluster_data:
            # Use centroid values as proxy for importance
            centroid = cluster_data['centroid']
            if isinstance(centroid, list) and len(centroid) == len(feature_names):
                feature_importance = [
                    (feature_names[i], abs(val))
                    for i, val in enumerate(centroid)
                ]
                feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Build card content
        card_content = {
            'cluster_id': cluster_id,
            'label': label,
            'size': cluster_data.get('size', 0),
            'stats': cluster_data.get('stats', {}),
            'top_features': feature_importance[:5],
            'flows_to': cluster_data.get('flows_to', []),
            'flows_from': cluster_data.get('flows_from', [])
        }
        
        return card_content
    
    def generate_batch_labels(
        self,
        clusters: Dict[str, Dict[str, Any]],
        feature_names: List[str],
        layer_info: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Generate labels for multiple clusters (synchronous wrapper).
        
        Args:
            clusters: Dictionary of cluster_id -> cluster_data
            feature_names: List of feature names
            layer_info: Dictionary of layer information
            
        Returns:
            Dictionary of cluster_id -> label
        """
        import asyncio
        
        async def _generate_all():
            tasks = []
            for cluster_id, cluster_data in clusters.items():
                layer_idx = int(cluster_id.split('_')[0].replace('L', ''))
                layer = layer_info.get(f'layer_{layer_idx}', {})
                
                task = self.generate_cluster_label(
                    cluster_id,
                    cluster_data.get('stats', {}),
                    feature_names,
                    layer
                )
                tasks.append((cluster_id, task))
            
            results = {}
            for cluster_id, task in tasks:
                try:
                    label = await task
                    results[cluster_id] = label
                except Exception as e:
                    print(f"Error generating label for {cluster_id}: {e}")
                    results[cluster_id] = f"Cluster {cluster_id}"
            
            return results
        
        # Run async function
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(_generate_all())
    
    def analyze_feature_importance(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Analyze feature importance for each cluster.
        
        Args:
            activations: Activation array
            labels: Cluster labels
            feature_names: Feature names
            
        Returns:
            Dictionary mapping cluster ID to feature importance
        """
        importance_by_cluster = {}
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get activations for this cluster
            cluster_mask = labels == cluster_id
            cluster_activations = activations[cluster_mask]
            
            # Calculate mean activation per feature
            mean_activations = np.mean(cluster_activations, axis=0)
            
            # Normalize to get relative importance
            total = np.sum(np.abs(mean_activations))
            if total > 0:
                importance = np.abs(mean_activations) / total
            else:
                importance = np.zeros_like(mean_activations)
            
            # Create sorted list of (feature_name, importance)
            feature_importance = [
                (feature_names[i] if i < len(feature_names) else f"feature_{i}", 
                 float(importance[i]))
                for i in range(len(importance))
            ]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            importance_by_cluster[int(cluster_id)] = feature_importance
        
        return importance_by_cluster