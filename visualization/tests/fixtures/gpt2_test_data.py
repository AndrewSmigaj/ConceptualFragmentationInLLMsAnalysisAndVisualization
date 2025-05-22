"""
GPT-2 test data generators for visualization tests.

This module provides utilities to generate realistic mock data for testing
GPT-2 analysis and visualization components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import os
import json
from pathlib import Path


class GPT2TestDataGenerator:
    """Generate realistic mock data for GPT-2 visualization tests."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the test data generator.
        
        Args:
            seed: Random seed for reproducible test data
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Common test parameters
        self.default_model_type = "gpt2-small"
        self.default_feature_dim = 768
        self.default_num_heads = 12
        self.default_vocab_size = 50257
        
        # Sample tokens for testing
        self.sample_tokens = [
            "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", ".",
            "Hello", ",", " world", "!", " How", " are", " you", " today", "?",
            "This", " is", " a", " test", " of", " the", " GPT", "-", "2", " model", "."
        ]
        
        self.sample_token_ids = [
            464, 2068, 6282, 4796, 18045, 625, 262, 16931, 3290, 13,
            15496, 11, 995, 0, 1374, 389, 345, 1909, 30,
            1212, 318, 257, 1332, 286, 262, 402, 11571, 12, 17, 2746, 13
        ]
    
    def create_mock_analysis_results(
        self,
        num_layers: int = 4,
        seq_length: int = 8,
        batch_size: int = 1,
        num_clusters: int = 3,
        include_attention: bool = True,
        include_persistence: bool = True
    ) -> Dict[str, Any]:
        """
        Create comprehensive mock GPT-2 analysis results.
        
        Args:
            num_layers: Number of transformer layers
            seq_length: Sequence length for tokens
            batch_size: Batch size for processing
            num_clusters: Number of clusters per layer
            include_attention: Whether to include attention data
            include_persistence: Whether to include persistence metadata
            
        Returns:
            Complete mock analysis results dictionary
        """
        # Generate layer names
        layers = [f"layer_{i}" for i in range(num_layers)]
        
        # Create components
        activations = self.create_mock_activations(layers, seq_length, batch_size)
        token_metadata = self.create_mock_token_metadata(seq_length, batch_size)
        cluster_labels = self.create_mock_cluster_labels(layers, seq_length, batch_size, num_clusters)
        token_paths = self.create_mock_token_paths(layers, token_metadata, cluster_labels)
        cluster_metrics = self.create_mock_cluster_metrics(layers, num_clusters)
        similarity_data = self.create_mock_similarity_data(token_metadata["tokens"], layers)
        
        # Base analysis results
        analysis_results = {
            "model_type": self.default_model_type,
            "layers": layers,
            "activations": activations,
            "token_metadata": token_metadata,
            "cluster_labels": cluster_labels,
            "token_paths": token_paths,
            "cluster_metrics": cluster_metrics,
            "similarity": similarity_data,
            "path_archetypes": self.create_mock_path_archetypes(token_paths, layers)
        }
        
        # Add attention data if requested
        if include_attention:
            attention_data = self.create_mock_attention_data(layers, seq_length, batch_size)
            analysis_results["attention_data"] = attention_data
        
        # Add persistence metadata if requested
        if include_persistence:
            analysis_results.update({
                "input_text": " ".join(token_metadata["tokens"][0]),
                "analysis_timestamp": "2023-01-01T12:00:00",
                "config": {
                    "num_layers": num_layers,
                    "seq_length": seq_length,
                    "batch_size": batch_size,
                    "num_clusters": num_clusters
                }
            })
        
        return analysis_results
    
    def create_mock_activations(
        self,
        layers: List[str],
        seq_length: int,
        batch_size: int,
        feature_dim: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate mock activation tensors.
        
        Args:
            layers: List of layer names
            seq_length: Sequence length
            batch_size: Batch size
            feature_dim: Feature dimension (default: 768 for GPT-2 small)
            
        Returns:
            Dictionary mapping layer names to activation arrays
        """
        if feature_dim is None:
            feature_dim = self.default_feature_dim
        
        activations = {}
        
        for layer in layers:
            # Generate realistic activation patterns
            # Use different distributions for different layers
            layer_idx = int(layer.split("_")[-1])
            
            # Early layers have more varied activations
            if layer_idx < 2:
                base_activations = np.random.normal(0, 1, (batch_size, seq_length, feature_dim))
            else:
                # Later layers have more focused patterns
                base_activations = np.random.normal(0, 0.5, (batch_size, seq_length, feature_dim))
            
            # Add some structure to make clusters meaningful
            for token_idx in range(seq_length):
                cluster_influence = np.sin(token_idx * np.pi / seq_length)
                base_activations[:, token_idx, :] += cluster_influence * 0.5
            
            activations[layer] = base_activations.astype(np.float32)
        
        return activations
    
    def create_mock_attention_data(
        self,
        layers: List[str],
        seq_length: int,
        batch_size: int,
        num_heads: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate mock attention weights and patterns.
        
        Args:
            layers: List of layer names
            seq_length: Sequence length
            batch_size: Batch size
            num_heads: Number of attention heads
            
        Returns:
            Dictionary with attention data for each layer
        """
        if num_heads is None:
            num_heads = self.default_num_heads
        
        attention_data = {}
        
        for layer in layers:
            layer_idx = int(layer.split("_")[-1])
            
            # Generate attention weights (batch_size, num_heads, seq_length, seq_length)
            attention_weights = np.random.dirichlet(
                np.ones(seq_length), 
                size=(batch_size, num_heads, seq_length)
            )
            
            # Make attention patterns more structured
            # Early layers: more local attention
            # Later layers: more global attention
            if layer_idx < 2:
                # Local attention pattern
                for head in range(num_heads):
                    for i in range(seq_length):
                        # Focus on nearby tokens
                        local_mask = np.exp(-0.5 * ((np.arange(seq_length) - i) / 2) ** 2)
                        attention_weights[0, head, i] *= local_mask
                        attention_weights[0, head, i] /= attention_weights[0, head, i].sum()
            else:
                # Global attention pattern
                for head in range(num_heads):
                    # Some heads focus on beginning, others on end
                    if head % 2 == 0:
                        attention_weights[0, head, :, :3] *= 2
                    else:
                        attention_weights[0, head, :, -3:] *= 2
                    
                    # Renormalize
                    for i in range(seq_length):
                        attention_weights[0, head, i] /= attention_weights[0, head, i].sum()
            
            # Calculate attention statistics
            entropy = -np.sum(
                attention_weights * np.log(attention_weights + 1e-8),
                axis=-1
            ).mean()
            
            # Head agreement (correlation between heads)
            head_correlations = []
            for i in range(num_heads):
                for j in range(i + 1, num_heads):
                    corr = np.corrcoef(
                        attention_weights[0, i].flatten(),
                        attention_weights[0, j].flatten()
                    )[0, 1]
                    if not np.isnan(corr):
                        head_correlations.append(abs(corr))
            
            head_agreement = np.mean(head_correlations) if head_correlations else 0.0
            
            attention_data[layer] = {
                "weights": attention_weights,
                "entropy": float(entropy),
                "head_agreement": float(head_agreement),
                "num_heads": num_heads,
                "attention_patterns": self._extract_attention_patterns(attention_weights)
            }
        
        return attention_data
    
    def create_mock_cluster_labels(
        self,
        layers: List[str],
        seq_length: int,
        batch_size: int,
        num_clusters: int
    ) -> Dict[str, np.ndarray]:
        """
        Generate mock cluster assignments.
        
        Args:
            layers: List of layer names
            seq_length: Sequence length
            batch_size: Batch size
            num_clusters: Number of clusters per layer
            
        Returns:
            Dictionary mapping layer names to cluster label arrays
        """
        cluster_labels = {}
        
        # Create base cluster patterns
        base_pattern = np.random.randint(0, num_clusters, size=seq_length)
        
        for layer in layers:
            layer_idx = int(layer.split("_")[-1])
            
            # Create some evolution across layers
            if layer_idx == 0:
                # First layer: use base pattern
                layer_clusters = np.tile(base_pattern, (batch_size, 1)).flatten()
            else:
                # Later layers: evolve from previous layer
                prev_layer = f"layer_{layer_idx - 1}"
                prev_clusters = cluster_labels[prev_layer].reshape(batch_size, seq_length)
                
                # Add some random transitions
                layer_clusters = prev_clusters.copy()
                for batch in range(batch_size):
                    # 30% chance of cluster change for each token
                    change_mask = np.random.random(seq_length) < 0.3
                    layer_clusters[batch, change_mask] = np.random.randint(
                        0, num_clusters, size=change_mask.sum()
                    )
                
                layer_clusters = layer_clusters.flatten()
            
            cluster_labels[layer] = layer_clusters.astype(np.int32)
        
        return cluster_labels
    
    def create_mock_token_metadata(
        self, 
        seq_length: int, 
        batch_size: int
    ) -> Dict[str, Any]:
        """
        Generate mock token metadata.
        
        Args:
            seq_length: Sequence length
            batch_size: Batch size
            
        Returns:
            Token metadata dictionary
        """
        # Select tokens from sample
        tokens_per_batch = []
        token_ids_per_batch = []
        positions_per_batch = []
        
        for batch_idx in range(batch_size):
            # Use different starting points for variety
            start_idx = (batch_idx * 3) % len(self.sample_tokens)
            batch_tokens = []
            batch_token_ids = []
            
            for i in range(seq_length):
                token_idx = (start_idx + i) % len(self.sample_tokens)
                batch_tokens.append(self.sample_tokens[token_idx])
                batch_token_ids.append(self.sample_token_ids[token_idx])
            
            tokens_per_batch.append(batch_tokens)
            token_ids_per_batch.append(batch_token_ids)
            positions_per_batch.append(list(range(seq_length)))
        
        # Create attention mask (all ones for simplicity)
        attention_mask = np.ones((batch_size, seq_length), dtype=np.int32)
        
        return {
            "tokens": tokens_per_batch,
            "token_ids": np.array(token_ids_per_batch, dtype=np.int32),
            "positions": positions_per_batch,
            "attention_mask": attention_mask
        }
    
    def create_mock_token_paths(
        self,
        layers: List[str],
        token_metadata: Dict[str, Any],
        cluster_labels: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate mock token paths through clusters.
        
        Args:
            layers: List of layer names
            token_metadata: Token metadata
            cluster_labels: Cluster assignments
            
        Returns:
            Token paths dictionary
        """
        token_paths = {}
        tokens = token_metadata["tokens"]
        batch_size = len(tokens)
        seq_length = len(tokens[0])
        
        # Process each token position
        for batch_idx in range(batch_size):
            for pos_idx in range(seq_length):
                # Create unique token identifier
                token_id = f"{batch_idx * seq_length + pos_idx}"
                token_text = tokens[batch_idx][pos_idx]
                
                # Extract cluster path
                cluster_path = []
                for layer in layers:
                    flat_idx = batch_idx * seq_length + pos_idx
                    cluster_id = int(cluster_labels[layer][flat_idx])
                    cluster_path.append(cluster_id)
                
                # Calculate path metrics
                cluster_changes = sum(
                    1 for i in range(1, len(cluster_path))
                    if cluster_path[i] != cluster_path[i-1]
                )
                
                # Path length (Euclidean distance between cluster centers)
                path_length = self._calculate_path_length(cluster_path)
                
                # Mobility score (normalized cluster changes)
                mobility_score = cluster_changes / max(1, len(layers) - 1)
                
                token_paths[token_id] = {
                    "token_text": token_text,
                    "position": pos_idx,
                    "batch_idx": batch_idx,
                    "cluster_path": cluster_path,
                    "path_length": float(path_length),
                    "cluster_changes": cluster_changes,
                    "mobility_score": float(mobility_score)
                }
        
        return token_paths
    
    def create_mock_cluster_metrics(
        self,
        layers: List[str],
        num_clusters: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate mock cluster quality metrics.
        
        Args:
            layers: List of layer names
            num_clusters: Number of clusters per layer
            
        Returns:
            Cluster metrics dictionary
        """
        cluster_metrics = {}
        
        for layer in layers:
            layer_idx = int(layer.split("_")[-1])
            
            # Simulate different quality across layers
            base_purity = 0.7 + 0.1 * (layer_idx / len(layers))
            base_silhouette = 0.5 + 0.2 * (layer_idx / len(layers))
            
            # Add some noise
            purity = base_purity + np.random.normal(0, 0.05)
            silhouette = base_silhouette + np.random.normal(0, 0.05)
            
            # Clamp to valid ranges
            purity = np.clip(purity, 0.0, 1.0)
            silhouette = np.clip(silhouette, -1.0, 1.0)
            
            cluster_metrics[layer] = {
                "purity": float(purity),
                "silhouette": float(silhouette),
                "num_clusters": num_clusters,
                "cluster_sizes": np.random.multinomial(100, [1/num_clusters] * num_clusters).tolist()
            }
        
        return cluster_metrics
    
    def create_mock_similarity_data(
        self,
        tokens_per_batch: List[List[str]],
        layers: List[str]
    ) -> Dict[str, Any]:
        """
        Generate mock similarity and fragmentation data.
        
        Args:
            tokens_per_batch: List of token sequences
            layers: List of layer names
            
        Returns:
            Similarity data dictionary
        """
        # Flatten tokens
        all_tokens = []
        for batch_tokens in tokens_per_batch:
            all_tokens.extend(batch_tokens)
        
        # Generate fragmentation scores
        fragmentation_scores = []
        for token in all_tokens:
            # Score based on token characteristics
            if token in [".", "!", "?"]:  # Punctuation: low fragmentation
                score = np.random.uniform(0.0, 0.2)
            elif token.startswith(" "):  # Word tokens: variable fragmentation
                score = np.random.uniform(0.3, 0.8)
            else:  # Other tokens: moderate fragmentation
                score = np.random.uniform(0.2, 0.6)
            
            fragmentation_scores.append(score)
        
        return {
            "fragmentation_scores": {
                "scores": fragmentation_scores,
                "tokens": all_tokens
            },
            "layer_similarity": self._create_layer_similarity_matrix(layers),
            "token_correlations": self._create_token_correlation_data(all_tokens)
        }
    
    def create_mock_path_archetypes(
        self,
        token_paths: Dict[str, Dict[str, Any]],
        layers: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate mock archetypal paths.
        
        Args:
            token_paths: Token path data
            layers: List of layer names
            
        Returns:
            List of archetypal path dictionaries
        """
        # Group paths by pattern
        path_patterns = {}
        
        for token_id, path_data in token_paths.items():
            cluster_path = tuple(path_data["cluster_path"])
            
            if cluster_path not in path_patterns:
                path_patterns[cluster_path] = {
                    "tokens": [],
                    "count": 0
                }
            
            path_patterns[cluster_path]["tokens"].append(path_data["token_text"])
            path_patterns[cluster_path]["count"] += 1
        
        # Convert to archetypal paths
        archetypes = []
        for i, (cluster_path, data) in enumerate(path_patterns.items()):
            archetypes.append({
                "id": i,
                "layers": layers,
                "clusters": list(cluster_path),
                "count": data["count"],
                "example_tokens": data["tokens"][:5],  # Limit examples
                "frequency": data["count"] / len(token_paths)
            })
        
        # Sort by frequency
        archetypes.sort(key=lambda x: x["count"], reverse=True)
        
        return archetypes
    
    def _extract_attention_patterns(self, attention_weights: np.ndarray) -> Dict[str, Any]:
        """Extract meaningful patterns from attention weights."""
        batch_size, num_heads, seq_length, _ = attention_weights.shape
        
        patterns = {
            "local_attention": [],
            "global_attention": [],
            "head_specialization": []
        }
        
        for head in range(num_heads):
            head_weights = attention_weights[0, head]  # First batch
            
            # Local attention score (diagonal strength)
            local_score = np.mean(np.diag(head_weights))
            patterns["local_attention"].append(float(local_score))
            
            # Global attention score (first/last token attention)
            global_score = np.mean(head_weights[:, [0, -1]])
            patterns["global_attention"].append(float(global_score))
            
            # Head specialization (entropy of average attention)
            avg_attention = np.mean(head_weights, axis=0)
            entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-8))
            patterns["head_specialization"].append(float(entropy))
        
        return patterns
    
    def _calculate_path_length(self, cluster_path: List[int]) -> float:
        """Calculate path length based on cluster transitions."""
        if len(cluster_path) < 2:
            return 0.0
        
        # Simple distance: sum of cluster differences
        total_distance = 0.0
        for i in range(1, len(cluster_path)):
            # Treat clusters as points in 1D space
            distance = abs(cluster_path[i] - cluster_path[i-1])
            total_distance += distance
        
        return total_distance
    
    def _create_layer_similarity_matrix(self, layers: List[str]) -> Dict[str, float]:
        """Create mock layer similarity matrix."""
        similarity_matrix = {}
        
        for i, layer1 in enumerate(layers):
            for j, layer2 in enumerate(layers):
                if i == j:
                    similarity = 1.0
                else:
                    # Closer layers are more similar
                    distance = abs(i - j)
                    similarity = max(0.1, 1.0 - 0.2 * distance)
                    similarity += np.random.normal(0, 0.05)
                    similarity = np.clip(similarity, 0.0, 1.0)
                
                # Use string key instead of tuple for JSON serialization
                key = f"{layer1}->{layer2}"
                similarity_matrix[key] = float(similarity)
        
        return similarity_matrix
    
    def _create_token_correlation_data(self, tokens: List[str]) -> Dict[str, float]:
        """Create mock token correlation data."""
        correlations = {}
        
        unique_tokens = list(set(tokens))
        for token in unique_tokens:
            # Generate correlation based on token type
            if token in [".", "!", "?"]:
                correlation = np.random.uniform(0.1, 0.3)
            elif token.startswith(" "):
                correlation = np.random.uniform(0.4, 0.8)
            else:
                correlation = np.random.uniform(0.2, 0.6)
            
            correlations[token] = float(correlation)
        
        return correlations
    
    def save_to_temp_directory(self, analysis_results: Dict[str, Any]) -> str:
        """
        Save mock analysis results to a temporary directory.
        
        Args:
            analysis_results: Analysis results to save
            
        Returns:
            Path to temporary directory
        """
        temp_dir = tempfile.mkdtemp(prefix="gpt2_test_")
        
        # Save main analysis file
        analysis_file = os.path.join(temp_dir, "analysis_results.json")
        with open(analysis_file, 'w') as f:
            # Custom serializer for numpy arrays
            def json_serializer(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            json.dump(analysis_results, f, indent=2, default=json_serializer)
        
        # Save individual components for testing
        components = ["activations", "cluster_labels", "attention_data"]
        for component in components:
            if component in analysis_results:
                comp_file = os.path.join(temp_dir, f"{component}.json")
                with open(comp_file, 'w') as f:
                    json.dump(analysis_results[component], f, indent=2, default=json_serializer)
        
        return temp_dir


def create_test_analysis_results(**kwargs) -> Dict[str, Any]:
    """
    Convenience function to create test analysis results.
    
    Args:
        **kwargs: Arguments to pass to GPT2TestDataGenerator.create_mock_analysis_results()
        
    Returns:
        Mock analysis results
    """
    generator = GPT2TestDataGenerator()
    return generator.create_mock_analysis_results(**kwargs)


def create_minimal_test_data() -> Dict[str, Any]:
    """Create minimal test data for quick tests."""
    return create_test_analysis_results(
        num_layers=2,
        seq_length=4,
        batch_size=1,
        num_clusters=2,
        include_attention=True,
        include_persistence=True
    )


def create_comprehensive_test_data() -> Dict[str, Any]:
    """Create comprehensive test data for full integration tests."""
    return create_test_analysis_results(
        num_layers=6,
        seq_length=12,
        batch_size=2,
        num_clusters=4,
        include_attention=True,
        include_persistence=True
    )