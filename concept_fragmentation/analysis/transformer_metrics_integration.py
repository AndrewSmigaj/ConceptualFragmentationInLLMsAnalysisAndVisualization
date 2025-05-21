"""
Integration module for GPT-2 and transformer-specific metrics.

This module provides functionality to integrate newly implemented transformer metrics
with the existing Archetypal Path Analysis framework, allowing for comprehensive
analysis of GPT-2 and other transformer models.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import os

# Import base metrics and analysis components
from concept_fragmentation.metrics.transformer_metrics import (
    calculate_attention_entropy,
    calculate_attention_sparsity,
    calculate_head_importance,
    calculate_cross_head_agreement,
    analyze_attention_patterns,
    TransformerMetricsCalculator
)

# Import token path metrics
from concept_fragmentation.metrics.token_path_metrics import (
    calculate_token_path_coherence,
    calculate_token_path_divergence,
    calculate_semantic_stability,
    calculate_neighborhood_preservation,
    analyze_token_paths,
    TokenPathMetricsResult
)

# Import concept purity metrics
from concept_fragmentation.metrics.concept_purity_metrics import (
    calculate_intra_cluster_coherence,
    calculate_cross_layer_stability,
    calculate_concept_separability,
    calculate_layer_concept_purity,
    analyze_concept_purity,
    ConceptPurityResult
)

# Import transformer analysis components
from concept_fragmentation.analysis.transformer_cross_layer import (
    TransformerCrossLayerAnalyzer,
    CrossLayerTransformerAnalysisResult
)

# Import GPT-2 specific components
from concept_fragmentation.analysis.gpt2_model_adapter import (
    GPT2ActivationExtractor,
    GPT2ModelType,
    GPT2ActivationConfig
)

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class GPT2AnalysisResult:
    """
    Comprehensive results from GPT-2 model analysis.
    
    This dataclass combines results from various analysis components:
    - Attention metrics (attention entropy, sparsity, head importance)
    - Token path metrics (path coherence, divergence, stability)
    - Concept purity metrics (cluster coherence, separability, contamination)
    - Cross-layer metrics (layer relationships, attention flow)
    
    Attributes:
        attention_metrics: Results from attention pattern analysis
        token_path_metrics: Results from token path analysis
        concept_purity_metrics: Results from concept purity analysis
        cross_layer_metrics: Results from cross-layer analysis
        model_info: Information about the analyzed model
        input_info: Information about the analyzed inputs
        metadata: Additional metadata and configuration details
    """
    attention_metrics: Optional[Dict[str, Any]] = None
    token_path_metrics: Optional[TokenPathMetricsResult] = None
    concept_purity_metrics: Optional[ConceptPurityResult] = None
    cross_layer_metrics: Optional[CrossLayerTransformerAnalysisResult] = None
    model_info: Optional[Dict[str, Any]] = None
    input_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate a concise summary of analysis results.
        
        Returns:
            Dictionary with summary metrics across all analysis components
        """
        summary = {
            "model": self.model_info.get("model_type", "unknown") if self.model_info else "unknown",
            "metrics": {}
        }
        
        # Add attention metrics summary
        if self.attention_metrics:
            summary["metrics"]["attention"] = {
                "entropy": self.attention_metrics.get("entropy", 0.0),
                "sparsity": self.attention_metrics.get("sparsity", 0.0),
                "n_heads_analyzed": len(self.attention_metrics.get("head_importance", {}))
            }
        
        # Add token path metrics summary
        if self.token_path_metrics:
            summary["metrics"]["token_paths"] = self.token_path_metrics.aggregated_metrics
        
        # Add concept purity metrics summary
        if self.concept_purity_metrics:
            summary["metrics"]["concept_purity"] = self.concept_purity_metrics.aggregated_metrics
        
        # Add cross-layer metrics summary
        if self.cross_layer_metrics:
            summary["metrics"]["cross_layer"] = self.cross_layer_metrics.get_summary()
        
        return summary
    
    def save(self, output_path: str, filename: str = "gpt2_analysis_results.json") -> str:
        """
        Save analysis results to a JSON file.
        
        Args:
            output_path: Directory to save results
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Prepare serializable data
        serializable_data = {
            "model_info": self.model_info,
            "input_info": self.input_info,
            "metadata": self.metadata,
            "summary": self.summary()
        }
        
        # Add serializable attention metrics
        if self.attention_metrics:
            # Convert complex structures to serializable format
            attention_data = {}
            for key, value in self.attention_metrics.items():
                if isinstance(value, dict):
                    attention_data[key] = {
                        str(k): float(v) for k, v in value.items() 
                        if not isinstance(v, (np.ndarray, torch.Tensor))
                    }
                elif not isinstance(value, (np.ndarray, torch.Tensor, list, set)):
                    attention_data[key] = value
            serializable_data["attention_metrics"] = attention_data
        
        # Add serializable token path metrics
        if self.token_path_metrics:
            # Convert dataclass to dict
            token_path_data = {
                "aggregated_metrics": self.token_path_metrics.aggregated_metrics,
                "layers_analyzed": self.token_path_metrics.layers_analyzed,
                "tokens_analyzed": self.token_path_metrics.tokens_analyzed,
                "path_coherence": {
                    str(k): float(v) for k, v in self.token_path_metrics.path_coherence.items()
                },
                "semantic_stability": {
                    str(k): float(v) for k, v in self.token_path_metrics.semantic_stability.items()
                },
                "token_influence": {
                    str(k): float(v) for k, v in self.token_path_metrics.token_influence.items()
                },
                "token_specialization": {
                    str(k): float(v) for k, v in self.token_path_metrics.token_specialization.items()
                }
            }
            serializable_data["token_path_metrics"] = token_path_data
        
        # Add serializable concept purity metrics
        if self.concept_purity_metrics:
            # Convert dataclass to dict
            concept_purity_data = {
                "aggregated_metrics": self.concept_purity_metrics.aggregated_metrics,
                "layers_analyzed": self.concept_purity_metrics.layers_analyzed,
                "concepts_analyzed": self.concept_purity_metrics.concepts_analyzed,
                "intra_cluster_coherence": {
                    str(k): float(v) for k, v in self.concept_purity_metrics.intra_cluster_coherence.items()
                },
                "layer_concept_purity": {
                    str(k): float(v) for k, v in self.concept_purity_metrics.layer_concept_purity.items()
                },
                "concept_entropy": {
                    str(k): float(v) for k, v in self.concept_purity_metrics.concept_entropy.items()
                }
            }
            serializable_data["concept_purity_metrics"] = concept_purity_data
        
        # Save to file
        file_path = os.path.join(output_path, filename)
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        return file_path


class GPT2MetricsAnalyzer:
    """
    Comprehensive analyzer for GPT-2 models.
    
    This class integrates various metrics and analysis components to provide
    a complete picture of GPT-2 model behavior, focusing on attention patterns,
    token paths, and concept representation.
    
    Attributes:
        extractor: GPT2ActivationExtractor for extracting activations
        metrics_calculator: Calculator for transformer-specific metrics
        cross_layer_analyzer: Analyzer for cross-layer relationships
        use_cache: Whether to use caching for faster analysis
        cache_dir: Directory for caching results
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        model_type: Union[str, GPT2ModelType] = GPT2ModelType.SMALL,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        random_state: int = 42,
        extractor: Optional[GPT2ActivationExtractor] = None,
        metrics_calculator: Optional[TransformerMetricsCalculator] = None,
        cross_layer_analyzer: Optional[TransformerCrossLayerAnalyzer] = None
    ):
        """
        Initialize the GPT-2 metrics analyzer.
        
        Args:
            model_type: Type of GPT-2 model ('small', 'medium', 'large', 'xl')
            use_cache: Whether to use caching for faster analysis
            cache_dir: Directory for caching results
            random_state: Random seed for reproducibility
            extractor: Optional pre-initialized GPT2ActivationExtractor
            metrics_calculator: Optional pre-initialized TransformerMetricsCalculator
            cross_layer_analyzer: Optional pre-initialized TransformerCrossLayerAnalyzer
        """
        # Set up activation extractor
        if extractor is None:
            if isinstance(model_type, str):
                model_type = GPT2ModelType.from_string(model_type)
            
            config = GPT2ActivationConfig(
                model_type=model_type,
                use_cache=use_cache,
                cache_dir=cache_dir
            )
            
            self.extractor = GPT2ActivationExtractor(config=config)
        else:
            self.extractor = extractor
        
        # Set up metrics calculator
        self.metrics_calculator = metrics_calculator or TransformerMetricsCalculator(
            use_cache=use_cache
        )
        
        # Set up cross-layer analyzer
        self.cross_layer_analyzer = cross_layer_analyzer or TransformerCrossLayerAnalyzer(
            cache_dir=cache_dir,
            use_cache=use_cache,
            random_state=random_state
        )
        
        # Set attributes
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.random_state = random_state
    
    def analyze_text(
        self,
        text: Union[str, List[str]],
        layer_indices: Optional[List[int]] = None,
        include_attention_metrics: bool = True,
        include_token_path_metrics: bool = True,
        include_concept_purity_metrics: bool = True,
        include_cross_layer_metrics: bool = True,
        ground_truth_labels: Optional[Union[List[int], np.ndarray]] = None,
        cluster_labels: Optional[Dict[str, Union[List[int], np.ndarray]]] = None,
        metric: str = "cosine",
        n_clusters: int = 8
    ) -> GPT2AnalysisResult:
        """
        Analyze GPT-2 model behavior on the provided text.
        
        This method performs comprehensive analysis using multiple metrics
        to provide insights into the model's internal representations.
        
        Args:
            text: Input text or list of texts
            layer_indices: Indices of layers to analyze (None for all)
            include_attention_metrics: Whether to include attention metrics
            include_token_path_metrics: Whether to include token path metrics
            include_concept_purity_metrics: Whether to include concept purity metrics
            include_cross_layer_metrics: Whether to include cross-layer metrics
            ground_truth_labels: Optional ground truth labels for supervised evaluation
            cluster_labels: Optional pre-computed cluster labels for each layer
            metric: Distance metric to use ('cosine', 'euclidean')
            n_clusters: Number of clusters to use for clustering-based metrics
            
        Returns:
            GPT2AnalysisResult with comprehensive analysis results
        """
        logger.info(f"Analyzing text with GPT-2 {self.extractor.config.model_type.value}")
        
        # Extract activations and prepare input information
        activations = self.extractor.get_apa_activations(text, layers=layer_indices)
        
        # Prepare token information
        inputs = self.extractor.prepare_inputs(text)
        token_ids = inputs["input_ids"].cpu().numpy()
        attention_mask = inputs["attention_mask"].cpu().numpy()
        
        # Decode tokens for readability
        tokens = []
        for batch_idx in range(token_ids.shape[0]):
            batch_tokens = []
            for token_id in token_ids[batch_idx]:
                batch_tokens.append(self.extractor.tokenizer.decode([token_id]))
            tokens.append(batch_tokens)
        
        # Create input info
        input_info = {
            "token_ids": token_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "tokens": tokens
        }
        
        # Extract attention patterns
        attention_patterns = None
        if include_attention_metrics or include_cross_layer_metrics:
            # Get layer indices
            if layer_indices is None:
                layer_indices = list(range(len(self.extractor.model.transformer.h)))
            
            # Format layer names for extraction
            attention_layer_names = [f"transformer_layer_{i}_attention" for i in layer_indices]
            
            # Extract attention patterns
            attention_patterns = self.extractor.get_attention_patterns(text, attention_layer_names)
        
        # Analyze attention metrics
        attention_metrics = None
        if include_attention_metrics and attention_patterns is not None:
            logger.info("Analyzing attention metrics")
            
            # Calculate attention metrics
            metrics_result = self.metrics_calculator.compute_attention_metrics(
                attention_data=attention_patterns,
                token_mask=attention_mask,
                token_info={"token_strings": tokens[0] if tokens else None},
                include_cross_head_agreement=True,
                agreement_metric=metric
            )
            
            # Convert to dictionary for easier serialization
            attention_metrics = {
                "entropy": metrics_result.entropy,
                "sparsity": metrics_result.sparsity,
                "head_importance": metrics_result.head_importance,
                "layer_entropy": metrics_result.layer_entropy,
                "layer_sparsity": metrics_result.layer_sparsity,
                "head_entropy": metrics_result.head_entropy,
                "head_sparsity": metrics_result.head_sparsity,
                "cross_head_agreement": metrics_result.cross_head_agreement,
                "layers_analyzed": metrics_result.layers_analyzed,
                "heads_analyzed": metrics_result.heads_analyzed
            }
        
        # Analyze token path metrics
        token_path_metrics = None
        if include_token_path_metrics:
            logger.info("Analyzing token path metrics")
            
            # Calculate token path metrics
            token_path_metrics = analyze_token_paths(
                layer_representations=activations,
                token_indices=None,  # Use all tokens
                token_mask=attention_mask,
                token_strings=tokens[0] if tokens else None,
                metric=metric,
                n_neighbors=5,
                layer_windows=2
            )
        
        # Analyze concept purity metrics
        concept_purity_metrics = None
        if include_concept_purity_metrics:
            logger.info("Analyzing concept purity metrics")
            
            # Generate cluster labels if not provided
            if cluster_labels is None:
                logger.info(f"Generating cluster labels with {n_clusters} clusters")
                generated_labels = {}
                
                for layer_name, layer_activations in activations.items():
                    # Reshape activations to [n_samples, n_features]
                    if len(layer_activations.shape) == 3:  # [batch_size, seq_len, hidden_dim]
                        batch_size, seq_len, hidden_dim = layer_activations.shape
                        reshaped_activations = layer_activations.reshape(-1, hidden_dim)
                    else:
                        reshaped_activations = layer_activations
                    
                    # Apply token mask
                    masked_indices = []
                    for b in range(attention_mask.shape[0]):
                        for s in range(attention_mask.shape[1]):
                            if attention_mask[b, s] > 0:
                                masked_indices.append(b * attention_mask.shape[1] + s)
                    
                    masked_activations = reshaped_activations[masked_indices]
                    
                    # Cluster activations
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                    cluster_assignments = kmeans.fit_predict(masked_activations)
                    
                    # Create full labels array
                    labels = np.zeros(reshaped_activations.shape[0], dtype=int)
                    for i, idx in enumerate(masked_indices):
                        labels[idx] = cluster_assignments[i]
                    
                    # Reshape back to original shape if needed
                    if len(layer_activations.shape) == 3:
                        labels = labels.reshape(batch_size, seq_len)
                    
                    generated_labels[layer_name] = labels
                
                cluster_labels = generated_labels
            
            # Calculate concept purity metrics
            concept_purity_metrics = analyze_concept_purity(
                layer_activations=activations,
                cluster_labels=cluster_labels,
                ground_truth_labels=ground_truth_labels,
                metric=metric
            )
        
        # Analyze cross-layer metrics
        cross_layer_metrics = None
        if include_cross_layer_metrics:
            logger.info("Analyzing cross-layer metrics")
            
            # Calculate cross-layer metrics
            cross_layer_metrics = self.cross_layer_analyzer.analyze_transformer_cross_layer(
                layer_activations=activations,
                attention_data=attention_patterns if attention_patterns else {},
                token_ids=token_ids,
                token_strings=tokens[0] if tokens else None,
                class_labels=ground_truth_labels
            )
        
        # Create model info
        model_info = {
            "model_type": self.extractor.config.model_type.value,
            "n_layers": len(self.extractor.model.transformer.h),
            "hidden_size": self.extractor.model.config.hidden_size,
            "n_heads": self.extractor.model.config.n_head
        }
        
        # Create metadata
        metadata = {
            "analysis_config": {
                "include_attention_metrics": include_attention_metrics,
                "include_token_path_metrics": include_token_path_metrics,
                "include_concept_purity_metrics": include_concept_purity_metrics,
                "include_cross_layer_metrics": include_cross_layer_metrics,
                "metric": metric,
                "n_clusters": n_clusters,
                "layer_indices": layer_indices
            }
        }
        
        # Create result
        result = GPT2AnalysisResult(
            attention_metrics=attention_metrics,
            token_path_metrics=token_path_metrics,
            concept_purity_metrics=concept_purity_metrics,
            cross_layer_metrics=cross_layer_metrics,
            model_info=model_info,
            input_info=input_info,
            metadata=metadata
        )
        
        return result


# Convenience functions for common use cases

def analyze_gpt2_small(
    text: Union[str, List[str]],
    layer_indices: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> GPT2AnalysisResult:
    """
    Analyze text with GPT-2 Small model.
    
    Args:
        text: Input text or list of texts
        layer_indices: Indices of layers to analyze (None for all)
        output_path: Optional path to save results
        **kwargs: Additional arguments for GPT2MetricsAnalyzer.analyze_text
        
    Returns:
        GPT2AnalysisResult with analysis results
    """
    analyzer = GPT2MetricsAnalyzer(model_type=GPT2ModelType.SMALL)
    result = analyzer.analyze_text(text, layer_indices, **kwargs)
    
    if output_path:
        result.save(output_path, filename="gpt2_small_analysis_results.json")
    
    return result


def analyze_gpt2_medium(
    text: Union[str, List[str]],
    layer_indices: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> GPT2AnalysisResult:
    """
    Analyze text with GPT-2 Medium model.
    
    Args:
        text: Input text or list of texts
        layer_indices: Indices of layers to analyze (None for all)
        output_path: Optional path to save results
        **kwargs: Additional arguments for GPT2MetricsAnalyzer.analyze_text
        
    Returns:
        GPT2AnalysisResult with analysis results
    """
    analyzer = GPT2MetricsAnalyzer(model_type=GPT2ModelType.MEDIUM)
    result = analyzer.analyze_text(text, layer_indices, **kwargs)
    
    if output_path:
        result.save(output_path, filename="gpt2_medium_analysis_results.json")
    
    return result


def analyze_gpt2_text(
    text: Union[str, List[str]],
    model_type: str = "small",
    layer_indices: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> GPT2AnalysisResult:
    """
    Analyze text with the specified GPT-2 model type.
    
    Args:
        text: Input text or list of texts
        model_type: Type of GPT-2 model ('small', 'medium', 'large', 'xl')
        layer_indices: Indices of layers to analyze (None for all)
        output_path: Optional path to save results
        **kwargs: Additional arguments for GPT2MetricsAnalyzer.analyze_text
        
    Returns:
        GPT2AnalysisResult with analysis results
    """
    # Convert model_type to enum
    if isinstance(model_type, str):
        model_type = GPT2ModelType.from_string(model_type)
    
    analyzer = GPT2MetricsAnalyzer(model_type=model_type)
    result = analyzer.analyze_text(text, layer_indices, **kwargs)
    
    if output_path:
        filename = f"gpt2_{model_type.value}_analysis_results.json"
        result.save(output_path, filename=filename)
    
    return result