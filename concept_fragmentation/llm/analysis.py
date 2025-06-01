"""
High-level API for LLM-based analysis of cluster paths.

This module provides functions for using LLMs to label clusters and generate
narratives for paths through the activation space.
"""

import os
import json
import asyncio
import time
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

from .factory import LLMClientFactory
from .responses import LLMResponse
from .cache_manager import CacheManager
from .batch_processor import batch_generate_labels, batch_generate_narratives
from .prompt_optimizer import optimize_cluster_label_prompt, optimize_path_narrative_prompt


class ClusterAnalysis:
    """
    High-level API for LLM-based analysis of clusters and paths.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        memory_only_cache: bool = False,
        save_interval: int = 10,
        optimize_prompts: bool = False,
        optimization_level: int = 1,
        debug: bool = False
    ):
        """
        Initialize the ClusterAnalysis.
        
        Args:
            provider: The LLM provider to use (e.g., "openai", "claude", "grok")
            model: The model to use (default: provider's default model)
            api_key: The API key to use (if None, will try to get from environment)
            use_cache: Whether to cache LLM responses
            cache_dir: Directory to store cache files (default: ./cache/llm)
            cache_ttl: Optional time-to-live for cache entries in seconds
            memory_only_cache: If True, don't persist cache to disk
            save_interval: Save to disk every N new cache items
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
        
        # Set up caching with the new cache manager
        self.use_cache = use_cache
        self.cache_manager = CacheManager(
            provider=self.provider,
            model=self.model,
            cache_dir=cache_dir,
            use_cache=use_cache,
            cache_ttl=cache_ttl,
            memory_only=memory_only_cache,
            save_interval=save_interval
        )
        
        # Prompt optimization settings
        self.optimize_prompts = optimize_prompts
        self.optimization_level = optimization_level
        self.debug = debug
    
    async def _generate_with_cache(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text with improved caching.
        
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
            cached_response = self.cache_manager.get(
                prompt=prompt, 
                temperature=temperature, 
                max_tokens=max_tokens, 
                **kwargs
            )
            
            if cached_response:
                return cached_response
        
        # Generate response from LLM
        response = await self.client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Cache the response if enabled
        if self.use_cache:
            self.cache_manager.store(
                prompt=prompt,
                response=response,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        
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
        try:
            # Try to get the event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there is no event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(
            self._generate_with_cache(prompt, temperature, max_tokens, **kwargs)
        )
    
    def _build_label_prompt(self, cluster_context_data: str) -> str:
        """Builds the prompt for generating a cluster label based on its demographic/statistical context string."""
        prompt = (
            "A cluster in a neural network layer is characterized by the following "
            "demographic and outcome statistics of its member data samples. "
            "Provide a concise, human-readable semantic label (3-5 words) that captures "
            "the dominant characteristics or concept represented by this group.\n\n"
            f"Cluster Demographic & Outcome Summary:\n{cluster_context_data}\n\n"
            "Semantic Label (3-5 words):"
        )
        return prompt

    async def label_cluster(
        self,
        cluster_id: str,
        cluster_profile_str: str,
        other_cluster_labels: Optional[List[str]] = None
    ) -> str:
        """
        Generate a human-readable label for a single cluster based on its textual profile.
        
        Args:
            cluster_id: The ID of the cluster being labeled.
            cluster_profile_str: A textual summary of the cluster's demographic/statistical profile.
            other_cluster_labels: Optional list of existing labels for other clusters to encourage distinctness.
            
        Returns:
            A human-readable label for the cluster.
        """
        prompt = self._build_label_prompt(cluster_profile_str)

        # Add information about other clusters if provided, to encourage distinct labels
        if other_cluster_labels:
            prompt += "\n\nConsider these existing labels for other clusters in the system:\n"
            for i, label in enumerate(other_cluster_labels[:5]):
                prompt += f"- {label}\n"
            prompt += "\nProvide a label that is distinct from these existing labels."
        
        # Update the main instruction part of the prompt for distinctness too
        prompt = prompt.replace("Semantic Label (3-5 words):", 
                                "Distinct Semantic Label (3-5 words):" if other_cluster_labels else "Semantic Label (3-5 words):")

        if self.debug:
            print(f"--- CLUSTER LABEL PROMPT for {cluster_id} ---\n{prompt[:1000]}\n--- END PROMPT ---\n")

        response = await self._generate_with_cache(
            prompt=prompt,
            temperature=0.3,
            max_tokens=20
        )
        
        label = response.text.strip().strip('"\'').replace("Label:", "").strip()
        if self.debug:
            print(f"--- CLUSTER LABEL RESPONSE for {cluster_id} ---\n{label}\n--- END RESPONSE ---\n")

        if len(label) > 50:
            label = label[:47] + "..."
        return label

    async def label_clusters(
        self,
        cluster_profiles: Dict[str, str],
        max_concurrency: int = 5
    ) -> Dict[str, str]:
        """
        Generate human-readable labels for multiple clusters based on their textual profiles.
        
        Args:
            cluster_profiles: Dictionary mapping cluster IDs to their textual demographic/statistical profiles.
            max_concurrency: Maximum number of concurrent API requests.
            
        Returns:
            Dictionary mapping cluster IDs to labels.
        """
        labels = {}
        
        if not cluster_profiles:
            return labels

        if len(cluster_profiles) <= 5 or max_concurrency <= 1:
            existing_labels_list = []
            for cluster_id, profile_str in cluster_profiles.items():
                label = await self.label_cluster(cluster_id, profile_str, other_cluster_labels=existing_labels_list)
                labels[cluster_id] = label
                existing_labels_list.append(label)
            return labels
        else:
            tasks = []
            for cluster_id, profile_str in cluster_profiles.items():
                tasks.append(self.label_cluster(cluster_id, profile_str, other_cluster_labels=None))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            idx = 0
            for cluster_id in cluster_profiles.keys():
                if isinstance(results[idx], Exception):
                    print(f"Error labeling cluster {cluster_id}: {results[idx]}")
                    labels[cluster_id] = "Error: Labeling failed"
                else:
                    labels[cluster_id] = results[idx]
                idx += 1
            return labels

    async def generate_path_narrative(
        self,
        path: List[str],
        cluster_labels: Dict[str, str],
        convergent_points: Optional[List[Tuple[str, str, float]]] = None,
        fragmentation_score: Optional[float] = None,
        demographic_info: Optional[Dict[str, Any]] = None,
        cluster_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        analysis_categories: Optional[List[str]] = None
    ) -> str:
        """
        Generate a human-readable narrative for a path through activation space.
        
        Args:
            path: List of cluster IDs representing the path
            cluster_labels: Dictionary mapping cluster IDs to human-readable labels
            convergent_points: Optional list of convergent points in the path
                Format: [(early_cluster_id, late_cluster_id, similarity)]
            fragmentation_score: Optional fragmentation score for the path
            demographic_info: Optional demographic information about datapoints in this path
            cluster_stats: Optional dictionary mapping cluster IDs to cluster statistics
            analysis_categories: Optional list of analysis types to include 
                (e.g., ['interpretation', 'bias', 'efficiency', 'robustness'])
                
        Returns:
            A human-readable narrative for the path
        """
        path_description_parts = []
        for cluster_id_in_path in path:
            label = cluster_labels.get(cluster_id_in_path, f"Unlabeled ({cluster_id_in_path})")
            path_description_parts.append(f"{cluster_id_in_path} ({label})")
        
        path_description_str = " → ".join(path_description_parts)

        prompt_lines = [
            "You are an AI expert analyzing neural network activation patterns.",
            "Generate a clear, insightful narrative that explains the following path through activation clusters in a neural network.",
            "Focus on the conceptual meaning (derived from cluster labels) and the potential decision process represented by this path.",
            f"Path: {path_description_str}"
        ]

        if cluster_stats:
            stats_summary_lines = ["\nCluster-Specific Demographics/Statistics along this path:"]
            for cluster_id_in_path in path:
                profile_str = cluster_stats.get(cluster_id_in_path)
                if profile_str:
                    stats_summary_lines.append(f"- {cluster_id_in_path} ({cluster_labels.get(cluster_id_in_path, '')}): {profile_str}")
            if len(stats_summary_lines) > 1:
                 prompt_lines.extend(stats_summary_lines)
        
        if demographic_info:
            prompt_lines.append("\nDemographic profile of samples following THIS ENTIRE PATH:")
            for key, value in demographic_info.items():
                if isinstance(value, dict):
                    dist_str = ", ".join([f"{k}: {v:.1%}" if isinstance(v,float) else f"{k}: {v}" for k,v in value.items()])
                    prompt_lines.append(f"- {key.replace('_', ' ').title()}: {dist_str}")
                elif isinstance(value, float):
                     prompt_lines.append(f"- {key.replace('_', ' ').title()}: {value:.2f}" + ("%" if "rate" in key or "percentage" in key else ""))
                else:
                    prompt_lines.append(f"- {key.replace('_', ' ').title()}: {value}")

        if convergent_points:
            prompt_lines.append("\nConceptual convergence points in this path:")
            for early_id, late_id, similarity in convergent_points:
                early_label = cluster_labels.get(early_id, early_id)
                late_label = cluster_labels.get(late_id, late_id)
                prompt_lines.append(f"- {early_id} ({early_label}) and {late_id} ({late_label}): {similarity:.2f} similarity")
        
        if fragmentation_score is not None:
            level = "high" if fragmentation_score > 0.7 else "low" if fragmentation_score < 0.3 else "moderate"
            prompt_lines.append(f"\nThis path has a {level} fragmentation score of {fragmentation_score:.2f}.")

        prompt_lines.extend([
            "\nBased on this information, write a concise narrative (2-4 sentences) that explains:",
            "1. What concepts or features (drawing from cluster labels and their demographic profiles) this path might represent.",
            "2. How the concept evolves or transforms across layers.",
            "3. Any potential insights about the model's decision-making process for data following this path.",
            "4. If relevant, how the overall demographic factors for this path relate to its journey.",
            "5. Any other insights you find relevant.",
            "Your explanation should be clear and insightful without being overly technical.",
            "Path narrative:"
        ])
        
        prompt = "\n".join(prompt_lines)
        if self.optimize_prompts:
            prompt = optimize_path_narrative_prompt(prompt, self.optimization_level)

        if self.debug:
            print(f"--- PATH NARRATIVE PROMPT for path {' → '.join(path)} ---\n{prompt[:1500]}\n--- END PROMPT ---\n")
        
        response = await self._generate_with_cache(
            prompt=prompt, temperature=0.4, max_tokens=300,
            system_prompt="You are an AI assistant that provides insightful, concise explanations of neural network behavior patterns."
        )
        
        if self.debug:
            print(f"--- PATH NARRATIVE RESPONSE ---\n{response.text[:400]}\n--- END RESPONSE ---\n")
        
        return response.text.strip()
    
    async def generate_path_narratives(
        self,
        paths: Dict[int, List[str]],
        cluster_labels: Dict[str, str],
        convergent_points: Optional[Dict[int, List[Tuple[str, str, float]]]] = None,
        fragmentation_scores: Optional[Dict[int, float]] = None,
        path_demographic_info: Optional[Dict[int, Dict[str, Any]]] = None,
        per_cluster_stats_for_paths: Optional[Dict[int, Dict[str, str]]] = None,
        analysis_categories: Optional[List[str]] = None
    ) -> str:
        """
        Generate comprehensive analysis of multiple paths in a single LLM call.
        
        Args:
            paths: Dictionary mapping path IDs to lists of cluster IDs
            cluster_labels: Dictionary mapping cluster IDs to human-readable labels
            convergent_points: Optional dictionary mapping path IDs to lists of convergent points
            fragmentation_scores: Optional dictionary mapping path IDs to fragmentation scores
            path_demographic_info: Optional dictionary mapping path IDs to demographic information
            per_cluster_stats_for_paths: Optional dictionary mapping path IDs to cluster profiles
            analysis_categories: Optional list of analysis types to include 
                (e.g., ['interpretation', 'bias', 'efficiency', 'robustness'])
                
        Returns:
            Comprehensive analysis text covering all requested categories
        """
        if not paths: 
            return ""

        # Default analysis categories
        if analysis_categories is None:
            analysis_categories = ['interpretation', 'bias']
            
        # Build comprehensive prompt with ALL paths for pattern detection
        prompt_lines = [
            "You are an AI expert analyzing neural network activation patterns.",
            "I will provide you with multiple archetypal paths through activation clusters.",
            "Analyze ALL paths together to identify patterns, especially those that only emerge when comparing paths.",
            "",
            f"Total archetypal paths to analyze: {len(paths)}",
            ""
        ]
        
        # Add all paths with their details
        prompt_lines.append("=== ARCHETYPAL PATHS ===")
        for path_id, path_clusters in paths.items():
            # Build path description
            path_desc = " → ".join([f"{cid} ({cluster_labels.get(cid, 'unlabeled')})" for cid in path_clusters])
            prompt_lines.append(f"\nPath {path_id}: {path_desc}")
            
            # Add demographic info if available
            if path_demographic_info and path_id in path_demographic_info:
                demo_info = path_demographic_info[path_id]
                prompt_lines.append("  Demographics for this path:")
                for key, value in demo_info.items():
                    if isinstance(value, dict):
                        dist_str = ", ".join([f"{k}: {v:.1%}" if isinstance(v,float) else f"{k}: {v}" for k,v in value.items()])
                        prompt_lines.append(f"    - {key}: {dist_str}")
                    else:
                        prompt_lines.append(f"    - {key}: {value}")
            
            # Add fragmentation score if available
            if fragmentation_scores and path_id in fragmentation_scores:
                prompt_lines.append(f"  Fragmentation score: {fragmentation_scores[path_id]:.3f}")
        
        # Add cluster statistics summary
        if per_cluster_stats_for_paths:
            prompt_lines.append("\n=== CLUSTER STATISTICS ===")
            all_clusters = {}
            for path_id, cluster_stats in per_cluster_stats_for_paths.items():
                for cluster_id, stats in cluster_stats.items():
                    if cluster_id not in all_clusters:
                        all_clusters[cluster_id] = stats
            
            for cluster_id, stats in sorted(all_clusters.items())[:20]:  # Limit to avoid token overflow
                prompt_lines.append(f"Cluster {cluster_id} ({cluster_labels.get(cluster_id, 'unlabeled')}): {stats}")
        
        # Add analysis instructions based on categories
        prompt_lines.append("\n=== ANALYSIS REQUIRED ===")
        prompt_lines.append("Analyze the above paths and provide insights on:")
        
        if 'interpretation' in analysis_categories:
            prompt_lines.extend([
                "\nINTERPRETATION:",
                "- What are the main conceptual paths through the network?",
                "- How do concepts transform as they move through layers?",
                "- What decision-making patterns emerge from these paths?"
            ])
        
        if 'bias' in analysis_categories:
            prompt_lines.extend([
                "\nBIAS ANALYSIS:",
                "- Are there systematic differences in how different demographic groups are routed through paths?",
                "- Do you see unexpected segregation where similar inputs follow very different paths?",
                "- Are certain paths predominantly associated with specific demographic characteristics?",
                "- What patterns suggest potential unfair treatment or discrimination?",
                "- Are there statistical anomalies in path distributions that indicate bias?"
            ])
        
        if 'efficiency' in analysis_categories:
            prompt_lines.extend([
                "\nEFFICIENCY:",
                "- Which paths seem redundant or could be consolidated?",
                "- Are there inefficiencies in how information flows through the network?",
                "- What opportunities exist for model compression?"
            ])
        
        if 'robustness' in analysis_categories:
            prompt_lines.extend([
                "\nROBUSTNESS:",
                "- Which paths appear most stable vs fragile?",
                "- Are there paths that might be vulnerable to adversarial inputs?",
                "- How consistent are the conceptual representations?"
            ])
        
        prompt_lines.extend([
            "\nProvide a comprehensive analysis addressing each category.",
            "Be specific about which paths exhibit which patterns.",
            "Focus on actionable insights and patterns that span multiple paths."
        ])
        
        prompt = "\n".join(prompt_lines)
        
        # Make single API call with all data
        response = await self._generate_with_cache(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1500,  # Increased for comprehensive analysis
            system_prompt="You are an AI assistant specializing in neural network analysis, pattern recognition, and bias detection."
        )
        
        return response.text.strip()
    
    def label_clusters_sync(
        self,
        cluster_profiles: Dict[str, str],
        max_concurrency: int = 5
    ) -> Dict[str, str]:
        """
        Synchronous wrapper for label_clusters.
        
        Args:
            cluster_profiles: Dictionary mapping cluster IDs to their textual demographic/statistical profiles
            
        Returns:
            Dictionary mapping cluster IDs to labels
        """
        try:
            # Try to get the event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there is no event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(
            self.label_clusters(cluster_profiles, max_concurrency=max_concurrency)
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.use_cache:
            return {"enabled": False}
            
        return self.cache_manager.get_stats()
    
    def prewarm_cache(self, prompts: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Prewarm the cache with a batch of prompts.
        
        Args:
            prompts: List of (prompt, kwargs) tuples to generate and cache
            
        Returns:
            Statistics about the prewarming operation
        """
        if not self.use_cache:
            return {"enabled": False, "warmed": 0}
        
        start_time = time.time()
        cache_size_before = len(self.cache_manager)
        
        # Process all prompts
        processed = 0
        cached = 0
        
        for prompt, kwargs in prompts:
            # Check if already in cache
            cache_key = self.cache_manager._generate_cache_key(prompt, **kwargs)
            if cache_key in self.cache_manager.cache:
                cached += 1
                continue
            
            # Generate and cache
            response = self.generate_with_cache(prompt, **kwargs)
            processed += 1
        
        # Get stats
        elapsed = time.time() - start_time
        cache_size_after = len(self.cache_manager)
        
        return {
            "enabled": True,
            "processed": processed,
            "already_cached": cached,
            "elapsed_seconds": elapsed,
            "cache_size_before": cache_size_before,
            "cache_size_after": cache_size_after,
            "new_items_added": cache_size_after - cache_size_before
        }
    
    def clear_cache(self, force_save: bool = True) -> None:
        """
        Clear the cache.
        
        Args:
            force_save: Whether to save the empty cache to disk
        """
        if not self.use_cache:
            return
            
        self.cache_manager.clear(force_save=force_save)
    
    def close(self) -> None:
        """
        Properly close the cache manager and any other resources.
        """
        if hasattr(self, 'cache_manager'):
            self.cache_manager.close()
    
    def generate_path_narratives_sync(
        self,
        paths: Dict[int, List[str]],
        cluster_labels: Dict[str, str],
        convergent_points: Optional[Dict[int, List[Tuple[str, str, float]]]] = None,
        fragmentation_scores: Optional[Dict[int, float]] = None,
        path_demographic_info: Optional[Dict[int, Dict[str, Any]]] = None,
        per_cluster_stats_for_paths: Optional[Dict[int, Dict[str, str]]] = None,
        analysis_categories: Optional[List[str]] = None
    ) -> str:
        """
        Synchronous wrapper for generate_path_narratives.
        
        Args:
            paths: Dictionary mapping path IDs to lists of cluster IDs
            cluster_labels: Dictionary mapping cluster IDs to human-readable labels
            convergent_points: Optional dictionary mapping path IDs to lists of convergent points
            fragmentation_scores: Optional dictionary mapping path IDs to fragmentation scores
            path_demographic_info: Optional dictionary mapping path IDs to demographic information
            per_cluster_stats_for_paths: Optional dictionary mapping path IDs to cluster profiles
            analysis_categories: Optional list of analysis types to include 
                (e.g., ['interpretation', 'bias', 'efficiency', 'robustness'])
                
        Returns:
            Comprehensive analysis text covering all requested categories
        """
        try:
            # Try to get the event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there is no event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(
            self.generate_path_narratives(
                paths, cluster_labels, convergent_points, fragmentation_scores, 
                path_demographic_info, per_cluster_stats_for_paths,
                analysis_categories=analysis_categories
            )
        )