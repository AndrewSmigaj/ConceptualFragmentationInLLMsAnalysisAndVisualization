"""
Test script for LLM integration with cluster paths.

This script tests the integration between LLM providers and cluster path analysis.
It verifies:
1. LLM provider connectivity and authentication
2. Cluster centroid processing and path extraction
3. Cluster labeling capabilities
4. Path narrative generation
5. Bias audit functionality (if demographic data is available)

Usage:
    python test_integration.py --providers grok --dataset titanic --mode quick
    python test_integration.py --providers all --dataset titanic --mode comprehensive
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

# Add parent directory to path for imports if needed
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import LLM module components
try:
    from concept_fragmentation.llm.analysis import ClusterAnalysis
    from concept_fragmentation.llm.factory import LLMClientFactory
    from concept_fragmentation.llm.bias_audit import generate_bias_report, analyze_bias_with_llm
except ImportError as e:
    print(f"Warning: Could not import LLM module: {e}")
    print("Some functionality may be limited.")

# Tester classes
class ProviderTester:
    """Class for testing LLM provider connectivity and basic functionality."""
    
    def __init__(
        self, 
        provider_name: str, 
        use_cache: bool = False,
        cache_ttl: Optional[int] = None,
        memory_only_cache: bool = False,
        debug: bool = False
    ):
        """
        Initialize the provider tester.
        
        Args:
            provider_name: Name of the LLM provider to test
            use_cache: Whether to use cached responses
            cache_ttl: Time-to-live for cache entries in seconds
            memory_only_cache: If True, don't persist cache to disk
            debug: Whether to enable debug logging
        """
        self.provider_name = provider_name
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.memory_only_cache = memory_only_cache
        self.debug = debug
        self.analyzer = None
        self.connection_result = None
        self.generation_result = None
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to the provider.
        
        Returns:
            Dictionary with connection test results
        """
        start_time = time.time()
        success = False
        error_message = None
        
        try:
            # Try to create analyzer
            self.analyzer = ClusterAnalysis(
                provider=self.provider_name,
                use_cache=self.use_cache,
                cache_ttl=self.cache_ttl,
                memory_only_cache=self.memory_only_cache,
                optimize_prompts=args.optimize_prompts if hasattr(args, 'optimize_prompts') else False,
                optimization_level=args.optimization_level if hasattr(args, 'optimization_level') else 1
            )
            
            # Get client info
            client_info = {
                "model": self.analyzer.model,
                "provider": self.analyzer.provider
            }
            
            success = True
            
        except Exception as e:
            error_message = str(e)
            client_info = {}
            if self.debug:
                import traceback
                traceback.print_exc()
        
        elapsed_time = time.time() - start_time
        
        self.connection_result = {
            "success": success,
            "elapsed_time": elapsed_time,
            "error": error_message,
            "client_info": client_info
        }
        
        return self.connection_result
    
    def test_basic_generation(self) -> Dict[str, Any]:
        """
        Test basic text generation.
        
        Returns:
            Dictionary with generation test results
        """
        if not self.connection_result or not self.connection_result["success"]:
            return {"success": False, "error": "Connection test failed"}
        
        start_time = time.time()
        success = False
        error_message = None
        response_text = None
        tokens_used = None
        
        # Simple test prompt
        test_prompt = "What is the capital of France? Give a one-word answer."
        
        try:
            # Generate response
            response = self.analyzer.generate_with_cache(
                prompt=test_prompt,
                temperature=0.0,  # Use deterministic output for testing
                max_tokens=10
            )
            
            response_text = response.text.strip()
            tokens_used = response.tokens_used
            success = True
            
        except Exception as e:
            error_message = str(e)
            if self.debug:
                import traceback
                traceback.print_exc()
        
        elapsed_time = time.time() - start_time
        
        self.generation_result = {
            "success": success,
            "elapsed_time": elapsed_time,
            "error": error_message,
            "response_text": response_text,
            "tokens_used": tokens_used,
            "expected_text": "Paris"
        }
        
        # Validate response
        if success:
            self.generation_result["response_matches"] = "paris" in response_text.lower()
        
        return self.generation_result
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of provider tests.
        
        Returns:
            Dictionary with test summary
        """
        return {
            "provider": self.provider_name,
            "connection": self.connection_result,
            "basic_generation": self.generation_result,
            "overall_success": (
                self.connection_result and 
                self.connection_result["success"] and
                self.generation_result and 
                self.generation_result["success"]
            )
        }


class AnalysisTester:
    """Class for testing LLM analysis capabilities."""
    
    def __init__(
        self, 
        analyzer: 'ClusterAnalysis', 
        test_data: Dict[str, Any],
        debug: bool = False
    ):
        """
        Initialize analysis tester.
        
        Args:
            analyzer: ClusterAnalysis instance
            test_data: Test data dictionary
            debug: Whether to enable debug logging
        """
        self.analyzer = analyzer
        self.test_data = test_data
        self.debug = debug
        self.results = {
            "cluster_labeling": None,
            "path_narratives": None,
            "bias_audit": None
        }
    
    def test_cluster_labeling(self, sample_size: int = 3) -> Dict[str, Any]:
        """
        Test cluster labeling functionality.
        
        Args:
            sample_size: Number of clusters to sample for testing (0 for all)
            
        Returns:
            Dictionary with cluster labeling test results
        """
        start_time = time.time()
        success = False
        error_message = None
        labels = {}
        
        try:
            # Extract centroids
            centroids = extract_centroids(self.test_data)
            
            # Limit to sample size if needed
            if sample_size > 0 and sample_size < len(centroids):
                # Select evenly distributed samples
                centroid_items = list(centroids.items())
                sample_indices = np.linspace(0, len(centroid_items)-1, sample_size, dtype=int)
                sampled_centroids = {
                    k: v for i, (k, v) in enumerate(centroid_items) if i in sample_indices
                }
            else:
                sampled_centroids = centroids
            
            # Generate labels
            labels = self.analyzer.label_clusters_sync(sampled_centroids)
            
            success = len(labels) > 0
            
        except Exception as e:
            error_message = str(e)
            if self.debug:
                import traceback
                traceback.print_exc()
        
        elapsed_time = time.time() - start_time
        
        # Basic quality check
        quality_metrics = {}
        if success:
            quality_metrics = self._evaluate_label_quality(labels)
        
        self.results["cluster_labeling"] = {
            "success": success,
            "elapsed_time": elapsed_time,
            "error": error_message,
            "label_count": len(labels),
            "labels": labels,
            "quality": quality_metrics
        }
        
        return self.results["cluster_labeling"]
    
    def test_path_narratives(self, sample_size: int = 2) -> Dict[str, Any]:
        """
        Test path narrative generation.
        
        Args:
            sample_size: Number of paths to sample for testing (0 for all)
            
        Returns:
            Dictionary with path narrative test results
        """
        # First ensure we have cluster labels
        if not self.results["cluster_labeling"] or not self.results["cluster_labeling"]["success"]:
            if self.debug:
                print("Running cluster labeling first...")
            self.test_cluster_labeling()
        
        start_time = time.time()
        success = False
        error_message = None
        narratives = {}
        
        try:
            # Extract clusters and paths
            centroids = extract_centroids(self.test_data)
            paths = extract_paths(self.test_data)
            cluster_labels = self.results["cluster_labeling"]["labels"]
            
            # Limit to sample size if needed
            if sample_size > 0 and sample_size < len(paths):
                sampled_paths = {k: v for i, (k, v) in enumerate(paths.items()) if i < sample_size}
            else:
                sampled_paths = paths
            
            # Extract fragmentation scores if available
            fragmentation_scores = {}
            if "similarity" in self.test_data and "fragmentation_scores" in self.test_data["similarity"]:
                scores = self.test_data["similarity"]["fragmentation_scores"].get("scores", [])
                fragmentation_scores = {
                    i: score for i, score in enumerate(scores) 
                    if i in sampled_paths
                }
            
            # Generate narratives
            narratives = self.analyzer.generate_path_narratives_sync(
                sampled_paths,
                cluster_labels,
                centroids,
                fragmentation_scores=fragmentation_scores
            )
            
            success = len(narratives) > 0
            
        except Exception as e:
            error_message = str(e)
            if self.debug:
                import traceback
                traceback.print_exc()
        
        elapsed_time = time.time() - start_time
        
        # Basic quality check
        quality_metrics = {}
        if success:
            quality_metrics = self._evaluate_narrative_quality(narratives, paths)
        
        self.results["path_narratives"] = {
            "success": success,
            "elapsed_time": elapsed_time,
            "error": error_message,
            "narrative_count": len(narratives),
            "narratives": narratives,
            "quality": quality_metrics
        }
        
        return self.results["path_narratives"]
    
    def test_bias_audit(self) -> Dict[str, Any]:
        """
        Test bias audit functionality.
        
        Returns:
            Dictionary with bias audit test results
        """
        # Only run if we have demographic info
        demographic_info = self._extract_demographic_info()
        if not demographic_info:
            return {
                "success": False,
                "error": "No demographic information available for bias audit",
                "skipped": True
            }
        
        start_time = time.time()
        success = False
        error_message = None
        bias_report = {}
        bias_analysis = ""
        
        try:
            # Extract paths and cluster labels
            paths = extract_paths(self.test_data)
            
            # Ensure we have cluster labels
            if not self.results["cluster_labeling"] or not self.results["cluster_labeling"]["success"]:
                if self.debug:
                    print("Running cluster labeling first...")
                self.test_cluster_labeling()
            
            cluster_labels = self.results["cluster_labeling"]["labels"]
            
            # Determine demographic columns
            demographic_columns = self._identify_demographic_columns()
            
            # Generate bias report
            bias_report = generate_bias_report(
                paths=paths,
                demographic_info=demographic_info,
                demographic_columns=demographic_columns,
                cluster_labels=cluster_labels
            )
            
            # Generate bias analysis
            bias_analysis = analyze_bias_with_llm(self.analyzer, bias_report)
            
            success = len(bias_report) > 0 and len(bias_analysis) > 0
            
        except Exception as e:
            error_message = str(e)
            if self.debug:
                import traceback
                traceback.print_exc()
        
        elapsed_time = time.time() - start_time
        
        self.results["bias_audit"] = {
            "success": success,
            "elapsed_time": elapsed_time,
            "error": error_message,
            "bias_report": bias_report,
            "bias_analysis": bias_analysis
        }
        
        return self.results["bias_audit"]
    
    def _evaluate_label_quality(self, labels: Dict[str, str]) -> Dict[str, Any]:
        """
        Evaluate the quality of cluster labels.
        
        Args:
            labels: Dictionary of cluster labels
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            "avg_length": sum(len(label) for label in labels.values()) / max(1, len(labels)),
            "empty_labels": sum(1 for label in labels.values() if not label),
            "has_variety": len(set(labels.values())) / max(1, len(labels)) > 0.5,
        }
        return metrics
    
    def _evaluate_narrative_quality(self, narratives: Dict[int, str], paths: Dict[int, List[str]]) -> Dict[str, Any]:
        """
        Evaluate the quality of path narratives.
        
        Args:
            narratives: Dictionary of path narratives
            paths: Dictionary of paths
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            "avg_length": sum(len(narrative) for narrative in narratives.values()) / max(1, len(narratives)),
            "empty_narratives": sum(1 for narrative in narratives.values() if not narrative),
            "mentions_path": sum(1 for path_id, narrative in narratives.items() 
                               if path_id in paths and any(cluster_id in narrative for cluster_id in paths[path_id]))
        }
        return metrics
    
    def _extract_demographic_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Extract demographic info from test data.
        
        Returns:
            Dictionary with demographic information
        """
        demographic_info = {}
        if "path_archetypes" in self.test_data:
            for i, archetype in enumerate(self.test_data["path_archetypes"]):
                if "demo_stats" in archetype:
                    demographic_info[i] = archetype["demo_stats"]
        return demographic_info
    
    def _identify_demographic_columns(self) -> List[str]:
        """
        Identify demographic columns based on dataset.
        
        Returns:
            List of demographic column names
        """
        demo_columns = []
        dataset_name = self.test_data.get("dataset", "").lower()
        
        if dataset_name == "titanic":
            demo_columns = ["sex", "age", "pclass"]
        elif dataset_name == "adult":
            demo_columns = ["sex", "race", "age", "education"]
        elif dataset_name == "heart":
            demo_columns = ["sex", "age"]
        
        # Filter to include only available columns
        available_demo_cols = []
        demo_info = self._extract_demographic_info()
        
        for path_id, demos in demo_info.items():
            for col in demo_columns:
                if col in demos and col not in available_demo_cols:
                    available_demo_cols.append(col)
        
        return available_demo_cols
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of analysis tests.
        
        Returns:
            Dictionary with test summary
        """
        return {
            "provider": self.analyzer.provider,
            "model": self.analyzer.model,
            "cluster_labeling": self._summarize_test_result("cluster_labeling"),
            "path_narratives": self._summarize_test_result("path_narratives"),
            "bias_audit": self._summarize_test_result("bias_audit"),
            "cache_performance": self._summarize_test_result("cache_performance") if "cache_performance" in self.results else None,
            "batch_performance": self._summarize_test_result("batch_performance") if "batch_performance" in self.results else None,
            "prompt_optimization": self._summarize_test_result("prompt_optimization") if "prompt_optimization" in self.results else None,
            "overall_success": all(
                result and result["success"] 
                for result in [
                    self.results["cluster_labeling"],
                    self.results["path_narratives"]
                ]
                if result is not None
            )
        }
    
    def test_prompt_optimization(self) -> Dict[str, Any]:
        """
        Test prompt optimization.
        
        Returns:
            Dictionary with prompt optimization test results
        """
        start_time = time.time()
        success = False
        error_message = None
        optimization_stats = {}
        
        try:
            # Import the prompt optimizer
            from concept_fragmentation.llm.prompt_optimizer import (
                optimize_cluster_label_prompt, 
                optimize_path_narrative_prompt,
                estimate_token_savings
            )
            
            # Sample prompts for testing
            cluster_prompt = """You are an AI expert analyzing neural network activations. 
            
            Given the centroid of a cluster in activation space, provide a concise, meaningful label that captures the concept this cluster might represent.
            
            Cluster centroid top 10 features:
            feature_1: 0.7531
            feature_2: -0.4299
            feature_3: 0.6012
            feature_4: 0.3856
            feature_5: -0.5129
            feature_6: 0.2811
            feature_7: -0.3301
            feature_8: 0.4772
            feature_9: -0.1908
            feature_10: 0.6233
            
            Your label should be concise (1-5 words) and interpretable. Focus on potential semantic meaning rather than technical details.
            
            Cluster label:"""
            
            path_prompt = """You are an AI expert analyzing neural network activation patterns.
            
            Generate a clear, insightful narrative that explains the following path through activation clusters in a neural network. 
            Focus on the conceptual meaning and the potential decision process represented by this path.
            
            Path: L1C0 (Abstract Shapes) → L2C3 (Edge Detection) → L3C1 (Object Recognition) → L4C2 (Higher-level Features)
            
            Based on this information, write a concise narrative (2-4 sentences) that explains:
            1. What concepts or features this path might represent
            2. How the concept evolves or transforms across layers (especially if there are convergence points)
            3. Any potential insights about the model's decision-making process
            
            Your explanation should be clear and insightful without being overly technical.
            
            Path narrative:"""
            
            # Optimize prompts at different levels
            optimized_levels = {}
            for level in range(1, 4):
                optimized_cluster = optimize_cluster_label_prompt(cluster_prompt, level)
                optimized_path = optimize_path_narrative_prompt(path_prompt, level)
                
                # Calculate token savings
                cluster_savings = estimate_token_savings(cluster_prompt, optimized_cluster)
                path_savings = estimate_token_savings(path_prompt, optimized_path)
                
                optimized_levels[level] = {
                    "cluster": {
                        "original_length": len(cluster_prompt),
                        "optimized_length": len(optimized_cluster),
                        "savings_percent": cluster_savings["percent_reduction"]
                    },
                    "path": {
                        "original_length": len(path_prompt),
                        "optimized_length": len(optimized_path),
                        "savings_percent": path_savings["percent_reduction"]
                    }
                }
            
            # Test with real client if available
            token_comparison = None
            if hasattr(self.analyzer, 'optimize_prompts'):
                # Get token counts for original and optimized prompts
                original_temp = self.analyzer.optimize_prompts
                original_level = self.analyzer.optimization_level
                
                # No optimization first
                self.analyzer.optimize_prompts = False
                response1 = self.analyzer.generate_with_cache(
                    prompt="What is cluster analysis? Answer in one sentence.",
                    temperature=0.0
                )
                tokens_original = response1.prompt_tokens
                
                # With optimization
                self.analyzer.optimize_prompts = True
                self.analyzer.optimization_level = 2  # Moderate optimization
                response2 = self.analyzer.generate_with_cache(
                    prompt="What is cluster analysis? Answer in one sentence.",
                    temperature=0.0
                )
                tokens_optimized = response2.prompt_tokens
                
                # Restore original settings
                self.analyzer.optimize_prompts = original_temp
                self.analyzer.optimization_level = original_level
                
                if tokens_original > 0 and tokens_optimized > 0:
                    token_comparison = {
                        "original_tokens": tokens_original,
                        "optimized_tokens": tokens_optimized,
                        "tokens_saved": tokens_original - tokens_optimized,
                        "percent_reduction": round((tokens_original - tokens_optimized) / tokens_original * 100, 1)
                    }
            
            # Collect results
            optimization_stats = {
                "by_level": optimized_levels,
                "token_comparison": token_comparison
            }
            
            success = True
            
        except Exception as e:
            error_message = str(e)
            if self.debug:
                import traceback
                traceback.print_exc()
        
        elapsed_time = time.time() - start_time
        
        self.results["prompt_optimization"] = {
            "success": success,
            "elapsed_time": elapsed_time,
            "error": error_message,
            "optimization_stats": optimization_stats
        }
        
        return self.results["prompt_optimization"]
    
    def test_batch_performance(self) -> Dict[str, Any]:
        """
        Test batch processing performance.
        
        Returns:
            Dictionary with batch performance test results
        """
        start_time = time.time()
        success = False
        error_message = None
        batch_stats = {}
        
        try:
            # Create a small batch of similar prompts
            prompts = [
                "What is a cluster centroid? Explain briefly.",
                "What is activation space in neural networks? Explain briefly.",
                "What is dimensionality reduction? Explain briefly.",
                "What is t-SNE visualization? Explain briefly."
            ]
            
            # First run sequentially to establish baseline
            sequential_start = time.time()
            
            for i, prompt in enumerate(prompts):
                response = self.analyzer.generate_with_cache(prompt, temperature=0.0)
            
            sequential_time = time.time() - sequential_start
            
            # Now run with batch processing (concurrency)
            # Clear cache first to ensure fair comparison
            if hasattr(self.analyzer, "clear_cache"):
                self.analyzer.clear_cache()
            
            batch_start = time.time()
            
            # Run with batch processor
            # We'll create a batch directly with the prompts
            batch_items = [(i, {"prompt": prompt, "temperature": 0.0}) 
                          for i, prompt in enumerate(prompts)]
            
            import asyncio
            
            async def run_batch():
                from concept_fragmentation.llm.batch_processor import process_items_concurrently
                
                async def generate_text(params):
                    return await self.analyzer._generate_with_cache(**params)
                
                return await process_items_concurrently(
                    batch_items, 
                    generate_text,
                    max_concurrency=len(prompts),
                    report_progress=self.debug
                )
            
            loop = asyncio.get_event_loop()
            batch_results = loop.run_until_complete(run_batch())
            
            batch_time = time.time() - batch_start
            
            # Calculate speedup
            if sequential_time > 0:
                speedup = sequential_time / max(0.001, batch_time)
            else:
                speedup = 1.0
            
            batch_stats = {
                "sequential_time": sequential_time,
                "batch_time": batch_time,
                "speedup_factor": speedup,
                "batch_size": len(prompts),
                "all_succeeded": all(not isinstance(result, Exception) for result in batch_results.values())
            }
            
            success = batch_stats["all_succeeded"] and speedup > 1.0
            
        except Exception as e:
            error_message = str(e)
            if self.debug:
                import traceback
                traceback.print_exc()
        
        elapsed_time = time.time() - start_time
        
        self.results["batch_performance"] = {
            "success": success,
            "elapsed_time": elapsed_time,
            "error": error_message,
            "batch_stats": batch_stats
        }
        
        return self.results["batch_performance"]
        
    def test_cache_performance(self) -> Dict[str, Any]:
        """
        Test cache performance.
        
        Returns:
            Dictionary with cache performance test results
        """
        start_time = time.time()
        success = False
        error_message = None
        cache_stats = None
        
        try:
            # Get initial cache stats
            cache_stats_before = self.analyzer.get_cache_stats()
            
            # Generate a few prompts for testing
            prompt1 = "What is Explainable Threshold Similarity? Explain in one sentence."
            prompt2 = "What does intra-class distance measure in neural networks? Be concise."
            
            # Test with prompts
            response1 = self.analyzer.generate_with_cache(prompt1, temperature=0.0)
            response2 = self.analyzer.generate_with_cache(prompt2, temperature=0.0)
            
            # Test cache hit (should be fast)
            cache_hit_start = time.time()
            response1_cached = self.analyzer.generate_with_cache(prompt1, temperature=0.0)
            cache_hit_time = time.time() - cache_hit_start
            
            # Get final cache stats
            cache_stats = self.analyzer.get_cache_stats()
            
            success = True
            
        except Exception as e:
            error_message = str(e)
            if self.debug:
                import traceback
                traceback.print_exc()
        
        elapsed_time = time.time() - start_time
        
        self.results["cache_performance"] = {
            "success": success,
            "elapsed_time": elapsed_time,
            "error": error_message,
            "cache_stats": cache_stats,
            "cache_hit_time": cache_hit_time if success else None
        }
        
        return self.results["cache_performance"]
    
    def _summarize_test_result(self, test_name: str) -> Dict[str, Any]:
        """
        Summarize a test result for the summary.
        
        Args:
            test_name: Name of the test result to summarize
            
        Returns:
            Summarized test result
        """
        result = self.results[test_name]
        if not result:
            return {"success": False, "error": "Test not run", "elapsed_time": 0}
        
        return {
            "success": result["success"],
            "error": result["error"],
            "elapsed_time": result["elapsed_time"]
        }


def load_cluster_paths_data(file_path):
    """Load cluster paths data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_centroids(data):
    """Extract centroids from cluster paths data."""
    centroids = {}
    
    # Check if unique_centroids exists in the data
    if "unique_centroids" in data:
        unique_centroids = data["unique_centroids"]
        id_mapping = data["id_mapping"]
        
        # Map unique IDs to cluster IDs
        for unique_id, centroid in unique_centroids.items():
            if unique_id in id_mapping:
                mapping = id_mapping[unique_id]
                layer_name = mapping["layer_name"]
                original_id = mapping["original_id"]
                cluster_id = f"{layer_name}C{original_id}"
                
                # Convert centroid to numpy array
                centroids[cluster_id] = np.array(centroid)
    
    return centroids

def extract_paths(data):
    """Extract paths from cluster paths data."""
    paths = {}
    
    # Extract path archetypes
    if "path_archetypes" in data:
        archetypes = data["path_archetypes"]
        
        for i, archetype in enumerate(archetypes):
            if "path" in archetype:
                path_str = archetype["path"]
                
                # Split the path string into cluster IDs
                path_parts = path_str.split("→")
                
                # Get layer information
                layers = data.get("layers", [])
                
                # Create the path with full cluster IDs
                path = []
                for j, part in enumerate(path_parts):
                    if j < len(layers):
                        layer = layers[j]
                        cluster_id = f"{layer}C{part}"
                        path.append(cluster_id)
                    else:
                        # Fallback if layer information is missing
                        path.append(f"L{j}C{part}")
                
                paths[i] = path
    
    return paths

def parse_arguments():
    """Parse command-line arguments for the test script."""
    parser = argparse.ArgumentParser(description="Test LLM integration with cluster paths")
    
    # Provider selection
    parser.add_argument(
        "--providers",
        type=str,
        default="grok",
        help="Comma-separated list of providers to test (grok,claude,openai,gemini,all)"
    )
    
    # Test mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "comprehensive"],
        default="quick",
        help="Test mode: quick (basic tests) or comprehensive (all tests)"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        default="titanic",
        help="Dataset to use for testing (e.g., titanic, adult, heart)"
    )
    
    # Seed selection
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset"
    )
    
    # Output file
    parser.add_argument(
        "--output",
        type=str,
        default="llm_test_results.json",
        help="Output file for test results"
    )
    
    # Cache control
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached LLM responses (default: False)"
    )
    
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=None,
        help="Cache time-to-live in seconds (default: no expiration)"
    )
    
    parser.add_argument(
        "--memory-only-cache",
        action="store_true",
        help="Use memory-only caching (don't persist to disk)"
    )
    
    # Prompt optimization
    parser.add_argument(
        "--optimize-prompts",
        action="store_true",
        help="Enable prompt optimization to reduce token usage"
    )
    
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Prompt optimization level (1=minimal, 2=moderate, 3=aggressive)"
    )
    
    # Debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate providers
    if args.providers.lower() == "all":
        args.provider_list = ["grok", "claude", "openai", "gemini"]
    else:
        args.provider_list = [p.strip() for p in args.providers.split(",")]
    
    return args

def find_test_data(dataset, seed):
    """Find test data file with fallback paths."""
    # List of potential paths to check
    potential_paths = [
        f"data/cluster_paths/{dataset}_seed_{seed}_paths.json",
        f"../data/cluster_paths/{dataset}_seed_{seed}_paths.json",
        f"../../data/cluster_paths/{dataset}_seed_{seed}_paths.json",
        f"/mnt/c/Repos/ConceptualFragmentationInLLMsAnalysisAndVisualization/data/cluster_paths/{dataset}_seed_{seed}_paths.json"
    ]
    
    # Check each path
    for path in potential_paths:
        if os.path.exists(path):
            return path
    
    # If we reach here, no file was found
    return None

def run_provider_tests(provider_names: List[str], args) -> Dict[str, Any]:
    """
    Run tests for specified providers.
    
    Args:
        provider_names: List of provider names to test
        args: Command-line arguments
        
    Returns:
        Dictionary with provider test results
    """
    results = {}
    
    for provider in provider_names:
        print(f"Testing provider: {provider}...")
        
        # Initialize tester
        tester = ProviderTester(
            provider_name=provider,
            use_cache=args.use_cache,
            cache_ttl=args.cache_ttl,
            memory_only_cache=args.memory_only_cache,
            debug=args.debug
        )
        
        # Test connection
        print(f"  Testing connection...")
        connection_result = tester.test_connection()
        if not connection_result["success"]:
            print(f"  ❌ Connection failed: {connection_result['error']}")
            results[provider] = tester.get_summary()
            continue
        
        print(f"  ✅ Connection successful ({connection_result['elapsed_time']:.2f}s)")
        
        # Test basic generation
        print(f"  Testing basic generation...")
        generation_result = tester.test_basic_generation()
        if not generation_result["success"]:
            print(f"  ❌ Generation failed: {generation_result['error']}")
            results[provider] = tester.get_summary()
            continue
        
        response_quality = "✅" if generation_result.get("response_matches", False) else "❌"
        print(f"  {response_quality} Generation successful ({generation_result['elapsed_time']:.2f}s)")
        print(f"    Response: \"{generation_result['response_text']}\"")
        
        # Record results
        results[provider] = tester.get_summary()
    
    return results


def run_analysis_tests(provider_name: str, test_data: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Run analysis tests for specified provider.
    
    Args:
        provider_name: Provider name to use for testing
        test_data: Test data dictionary
        args: Command-line arguments
        
    Returns:
        Dictionary with analysis test results
    """
    print(f"Running analysis tests with {provider_name}...")
    
    # Initialize provider tester first to get analyzer
    provider_tester = ProviderTester(
        provider_name=provider_name,
        use_cache=args.use_cache,
        cache_ttl=args.cache_ttl,
        memory_only_cache=args.memory_only_cache,
        debug=args.debug
    )
    
    # Test connection
    connection_result = provider_tester.test_connection()
    if not connection_result["success"]:
        print(f"❌ Connection failed: {connection_result['error']}")
        return {
            "provider": provider_name,
            "connection_error": connection_result['error'],
            "tests_run": False
        }
    
    # Initialize analysis tester
    analysis_tester = AnalysisTester(
        analyzer=provider_tester.analyzer,
        test_data=test_data,
        debug=args.debug
    )
    
    # Test cluster labeling
    print("  Testing cluster labeling...")
    label_result = analysis_tester.test_cluster_labeling(
        sample_size=3 if args.mode == "quick" else 0
    )
    
    if label_result["success"]:
        print(f"  ✅ Generated {label_result['label_count']} cluster labels ({label_result['elapsed_time']:.2f}s)")
        if args.debug:
            for cluster_id, label in list(label_result["labels"].items())[:3]:
                print(f"    {cluster_id}: {label}")
    else:
        print(f"  ❌ Cluster labeling failed: {label_result['error']}")
    
    # Test path narratives
    print("  Testing path narratives...")
    narrative_result = analysis_tester.test_path_narratives(
        sample_size=2 if args.mode == "quick" else 0
    )
    
    if narrative_result["success"]:
        print(f"  ✅ Generated {narrative_result['narrative_count']} path narratives ({narrative_result['elapsed_time']:.2f}s)")
        if args.debug:
            for path_id, narrative in list(narrative_result["narratives"].items())[:1]:
                print(f"    Path {path_id}: {narrative[:100]}...")
    else:
        print(f"  ❌ Path narratives failed: {narrative_result['error']}")
    
    # Test bias audit (only in comprehensive mode)
    if args.mode == "comprehensive":
        print("  Testing bias audit...")
        bias_result = analysis_tester.test_bias_audit()
        
        if bias_result.get("skipped", False):
            print("  ⚠️ Bias audit skipped: No demographic information available")
        elif bias_result["success"]:
            print(f"  ✅ Generated bias report and analysis ({bias_result['elapsed_time']:.2f}s)")
            if args.debug:
                print(f"    Analysis excerpt: {bias_result['bias_analysis'][:100]}...")
        else:
            print(f"  ❌ Bias audit failed: {bias_result['error']}")
        
        # Test cache performance
        print("  Testing cache performance...")
        cache_result = analysis_tester.test_cache_performance()
        
        if cache_result["success"]:
            # If cache hit time is available and cache is enabled
            if cache_result.get("cache_hit_time") and cache_result.get("cache_stats", {}).get("enabled", False):
                cache_hit_time = cache_result["cache_hit_time"] * 1000  # Convert to ms
                print(f"  ✅ Cache performance: {cache_hit_time:.2f}ms for cache hit")
                print(f"  Cache size: {cache_result['cache_stats'].get('size', 0)} items")
            else:
                print(f"  ✅ Cache test completed, but caching is disabled")
        else:
            print(f"  ❌ Cache performance test failed: {cache_result['error']}")
        
        # Test batch processing performance
        print("  Testing batch processing performance...")
        batch_result = analysis_tester.test_batch_performance()
        
        if batch_result["success"]:
            batch_stats = batch_result.get("batch_stats", {})
            speedup = batch_stats.get("speedup_factor", 1.0)
            sequential = batch_stats.get("sequential_time", 0)
            batch_time = batch_stats.get("batch_time", 0)
            batch_size = batch_stats.get("batch_size", 0)
            
            print(f"  ✅ Batch processing: {speedup:.2f}x speedup for {batch_size} requests")
            print(f"     Sequential: {sequential:.2f}s, Batch: {batch_time:.2f}s")
        else:
            print(f"  ❌ Batch performance test failed: {batch_result['error']}")
            
        # Test prompt optimization
        print("  Testing prompt optimization...")
        optimization_result = analysis_tester.test_prompt_optimization()
        
        if optimization_result["success"]:
            opt_stats = optimization_result.get("optimization_stats", {})
            level2_stats = opt_stats.get("by_level", {}).get(2, {})
            
            # Get cluster and path stats for level 2 (moderate)
            if level2_stats:
                cluster_savings = level2_stats.get("cluster", {}).get("savings_percent", 0)
                path_savings = level2_stats.get("path", {}).get("savings_percent", 0)
                print(f"  ✅ Prompt optimization: {cluster_savings:.1f}% savings for cluster labels, {path_savings:.1f}% for path narratives")
            
            # Show real token comparison if available
            token_comparison = opt_stats.get("token_comparison")
            if token_comparison:
                savings = token_comparison.get("percent_reduction", 0)
                tokens_saved = token_comparison.get("tokens_saved", 0)
                print(f"     Actual token savings: {tokens_saved} tokens ({savings:.1f}%)")
        else:
            print(f"  ❌ Prompt optimization test failed: {optimization_result['error']}")
    
    return analysis_tester.get_summary()


def main():
    """Main test function."""
    # Parse command-line arguments
    args = parse_arguments()
    
    print(f"=== LLM Integration Test ({args.mode} mode) ===")
    print(f"Testing providers: {', '.join(args.provider_list)}")
    print(f"Dataset: {args.dataset}, Seed: {args.seed}")
    
    # Find and load the cluster paths file
    file_path = find_test_data(args.dataset, args.seed)
    if not file_path:
        print(f"Error: Could not find test data for {args.dataset} with seed {args.seed}")
        print("Please ensure the dataset file exists or specify a different dataset.")
        return 1
    
    print(f"\nLoading cluster paths data from {file_path}")
    data = load_cluster_paths_data(file_path)
    
    # Extract centroids and paths
    centroids = extract_centroids(data)
    paths = extract_paths(data)
    
    # Verify centroids
    print(f"Found {len(centroids)} unique centroids")
    for i, (cluster_id, centroid) in enumerate(list(centroids.items())[:3]):
        print(f"Centroid {i+1}: {cluster_id} - Shape: {centroid.shape}, First few values: {centroid[:5]}")
    
    # Verify paths
    print(f"\nFound {len(paths)} unique paths")
    for i, (path_id, path) in enumerate(list(paths.items())[:3]):
        print(f"Path {i+1}: {' → '.join(path)}")
    
    # Check if LLM module is available
    llm_available = "ClusterAnalysis" in globals()
    
    if not llm_available:
        print("\nLLM module not available. Running basic data verification only.")
        
        # Verify we can mock LLM analysis by creating a simple cluster label
        print("\nSimulating LLM analysis:")
        for i, (cluster_id, centroid) in enumerate(list(centroids.items())[:3]):
            # Generate a mock cluster label based on the centroid values
            avg_val = np.mean(centroid)
            if avg_val > 0.5:
                mock_label = "High-Value Cluster"
            elif avg_val > 0:
                mock_label = "Moderate-Value Cluster"
            else:
                mock_label = "Low-Value Cluster"
            
            print(f"Mock LLM label for {cluster_id}: {mock_label} (avg_val={avg_val:.4f})")
        
        print("\nBasic integration test completed successfully!")
        return 0
    
    # Prepare result dictionary
    test_results = {
        "test_timestamp": datetime.now().isoformat(),
        "test_mode": args.mode,
        "dataset": args.dataset,
        "seed": args.seed,
        "provider_results": {},
        "analysis_results": {}
    }
    
    # Run provider tests if LLM module is available
    print("\n=== Provider Connectivity Tests ===")
    provider_results = run_provider_tests(args.provider_list, args)
    test_results["provider_results"] = provider_results
    
    # Determine which providers passed connection test
    successful_providers = [
        provider for provider, result in provider_results.items()
        if result["overall_success"]
    ]
    
    if not successful_providers:
        print("\nNo providers passed connection tests. Cannot continue with analysis tests.")
        
        # Save results
        print(f"\nSaving test results to {args.output}")
        with open(args.output, 'w') as f:
            json.dump(test_results, f, indent=2)
            
        return 1
    
    print(f"\n{len(successful_providers)} providers passed connection tests: {', '.join(successful_providers)}")
    
    # Run analysis tests with the first successful provider
    primary_provider = successful_providers[0]
    print(f"\n=== Analysis Tests with {primary_provider} ===")
    
    analysis_results = run_analysis_tests(primary_provider, data, args)
    test_results["analysis_results"] = analysis_results
    
    # Save results
    print(f"\nSaving test results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\n=== Test Summary ===")
    print(f"Providers tested: {len(args.provider_list)}")
    print(f"Providers with successful connection: {len(successful_providers)}")
    
    if analysis_results.get("tests_run", False) is not False:
        print("\nAnalysis tests:")
        print(f"Cluster labeling: {'✅ Passed' if analysis_results['cluster_labeling']['success'] else '❌ Failed'}")
        print(f"Path narratives: {'✅ Passed' if analysis_results['path_narratives']['success'] else '❌ Failed'}")
        
        if args.mode == "comprehensive":
            bias_result = analysis_results.get("bias_audit", {})
            if bias_result.get("success", False):
                print(f"Bias audit: ✅ Passed")
            elif bias_result.get("error", "").startswith("No demographic"):
                print(f"Bias audit: ⚠️ Skipped (no demographic data)")
            else:
                print(f"Bias audit: ❌ Failed")
                
            # Cache performance
            cache_result = analysis_results.get("cache_performance", {})
            if cache_result and cache_result.get("success", False):
                print(f"Cache performance: ✅ Passed (cache hit: {cache_result.get('cache_hit_time', 0)*1000:.2f}ms)")
            elif cache_result:
                print(f"Cache performance: ❌ Failed")
                
            # Batch processing performance
            batch_result = analysis_results.get("batch_performance", {})
            if batch_result and batch_result.get("success", False):
                batch_stats = batch_result.get("batch_stats", {})
                speedup = batch_stats.get("speedup_factor", 1.0)
                print(f"Batch performance: ✅ Passed ({speedup:.2f}x speedup)")
            elif batch_result:
                print(f"Batch performance: ❌ Failed")
                
            # Prompt optimization
            opt_result = analysis_results.get("prompt_optimization", {})
            if opt_result and opt_result.get("success", False):
                opt_stats = opt_result.get("optimization_stats", {})
                token_comparison = opt_stats.get("token_comparison", {})
                if token_comparison and token_comparison.get("percent_reduction", 0) > 0:
                    savings = token_comparison.get("percent_reduction", 0)
                    print(f"Prompt optimization: ✅ Passed ({savings:.1f}% token reduction)")
                else:
                    print(f"Prompt optimization: ✅ Passed")
            elif opt_result:
                print(f"Prompt optimization: ❌ Failed")
    
    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())