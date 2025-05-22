"""
LLM Path Analysis - Generate narratives for archetypal paths in neural networks.

This script:
1. Extracts real path data from cluster path files
2. Calculates meaningful statistics for each archetypal path 
3. Constructs detailed prompts with relevant context
4. Leverages LLM APIs to generate interpretive narratives
5. Stores and analyzes the responses for inclusion in the paper

Usage:
  python llm_path_analysis.py --dataset titanic --seed 0 --output_dir results/llm
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import argparse
from collections import Counter
import statistics
from datetime import datetime
import re

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import LLM client utilities
from concept_fragmentation.llm.client import LLMClient
from concept_fragmentation.llm.factory import create_llm_client
from concept_fragmentation.llm.analysis import ClusterAnalysis  # For cluster labeling


class PathDataProcessor:
    """Process and extract meaningful information from path data files."""
    
    def __init__(self, data_path: str):
        """
        Initialize the processor with path to data file.
        
        Args:
            data_path: Path to the JSON file containing path data
        """
        self.data_path = data_path
        self.path_data = self._load_path_data()
        self.dataset_name = self.path_data.get("dataset", "unknown")
        self.seed = self.path_data.get("seed", 0)
        self.paths = self.path_data.get("unique_paths", [])
        self.layers = self.path_data.get("layers", [])
        self.human_readable_paths = self.path_data.get("human_readable_paths", [])
        
        # Additional properties filled later
        self.path_counts = None
        self.path_stats = {}
        self.demographic_data = None
        self.path_archetypes = self.path_data.get("path_archetypes", [])
        
    def _load_path_data(self) -> Dict:
        """Load path data from JSON file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Path data file not found: {self.data_path}")
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def count_paths(self) -> Dict[str, int]:
        """
        Count the frequency of each unique path.
        
        Returns:
            Dictionary mapping human-readable path to count
        """
        if not self.human_readable_paths:
            return {}
        
        path_counts = Counter(self.human_readable_paths)
        self.path_counts = dict(path_counts.most_common())
        
        return self.path_counts
    
    def load_demographic_data(self, demographics_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load demographic data for the dataset.
        
        Args:
            demographics_path: Path to demographic data CSV file
                              (if None, will try to infer from dataset name)
        
        Returns:
            DataFrame with demographic information
        """
        # If path not provided, try to infer based on dataset name
        if demographics_path is None:
            if self.dataset_name == "titanic":
                demographics_path = os.path.join("data", "titanic", "train.csv")
            elif self.dataset_name == "heart":
                demographics_path = os.path.join("data", "heart", "heart.csv")
        
        # If inferred path doesn't exist, raise an error - no synthetic data
        if not os.path.exists(demographics_path):
            raise FileNotFoundError(
                f"Required demographics file missing: {demographics_path}"
            )
            
        # Load real demographic data
        print(f"Loading demographics from {demographics_path}")
        self.demographic_data = pd.read_csv(demographics_path)
        
        return self.demographic_data
    
    def calculate_path_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics for each unique path.
        
        Returns:
            Dictionary mapping path string to statistics
        """
        if self.path_counts is None:
            self.count_paths()
        
        # If demographics not loaded, load them
        if self.demographic_data is None:
            self.load_demographic_data()
        
        # Calculate statistics for each path
        path_stats = {}
        
        # First, identify indices for each path
        path_indices = {}
        for i, path_str in enumerate(self.human_readable_paths):
            if path_str not in path_indices:
                path_indices[path_str] = []
            path_indices[path_str].append(i)
        
        # Use path archetypes if available (they contain pre-computed statistics)
        archetype_by_path = {}
        for archetype in self.path_archetypes:
            path = archetype.get("path", "")
            if path:
                archetype_by_path[path] = archetype
        
        # For each path, calculate statistics
        for path_str, indices in path_indices.items():
            # If we have a pre-computed archetype, use it
            if path_str in archetype_by_path:
                archetype = archetype_by_path[path_str]
                path_stats[path_str] = {
                    "count": archetype.get("count", len(indices)),
                    "percentage": archetype.get("percentage", 100 * len(indices) / len(self.paths)),
                    "member_indices": archetype.get("member_indices", indices[:50]),
                    "demo_stats": archetype.get("demo_stats", {}),
                }
                
                # Add all relevant statistics from the archetype
                # Also, populate demo_stats for backward compatibility
                current_demo_stats = path_stats[path_str].get("demo_stats", {})
                for key, val in archetype.items():
                    # Copy all keys that are not part of basic path identification or structure
                    if key not in {"path", "numeric_path", "member_indices", "count", "percentage", "demo_stats"}:
                        path_stats[path_str][key] = val
                        
                        # Add to demo_stats if it's a demographic-like key for backward compatibility
                        # This heuristic covers common patterns for demographic and target-related stats.
                        if key.endswith(("_mean", "_std", "_min", "_max", "_distribution", "_rate", "_percentage")):
                             current_demo_stats[key] = val
                
                # Ensure demo_stats in the archetype (if any) are preserved if not overwritten
                # and that new ones are added.
                # The original archetype["demo_stats"] might contain other specific fields.
                # We prioritize individual keys copied above if there's a name clash.
                original_archetype_demo_stats = archetype.get("demo_stats", {})
                for ds_key, ds_val in original_archetype_demo_stats.items():
                    if ds_key not in current_demo_stats: # Avoid overwriting already processed/copied keys
                        current_demo_stats[ds_key] = ds_val
                        
                path_stats[path_str]["demo_stats"] = current_demo_stats
                
                continue
            
            # If no pre-computed archetype, calculate statistics from demographic data
            if len(indices) == 0:
                continue
                
            # Get demographic data for this path
            if self.demographic_data is not None and len(self.demographic_data) >= max(indices) + 1:
                path_demos = self.demographic_data.iloc[indices].copy()
                
                # Calculate basic statistics
                stats = {
                    "count": len(indices),
                    "percentage": 100 * len(indices) / len(self.paths),
                    "member_indices": indices[:50]  # Limit to first 50 indices
                }
                
                # Calculate demographic statistics based on dataset
                if self.dataset_name == "titanic":
                    # Survived column may be lowercase in CSV
                    survived_col = "Survived" if "Survived" in path_demos.columns else "survived" if "survived" in path_demos.columns else None
                    if survived_col:
                        stats["survived_rate"] = float(path_demos[survived_col].mean())
                    
                    # Get class distribution
                    pclass_col = "Pclass" if "Pclass" in path_demos.columns else "pclass" if "pclass" in path_demos.columns else None
                    if pclass_col:
                        stats["class_distribution"] = path_demos[pclass_col].value_counts(normalize=True).to_dict()
                        # Convert keys to strings for JSON serialization
                        stats["class_distribution"] = {str(k): float(v) for k, v in stats["class_distribution"].items()}
                    
                    # Get gender distribution
                    sex_col = "Sex" if "Sex" in path_demos.columns else "sex" if "sex" in path_demos.columns else None
                    if sex_col:
                        stats["gender_distribution"] = path_demos[sex_col].value_counts(normalize=True).to_dict()
                        stats["male_percentage"] = float(100 * (path_demos[sex_col].str.lower() == "male").mean())
                    
                    # Get age statistics
                    age_col = "Age" if "Age" in path_demos.columns else "age" if "age" in path_demos.columns else None
                    if age_col:
                        valid_ages = path_demos[age_col].dropna()
                        if len(valid_ages) > 0:
                            stats["age_mean"] = float(valid_ages.mean())
                            stats["age_std"] = float(valid_ages.std())
                            stats["age_min"] = float(valid_ages.min())
                            stats["age_max"] = float(valid_ages.max())
                    
                    # Get fare statistics
                    fare_col = "Fare" if "Fare" in path_demos.columns else "fare" if "fare" in path_demos.columns else None
                    if fare_col:
                        valid_fares = path_demos[fare_col].dropna()
                        if len(valid_fares) > 0:
                            stats["fare_mean"] = float(valid_fares.mean())
                            stats["fare_std"] = float(valid_fares.std())
                            stats["fare_min"] = float(valid_fares.min())
                            stats["fare_max"] = float(valid_fares.max())
                            
                elif self.dataset_name == "heart":
                    # Get disease status rate
                    if "target" in path_demos.columns:
                        stats["disease_rate"] = float(path_demos["target"].mean())
                    
                    # Get gender distribution
                    if "sex" in path_demos.columns:
                        stats["male_percentage"] = float(100 * path_demos["sex"].mean())
                    
                    # Get age statistics
                    if "age" in path_demos.columns:
                        stats["age_mean"] = float(path_demos["age"].mean())
                        stats["age_std"] = float(path_demos["age"].std())
                    
                    # Get chest pain type distribution
                    if "cp" in path_demos.columns:
                        stats["chest_pain_distribution"] = path_demos["cp"].value_counts(normalize=True).to_dict()
                        stats["chest_pain_distribution"] = {str(k): float(v) for k, v in stats["chest_pain_distribution"].items()}
                    
                    # Get cholesterol statistics
                    if "chol" in path_demos.columns:
                        stats["chol_mean"] = float(path_demos["chol"].mean())
                        stats["chol_std"] = float(path_demos["chol"].std())
                
                path_stats[path_str] = stats
            else:
                # If demographic data not available or indices out of bounds, provide basic stats
                path_stats[path_str] = {
                    "count": len(indices),
                    "percentage": 100 * len(indices) / len(self.paths),
                    "member_indices": indices[:50]  # Limit to first 50 indices
                }
        
        # Sort paths by frequency
        self.path_stats = {k: path_stats[k] for k in sorted(path_stats, key=lambda x: path_stats[x]["count"], reverse=True)}
        
        return self.path_stats
    
    def calculate_similarity_metrics(self) -> Dict[str, Any]:
        """
        Calculate similarity metrics between clusters across layers.
        
        Returns:
            Dictionary with similarity metrics
        """
        # Check if similarity data is already in the path data
        if "similarity" in self.path_data and self.path_data["similarity"]:
            return self.path_data["similarity"]
        
        # If not, return empty dict (would normally calculate here)
        return {}
    
    def get_top_paths(self, n: int = 3) -> List[str]:
        """
        Get the top N most frequent paths.
        
        Args:
            n: Number of paths to return
            
        Returns:
            List of human-readable path strings
        """
        if self.path_counts is None:
            self.count_paths()
        
        return list(self.path_counts.keys())[:n]
    
    def get_path_statistics_summary(self, path: str) -> str:
        """
        Generate a human-readable summary of path statistics.
        
        Args:
            path: Human-readable path string
            
        Returns:
            Formatted string with path statistics
        """
        if self.path_stats is None or len(self.path_stats) == 0:
            self.calculate_path_statistics()
        
        if path not in self.path_stats:
            return f"No statistics available for path {path}"
        
        stats = self.path_stats[path]
        
        # Create summary based on dataset
        if self.dataset_name == "titanic":
            summary = [
                f"Path: {path}",
                f"Count: {stats['count']} passengers ({stats['percentage']:.2f}% of dataset)",
            ]
            
            # Add survival information if available
            if "survived_rate" in stats:
                survived_pct = stats["survived_rate"] * 100
                summary.append(f"Survival Rate: {survived_pct:.1f}%")
            
            # Add demographic information
            if "age_mean" in stats:
                summary.append(f"Age: mean {stats['age_mean']:.1f} years (±{stats['age_std']:.1f})")
            
            if "male_percentage" in stats:
                summary.append(f"Gender: {stats['male_percentage']:.1f}% male, {100-stats['male_percentage']:.1f}% female")
            
            if "class_distribution" in stats:
                class_dist = stats["class_distribution"]
                first_class = float(class_dist.get("1", 0)) * 100
                second_class = float(class_dist.get("2", 0)) * 100
                third_class = float(class_dist.get("3", 0)) * 100
                summary.append(f"Class: {first_class:.1f}% first, {second_class:.1f}% second, {third_class:.1f}% third")
            
            if "fare_mean" in stats:
                summary.append(f"Fare: mean £{stats['fare_mean']:.2f} (±{stats['fare_std']:.2f})")
            
        elif self.dataset_name == "heart":
            summary = [
                f"Path: {path}",
                f"Count: {stats['count']} patients ({stats['percentage']:.2f}% of dataset)",
            ]
            
            # Add disease information if available
            if "disease_rate" in stats:
                disease_pct = stats["disease_rate"] * 100
                summary.append(f"Heart Disease Rate: {disease_pct:.1f}%")
            
            # Add demographic information
            if "age_mean" in stats:
                summary.append(f"Age: mean {stats['age_mean']:.1f} years (±{stats['age_std']:.1f})")
            
            if "male_percentage" in stats:
                summary.append(f"Gender: {stats['male_percentage']:.1f}% male, {100-stats['male_percentage']:.1f}% female")
            
            if "chest_pain_distribution" in stats:
                cp_dist = stats["chest_pain_distribution"]
                cp_types = [
                    f"Type 0 (typical angina): {float(cp_dist.get('0', 0))*100:.1f}%",
                    f"Type 1 (atypical angina): {float(cp_dist.get('1', 0))*100:.1f}%",
                    f"Type 2 (non-anginal pain): {float(cp_dist.get('2', 0))*100:.1f}%",
                    f"Type 3 (asymptomatic): {float(cp_dist.get('3', 0))*100:.1f}%"
                ]
                summary.append(f"Chest Pain: {', '.join(cp_types)}")
            
            if "chol_mean" in stats:
                summary.append(f"Cholesterol: mean {stats['chol_mean']:.1f} mg/dl (±{stats['chol_std']:.1f})")
        
        else:
            # Generic summary for unknown datasets
            summary = [
                f"Path: {path}",
                f"Count: {stats['count']} samples ({stats['percentage']:.2f}% of dataset)",
            ]
            
            # Add any available statistics
            for key, value in stats.items():
                if key not in ["count", "percentage", "member_indices", "demo_stats"]:
                    if isinstance(value, float):
                        summary.append(f"{key}: {value:.2f}")
                    else:
                        summary.append(f"{key}: {value}")
        
        return "\n".join(summary)
    
    def parse_similarity_data(self) -> Dict[str, Any]:
        """
        Parse similarity data from the path data file.
        
        Returns:
            Dictionary with parsed similarity information
        """
        similarity_info = {}
        
        # Check if similarity data is in the path data
        if "similarity" not in self.path_data or not self.path_data["similarity"]:
            return similarity_info
        
        similarity_data = self.path_data["similarity"]
        
        # Extract fragmentation scores if available
        if "fragmentation_scores" in similarity_data:
            frag_scores = similarity_data["fragmentation_scores"]
            similarity_info["fragmentation"] = {
                "mean": frag_scores.get("mean", 0),
                "median": frag_scores.get("median", 0),
                "std": frag_scores.get("std", 0),
                "high_threshold": frag_scores.get("high_threshold", 0),
                "low_threshold": frag_scores.get("low_threshold", 0),
                "scores": frag_scores.get("scores", [])
            }
            
            # Get high and low fragmentation paths
            similarity_info["high_fragmentation_paths"] = frag_scores.get("high_fragmentation_paths", [])
            similarity_info["low_fragmentation_paths"] = frag_scores.get("low_fragmentation_paths", [])
        
        # Extract convergent paths if available
        if "convergent_paths" in similarity_data:
            similarity_info["convergent_paths"] = similarity_data["convergent_paths"]
        
        # Extract human-readable convergent paths if available
        if "human_readable_convergent_paths" in similarity_data:
            similarity_info["human_readable_convergent_paths"] = similarity_data["human_readable_convergent_paths"]
        
        # ------------------------------------------------------------------
        # NEW: pull additional fragmentation metrics (entropy & angle)
        # ------------------------------------------------------------------
        if "metrics" in self.path_data:
            metrics_block = self.path_data["metrics"]

            if "entropy_fragmentation" in metrics_block:
                similarity_info["entropy_fragmentation"] = metrics_block["entropy_fragmentation"]

            if "angle_fragmentation" in metrics_block:
                similarity_info["angle_fragmentation"] = metrics_block["angle_fragmentation"]

            if "k_star" in metrics_block:
                similarity_info["k_star"] = metrics_block["k_star"]
        
        return similarity_info


class PromptGenerator:
    """Generate prompts for LLM based on path data."""
    
    def __init__(self, processor: PathDataProcessor, cluster_labels: Dict[str, str]):
        """
        Initialize with a path data processor and cluster labels.
        
        Args:
            processor: PathDataProcessor instance with loaded data
            cluster_labels: Dictionary mapping cluster IDs to labels
        """
        self.processor = processor
        self.cluster_labels = cluster_labels or {}
        
    def _cluster_label_string(self, path: str) -> str:
        """Return comma-separated label descriptions for clusters in a path."""
        parts = []
        for cluster_id in path.split("→"):
            label = self.cluster_labels.get(cluster_id.strip(), "(unlabeled)")
            parts.append(f"{cluster_id}: {label}")
        return ", ".join(parts)
    
    def generate_technical_analysis_prompt(self, path: str) -> str:
        """
        Generate a technical analysis prompt for the given path.
        
        Args:
            path: Human-readable path string
            
        Returns:
            Prompt for technical analysis
        """
        # Get statistics for the path
        path_summary = self.processor.get_path_statistics_summary(path)
        
        # Build cluster label map
        label_line = self._cluster_label_string(path)
        
        # Get similarity information
        similarity_info = self.processor.parse_similarity_data()
        
        # Create the prompt
        prompt = [
            "# Technical Analysis of Neural Network Path",
            "",
            "## Path Statistics",
            path_summary,
            f"\nCluster Descriptors: {label_line}",
            "",
            "## Model Overview",
            f"Dataset: {self.processor.dataset_name}",
            f"Model: 3-layer feedforward neural network",
            f"Layers analyzed: {', '.join(self.processor.layers)}",
            "",
            "## Task",
            "Analyze this path from a technical perspective. Consider:",
            "1. What does the path's structure tell us about the model's internal representation?",
            "2. How does this path relate to the overall dataset distribution?",
            "3. Are there any technical indicators of interesting phenomena (convergence, fragmentation)?",
            "",
            "Provide a detailed technical analysis that would be useful for a machine learning researcher."
        ]
        
        # Add fragmentation information if available
        if similarity_info and "fragmentation" in similarity_info:
            frag = similarity_info["fragmentation"]
            
            # Check if this path has high or low fragmentation
            path_idx = self.processor.human_readable_paths.index(path) if path in self.processor.human_readable_paths else -1
            
            is_high_frag = path_idx in similarity_info.get("high_fragmentation_paths", [])
            is_low_frag = path_idx in similarity_info.get("low_fragmentation_paths", [])
            
            frag_level = "HIGH" if is_high_frag else "LOW" if is_low_frag else "AVERAGE"
            
            # Retrieve numeric fragmentation score if available
            frag_scores = similarity_info["fragmentation"].get("scores", [])
            path_score = None
            if 0 <= path_idx < len(frag_scores):
                path_score = frag_scores[path_idx]
            
            prompt.append("")
            prompt.append("## Fragmentation Analysis")
            prompt.append(f"Overall dataset fragmentation: mean={frag['mean']:.3f}, median={frag['median']:.3f}, std={frag['std']:.3f}")
            prompt.append(f"High fragmentation threshold: {frag['high_threshold']:.3f}")
            prompt.append(f"Low fragmentation threshold: {frag['low_threshold']:.3f}")
            prompt.append(f"This path shows {frag_level} fragmentation.")
            if path_score is not None:
                prompt.append(f"Path fragmentation score: {path_score:.3f}")
        
        # Add entropy & angle fragmentation metrics if available
        if similarity_info:
            if "entropy_fragmentation" in similarity_info:
                e_frag = similarity_info["entropy_fragmentation"]
                prompt.append("")
                prompt.append("## Entropy Fragmentation (label cohesion)")
                prompt.append(f"Mean entropy across layers: {e_frag.get('mean', float('nan')):.3f} ± {e_frag.get('std', float('nan')):.3f}")
            if "angle_fragmentation" in similarity_info:
                a_frag = similarity_info["angle_fragmentation"]
                prompt.append("## Sub-space Angle Fragmentation (class separation)")
                prompt.append(f"Mean angle across layers: {a_frag.get('mean', float('nan')):.1f}° ± {a_frag.get('std', float('nan')):.1f}°")
            if "k_star" in similarity_info:
                ks = similarity_info["k_star"]
                k_vals = list(ks.values())
                if k_vals:
                    prompt.append(f"Optimal cluster count k*: mean {np.mean(k_vals):.1f} (range {min(k_vals)}–{max(k_vals)})")
        
        return "\n".join(prompt)
    
    def generate_fairness_analysis_prompt(self, path: str) -> str:
        """
        Generate a fairness analysis prompt for the given path.
        
        Args:
            path: Human-readable path string
            
        Returns:
            Prompt for fairness analysis
        """
        # Get statistics for the path
        path_summary = self.processor.get_path_statistics_summary(path)
        
        # Build cluster label map
        label_line = self._cluster_label_string(path)
        
        # Create the prompt
        prompt = [
            "# Fairness Analysis of Neural Network Path",
            "",
            "## Path Demographics",
            path_summary,
            f"\nCluster Descriptors: {label_line}",
            "",
            "## Task",
            "Analyze this path from a fairness and ethics perspective. Consider:",
            "1. Does this path show evidence of disparate treatment of different demographic groups?",
            "2. How might the model's internal clustering reflect societal biases?",
            "3. What are the ethical implications of how different groups are processed?",
            "4. What interventions might address any fairness concerns you identify?",
            "",
            "Provide a nuanced fairness analysis that considers both technical and ethical factors."
        ]
        
        # Add dataset-specific guidance
        if self.processor.dataset_name == "titanic":
            prompt.append("")
            prompt.append("## Historical Context")
            prompt.append("The Titanic disaster had documented disparities in survival rates based on class, gender, and age.")
            prompt.append("First-class passengers had significantly higher survival rates than third-class.")
            prompt.append("Women and children were prioritized for lifeboats ('women and children first' policy).")
        
        elif self.processor.dataset_name == "heart":
            prompt.append("")
            prompt.append("## Medical Context")
            prompt.append("Heart disease diagnosis has documented disparities across gender and age groups.")
            prompt.append("Women are historically underdiagnosed for heart conditions compared to men.")
            prompt.append("Different age groups may present with varying symptoms that affect diagnosis accuracy.")
        
        return "\n".join(prompt)
    
    def generate_narrative_synthesis_prompt(self, path: str) -> str:
        """
        Generate a narrative synthesis prompt for the given path.
        
        Args:
            path: Human-readable path string
            
        Returns:
            Prompt for narrative synthesis
        """
        # Get statistics for the path
        path_summary = self.processor.get_path_statistics_summary(path)
        
        # Build cluster label map
        label_line = self._cluster_label_string(path)
        
        # Create the prompt
        prompt = [
            "# Narrative Synthesis for Neural Network Path",
            "",
            "## Path Information",
            path_summary,
            f"\nCluster Descriptors: {label_line}",
            "",
            "## Task",
            "You are an interpretability analyst narrating how a neural network processes data. Based on the path information provided:",
            "",
            "1. Create a vivid, human-readable story that explains this path through the network's latent space",
            "2. Highlight the meaning of transitions between clusters (what features or concepts does the model focus on?)",
            "3. Explain what this path reveals about the model's decision-making process",
            "4. Include relevant statistical details while maintaining an engaging narrative style",
            "",
            "Your narrative should be compelling and insightful, translating technical patterns into an interpretable story."
        ]
        
        # Add dataset-specific guidance
        if self.processor.dataset_name == "titanic":
            prompt.append("")
            prompt.append("## Narrative Framing")
            prompt.append("Frame your narrative around passengers' journey through the Titanic disaster.")
            prompt.append("Each layer represents a different stage of the model's internal reasoning about survival likelihood.")
            prompt.append("Transitions between clusters represent shifts in how the model perceives passenger characteristics.")
        
        elif self.processor.dataset_name == "heart":
            prompt.append("")
            prompt.append("## Narrative Framing")
            prompt.append("Frame your narrative around patients' journey through the diagnostic process.")
            prompt.append("Each layer represents a different stage of the model's internal reasoning about disease likelihood.")
            prompt.append("Transitions between clusters represent shifts in how the model interprets patient symptoms and risk factors.")
        
        return "\n".join(prompt)
    
    def generate_comprehensive_analysis_prompt(self, path: str) -> str:
        """
        Generate a comprehensive analysis prompt combining technical, fairness, and narrative elements.
        
        Args:
            path: Human-readable path string
            
        Returns:
            Comprehensive prompt
        """
        # Get statistics for the path
        path_summary = self.processor.get_path_statistics_summary(path)
        
        # Build cluster label map
        label_line = self._cluster_label_string(path)
        
        # Get similarity information
        similarity_info = self.processor.parse_similarity_data()
        
        # Get path fragmentation level
        path_idx = self.processor.human_readable_paths.index(path) if path in self.processor.human_readable_paths else -1
        frag_level = "Unknown"
        path_score = None
        
        if similarity_info and "fragmentation" in similarity_info:
            is_high_frag = path_idx in similarity_info.get("high_fragmentation_paths", [])
            is_low_frag = path_idx in similarity_info.get("low_fragmentation_paths", [])
            frag_level = "High" if is_high_frag else "Low" if is_low_frag else "Average"
            
            # Numeric score
            frag_scores = similarity_info["fragmentation"].get("scores", [])
            if 0 <= path_idx < len(frag_scores):
                path_score = frag_scores[path_idx]
        
        # Create the prompt
        prompt = [
            "# Comprehensive Neural Network Path Analysis",
            "",
            "## Path Information",
            path_summary,
            f"Fragmentation Level: {frag_level}",
            f"\nCluster Descriptors: {label_line}",
            "",
            "## Technical Context",
            f"Model: 3-layer feedforward neural network trained on the {self.processor.dataset_name} dataset",
            f"Layers analyzed: {', '.join(self.processor.layers)}",
            "Clusters in each layer represent distinct activation patterns the model has learned",
            "Transitions between clusters show how the model refines its understanding across layers",
            "",
            "## Analysis Request",
            "You are an interpretability analyst providing insight into neural network behavior. Please provide a comprehensive analysis of this path that includes:",
            "",
            "### 1. Technical Insights (30%)",
            "- What does this path's structure reveal about the model's internal representations?",
            "- How do the transitions between clusters reflect the model's feature learning process?",
            "- What computational patterns (convergence, divergence, oscillation) does this path exhibit?",
            "",
            "### 2. Fairness Assessment (30%)",
            "- Does this path show evidence of disparate treatment across demographic groups?",
            "- What societal biases might be reflected in this model pathway?",
            "- What interventions could address any fairness concerns?",
            "",
            "### 3. Narrative Synthesis (40%)",
            "- Create a vivid, compelling story explaining this path through the network's latent space",
            "- Highlight the meaning of transitions between clusters",
            "- Explain what this path reveals about the model's decision-making logic",
            "- Translate technical patterns into an accessible, interpretable narrative",
            "",
            "Your analysis should be data-driven, insightful, and balance technical rigor with interpretability."
        ]
        
        # Add numeric score if exists
        if path_score is not None:
            prompt.append(f"Path fragmentation score: {path_score:.3f}")
        
        # Add entropy & angle metrics if present
        if similarity_info:
            if "entropy_fragmentation" in similarity_info:
                e_frag = similarity_info["entropy_fragmentation"]
                prompt.append(f"Mean entropy across layers: {e_frag.get('mean', float('nan')):.3f} ± {e_frag.get('std', float('nan')):.3f}")
            if "angle_fragmentation" in similarity_info:
                a_frag = similarity_info["angle_fragmentation"]
                prompt.append(f"Mean sub-space angle across layers: {a_frag.get('mean', float('nan')):.1f}° ± {a_frag.get('std', float('nan')):.1f}°")
            if "k_star" in similarity_info:
                ks = similarity_info["k_star"]
                k_vals = list(ks.values())
                if k_vals:
                    prompt.append(f"Optimal cluster count k*: mean {np.mean(k_vals):.1f} (range {min(k_vals)}–{max(k_vals)})")
        
        return "\n".join(prompt)

    def generate_holistic_system_prompt(self, num_top_archetypes_to_detail: int = 5) -> str:
        """
        Generate a prompt for a holistic analysis of the entire system of paths.
        
        Args:
            num_top_archetypes_to_detail: How many of the most frequent archetypal paths
                                          to include with their full statistical summaries.
                                          
        Returns:
            A comprehensive prompt string.
        """
        # 1. Gather all necessary data components
        dataset_name = self.processor.dataset_name
        model_description = "3-layer feedforward neural network" # Assuming this is constant
        layers_analyzed = ", ".join(self.processor.layers)
        
        # All cluster labels
        cluster_labels_section = ["## Cluster Labels Descriptions"]
        if self.cluster_labels:
            for cid, label in sorted(self.cluster_labels.items()):
                cluster_labels_section.append(f"- {cid}: {label}")
        else:
            cluster_labels_section.append("No cluster labels were generated.")
        cluster_labels_section_str = "\\n".join(cluster_labels_section)

        # Overall fragmentation and similarity metrics
        similarity_info = self.processor.parse_similarity_data()
        metrics_section = ["## Overall System Metrics"]
        if similarity_info and "fragmentation" in similarity_info:
            frag = similarity_info["fragmentation"]
            metrics_section.append(f"- Path-Centroid Fragmentation (FC): mean={frag.get('mean', 'N/A'):.3f}, median={frag.get('median', 'N/A'):.3f}, std={frag.get('std', 'N/A'):.3f}")
            metrics_section.append(f"  (High FC threshold: {frag.get('high_threshold', 'N/A'):.3f}, Low FC threshold: {frag.get('low_threshold', 'N/A'):.3f})")
        
        if similarity_info and "entropy_fragmentation" in similarity_info:
            e_frag = similarity_info["entropy_fragmentation"]
            metrics_section.append(f"- Intra-Class Cluster Entropy (CE): mean={e_frag.get('mean', 'N/A'):.3f} ± {e_frag.get('std', 'N/A'):.3f} (across layers)")
        
        if similarity_info and "angle_fragmentation" in similarity_info:
            a_frag = similarity_info["angle_fragmentation"]
            metrics_section.append(f"- Sub-space Angle Fragmentation (SA): mean={a_frag.get('mean', 'N/A'):.1f}° ± {a_frag.get('std', 'N/A'):.1f}° (across layers)")
            
        if similarity_info and "k_star" in similarity_info:
            ks = similarity_info["k_star"]
            k_vals = list(ks.values())
            if k_vals:
                metrics_section.append(f"- Optimal Cluster Count k*: mean {np.mean(k_vals):.1f} (range {min(k_vals)}–{max(k_vals)}) (across layers)")
        metrics_section_str = "\\n".join(metrics_section)

        # Add details for top N archetypes
        prompt_lines = [
            f"# Holistic System Analysis: {dataset_name.capitalize()} Dataset",
            f"Model: {model_description}",
            f"Layers Analyzed: {layers_analyzed}",
            "",
            cluster_labels_section_str,
            "",
            metrics_section_str,
            "",
            "## Analysis Task:",
            "You are an expert interpretability analyst. Based on all the provided system-level information (cluster labels, overall metrics, and detailed statistics for top archetypal paths):",
            "1.  **Holistic Interpretation**: Provide a high-level interpretation of how this neural network processes data.", 
            "    *   What general strategies or stages of refinement (abstraction, convergence) does the model seem to employ across its layers?",
            "    *   How do the overall system metrics (FC, CE, SA, k*) support your interpretation of the model's processing style?",
            "2.  **Comparative Path Analysis**: Contrast the characteristics of the detailed archetypal paths provided.",
            "    *   How do differences in their constituent cluster *labels* (semantic meanings) and their sequence explain variations in their demographic statistics and outcome rates?",
            "    *   Are there specific cluster *concepts* that act as critical decision points or divergence hubs for different groups of inputs?",
            "3.  **Key Decision-Making Insights**: What are the most significant insights into the model's decision-making logic?",
            "    *   Identify any particularly surprising or counter-intuitive relationships between paths, demographics, cluster concepts, and outcomes.",
            "4.  **System-Wide Fairness & Bias Assessment**: Discuss potential system-wide fairness concerns or biases.",
            "    *   Look for intersectional demographic patterns. Are certain combined demographic groups disproportionately affected by specific pathways or outcomes?",
            "    *   If certain groups are routed through paths with distinct fragmentation characteristics, what might this imply about the model's processing for them?",
            "5.  **Synthesis and Story**: Weave these insights into a coherent narrative that explains the model's overall behavior as a system.",
            "    *   Your narrative should explicitly touch upon several of the top archetypal paths provided, illustrating their individual significance within the broader system.",
            "    *   For at least 3-5 of the most distinct or impactful archetypes from the top 7, provide a brief (2-3 sentence) \'mini-story\' or characterization, explaining its typical journey by referencing its key demographic features and the semantic meaning of its clusters.",
            "    *   Conclude with an overall \'moral of the story\' for how this model operates on the given dataset.",
            "",
            "Structure your response clearly using well-organized Markdown. Use headings for each of the 5 analysis points. Within each point, use bullet points or short paragraphs for clarity."
        ]
        
        # Detailed statistics for top N archetypes
        top_archetypes = self.processor.path_data.get("path_archetypes", [])[:num_top_archetypes_to_detail]

        for i, archetype in enumerate(top_archetypes):
            prompt_lines.append("")
            prompt_lines.append(f"## Detailed Statistics for Top {i+1} Archetypal Paths")
            prompt_lines.append("")
            prompt_lines.append(f"### Archetype {i+1}: {archetype.get('path', 'Unknown Path')}")
            # Include all relevant stats from the archetype
            for key, val in archetype.items():
                if key not in ["member_indices", "numeric_path", "path"]: # Exclude verbose/internal keys
                    if isinstance(val, float):
                        # Format floats nicely (e.g., percentages, means)
                        if "rate" in key.lower() or "percentage" in key.lower() or key.endswith("%"):
                            prompt_lines.append(f"{key.replace('_', ' ').title()}: {val:.1%}")
                        else:
                            prompt_lines.append(f"{key.replace('_', ' ').title()}: {val:.2f}")
                    elif isinstance(val, dict):
                        # For dictionaries (like sex_distribution), format them nicely
                        dict_str = ", ".join([f"{k.title()}: {v:.1%}" for k, v in val.items()])
                        prompt_lines.append(f"{key.replace('_', ' ').title()}: {dict_str}")
                    else:
                        prompt_lines.append(f"{key.replace('_', ' ').title()}: {val}")

        return "\n".join(prompt_lines)


class LLMPathAnalyzer:
    """
    Analyze neural network paths using LLMs to generate interpretive narratives.
    """
    
    def __init__(self, processor: PathDataProcessor, output_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            processor: PathDataProcessor instance with loaded data
            output_dir: Directory to save outputs
        """
        self.processor = processor
        self.output_dir = output_dir
        
        # Prepare cluster labels first
        self.cluster_labels = self._ensure_cluster_labels()

        # Prompt generator now has access to labels
        self.prompt_generator = PromptGenerator(processor, self.cluster_labels)

        self.llm_client = self._create_llm_client()
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def _create_llm_client(self) -> LLMClient:
        """Create LLM client – always OpenAI, fail fast if key missing."""
        # Always use OpenAI – rely on api_keys.py or env var for key
        return create_llm_client("openai")
    
    def analyze_path(self, path: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze a single path using the specified analysis type.
        
        Args:
            path: Human-readable path string
            analysis_type: Type of analysis to perform
                          ("technical", "fairness", "narrative", "comprehensive")
        
        Returns:
            Dictionary with analysis results
        """
        # Generate the appropriate prompt
        if analysis_type == "technical":
            prompt = self.prompt_generator.generate_technical_analysis_prompt(path)
        elif analysis_type == "fairness":
            prompt = self.prompt_generator.generate_fairness_analysis_prompt(path)
        elif analysis_type == "narrative":
            prompt = self.prompt_generator.generate_narrative_synthesis_prompt(path)
        else:  # comprehensive is default
            prompt = self.prompt_generator.generate_comprehensive_analysis_prompt(path)
        
        # Get LLM response (sync)
        try:
            response_obj = self.llm_client.generate_sync(prompt)
            # The OpenAI client returns an LLMResponse; keep only the text for JSON serialization
            response_text = (
                response_obj.text if hasattr(response_obj, "text") else str(response_obj)
            )
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            response_text = f"Error: {str(e)}"
        
        # Store the result
        result = {
            "path": path,
            "analysis_type": analysis_type,
            "prompt": prompt,
            "response": response_text,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to results dictionary
        if path not in self.results:
            self.results[path] = {}
        
        self.results[path][analysis_type] = result
        
        return result
    
    def analyze_top_paths(self, n: Optional[int] = 3, analysis_type: str = "comprehensive") -> Dict[str, Dict[str, Any]]:
        """
        Analyze the top N most frequent paths.
        
        Args:
            n: Number of paths to analyze
            analysis_type: Type of analysis to perform
        
        Returns:
            Dictionary mapping paths to analysis results
        """
        # Use all paths if n is None
        if n is None:
            top_paths = self.processor.human_readable_paths
        else:
            top_paths = self.processor.get_top_paths(n)
        
        # Analyze each path
        top_path_results = {}
        for path in top_paths:
            result = self.analyze_path(path, analysis_type)
            top_path_results[path] = result
        
        return top_path_results
    
    def analyze_selected_paths(self, paths: List[str], analysis_type: str = "comprehensive") -> Dict[str, Dict[str, Any]]:
        """
        Analyze specific paths.
        
        Args:
            paths: List of path strings to analyze
            analysis_type: Type of analysis to perform
        
        Returns:
            Dictionary mapping paths to analysis results
        """
        # Analyze each path
        path_results = {}
        for path in paths:
            result = self.analyze_path(path, analysis_type)
            path_results[path] = result
        
        return path_results
    
    def analyze_system_holistically(self, num_top_archetypes_in_prompt: int = 5) -> Dict[str, Any]:
        """
        Analyze the entire system using a single comprehensive prompt.
        
        Args:
            num_top_archetypes_in_prompt: Number of top archetypal paths to include with full
                                          details in the holistic prompt.
        
        Returns:
            Dictionary with the holistic analysis results (prompt and response).
        """
        print(f"Generating holistic system prompt with details for top {num_top_archetypes_in_prompt} archetypes...")
        prompt = self.prompt_generator.generate_holistic_system_prompt(num_top_archetypes_to_detail=num_top_archetypes_in_prompt)
        
        print("Sending holistic prompt to LLM...")
        try:
            response_obj = self.llm_client.generate_sync(prompt)
            response_text = (
                response_obj.text if hasattr(response_obj, "text") else str(response_obj)
            )
        except Exception as e:
            print(f"Error generating LLM response for holistic analysis: {e}")
            response_text = f"Error: {str(e)}"
        
        # Store the result
        holistic_result = {
            "prompt_type": "holistic_system_analysis",
            "num_detailed_archetypes_in_prompt": num_top_archetypes_in_prompt,
            "prompt": prompt,
            "response": response_text,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results['holistic_analysis'] = holistic_result # Store under a specific key
        print("Holistic analysis received from LLM.")
        
        return holistic_result
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save analysis results to a JSON file.
        
        Args:
            filename: Name of the output file (if None, generates automatically)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            # Generate standard filename (no timestamp)
            filename = f"{self.processor.dataset_name}_seed_{self.processor.seed}_analysis.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Get formatted narrative report
        formatted_report = self.get_formatted_narrative_report()
        
        # Convert markdown to LaTeX
        latex_report = self._markdown_to_latex(formatted_report)
        
        # Create a combined results structure
        combined_results = {
            "path_narratives": {},
            "cluster_labels": self.cluster_labels,
            "bias_report": {},
            "full_report_latex": latex_report
        }
        
        # Extract individual narratives and other data from results (per-path)
        for path, analyses in self.results.items():
            if path == 'holistic_analysis': # Skip our new special key for this loop
                continue
            # Get comprehensive or narrative analysis
            analysis = analyses.get("comprehensive", analyses.get("narrative"))
            if analysis:
                combined_results["path_narratives"][path] = analysis["response"]

        # Add holistic analysis if it exists
        if 'holistic_analysis' in self.results:
            combined_results["holistic_system_analysis"] = self.results['holistic_analysis']
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        
        # Also save narrative report as markdown
        md_path = os.path.join(self.output_dir, f"{self.processor.dataset_name}_seed_{self.processor.seed}_narrative_report.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(formatted_report)
        
        print(f"Narrative report saved to {md_path}")
        
        # Also save narrative report as LaTeX
        tex_path = os.path.join(self.output_dir, f"{self.processor.dataset_name}_report.tex")
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_report)
        
        print(f"LaTeX report saved to {tex_path}")
        
        return output_path
    
    def _markdown_to_latex(self, md: str) -> str:
        """
        Convert markdown formatted text to LaTeX.
        
        Args:
            md: Markdown text
            
        Returns:
            LaTeX formatted text
        """
        # Escape special LaTeX chars first (except heading markers)
        md = md.replace("%", "\\%").replace("&", "\\&")

        lines = md.splitlines()
        converted = []
        in_code = False
        for line in lines:
            # Handle fenced code blocks
            if line.strip().startswith("```"):
                if not in_code:
                    converted.append("\\begin{verbatim}")
                    in_code = True
                else:
                    converted.append("\\end{verbatim}")
                    in_code = False
                continue
            if in_code:
                # keep line verbatim (escape backslashes)
                converted.append(line.replace("\\", "\\\\"))
                continue

            # Headings
            m = re.match(r"^(#+)\s+(.*)$", line)
            if m:
                level = len(m.group(1))
                content = m.group(2).strip()
                if level == 1:
                    converted.append(f"\\section*{{{content}}}")
                elif level == 2:
                    converted.append(f"\\subsection*{{{content}}}")
                else:
                    converted.append(f"\\subsubsection*{{{content}}}")
                continue

            # bold **text**
            line = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", line)
            # arrow symbol
            line = line.replace("→", "$\\rightarrow$")
            # underscores escape
            line = line.replace("_", "\\_")

            converted.append(line)

        # Close any open code block
        if in_code:
            converted.append("\\end{verbatim}")

        return "\n".join(converted)
    
    def get_formatted_narrative_report(self) -> str:
        """
        Generate a formatted narrative report for inclusion in the paper.
        Handles both per-path and holistic analysis results.
        """
        if not self.results:
            return "No analysis results available."

        # Check if holistic analysis was performed
        if 'holistic_analysis' in self.results and isinstance(self.results['holistic_analysis'], dict):
            holistic_data = self.results['holistic_analysis']
            response = holistic_data.get("response", "No response found in holistic analysis.")
            num_detailed = holistic_data.get("num_detailed_archetypes_in_prompt", "N/A")
            
            report = [
                f"# Holistic System Analysis Report: {self.processor.dataset_name.capitalize()} Dataset",
                f"(Details for top {num_detailed} archetypes were included in the prompt)",
                "",
                response # Assuming the LLM provides a Markdown formatted report
            ]
            return "\\n".join(report)

        # Fallback to per-path report generation (existing logic)
        report = [
            "# LLM-Generated Neural Network Path Narratives",
            "",
            f"## Dataset: {self.processor.dataset_name.capitalize()}",
            "",
            "This report presents LLM-generated narratives that interpret the archetypal paths",
            "identified in our neural network analysis. Each narrative translates technical",
            "patterns into an accessible explanation of the model's internal logic.",
            ""
        ]
        
        for path_idx, (path, analyses) in enumerate(self.results.items()):
            if path == 'holistic_analysis': # Should already be handled above, but good for safety
                continue

            analysis = analyses.get("comprehensive", analyses.get("narrative"))
            if not analysis or not isinstance(analysis, dict):
                continue
            
            report.append(f"## Archetypal Path {path_idx + 1}: {path}")
            report.append("")
            
            stats_summary = self.processor.get_path_statistics_summary(path)
            report.append("<details>")
            report.append("<summary>Path Statistics</summary>")
            report.append("")
            report.append("```")
            report.append(stats_summary)
            report.append("```")
            report.append("</details>")
            report.append("")
            
            response_text = analysis.get("response", "No response text found for this path.")
            
            if analysis.get("analysis_type") == "comprehensive":
                if "### 3. Narrative Synthesis" in response_text:
                    narrative_parts = response_text.split("### 3. Narrative Synthesis")
                    if len(narrative_parts) > 1:
                        narrative = narrative_parts[1]
                        if "###" in narrative: # Trim if next section like ### 4. exists
                            narrative = narrative.split("###")[0]
                        response_text = narrative.strip()
            
            report.append("### LLM Narrative")
            report.append("")
            report.append(response_text)
            report.append("")
            report.append("---")
            report.append("")
        
        return "\\n".join(report)
    
    # ------------------------------------------------------------------
    # Cluster labeling helper
    # ------------------------------------------------------------------
    def _ensure_cluster_labels(self) -> Dict[str, str]:
        """Return existing or newly generated cluster labels, now based on demographic profiles."""
        # Try to load labels if they are already in previously saved results (e.g., from a prior run)
        existing_labels = {}
        candidate_json = os.path.join(
            self.output_dir,
            f"{self.processor.dataset_name}_seed_{self.processor.seed}_analysis.json",
        )
        if os.path.isfile(candidate_json):
            try:
                with open(candidate_json) as f:
                    prior_results = json.load(f)
                existing_labels = prior_results.get("cluster_labels", {})
                if existing_labels: # Basic check: if labels exist, assume they are good for now
                    print(f"Loaded {len(existing_labels)} existing cluster labels from {candidate_json}")
                    return existing_labels
            except Exception as e:
                print(f"Could not load existing labels from {candidate_json}: {e}")

        # If no valid existing labels, generate them based on demographic profiles.
        print("No valid pre-existing labels found. Generating new labels based on demographic profiles...")

        # Ensure demographic data is loaded in the processor
        if self.processor.demographic_data is None:
            self.processor.load_demographic_data() # Uses default CSV paths
        
        if self.processor.demographic_data is None:
            print("Demographic data could not be loaded. Cannot generate profile-based cluster labels.")
            return {}

        all_layer_clusters_data = self.processor.path_data.get("layer_clusters", {})
        id_to_layer_cluster_map = self.processor.path_data.get("id_to_layer_cluster", {})

        if not all_layer_clusters_data or not id_to_layer_cluster_map:
            print("Missing layer_clusters or id_to_layer_cluster in path_data. Cannot generate profiles.")
            return {}

        cluster_profiles_for_labeling = {}

        # Iterate through unique cluster IDs that need labeling
        # These unique IDs come from `assign_unique_cluster_ids` in the original cluster_paths.py
        # and are the keys in `unique_centroids` and `id_to_layer_cluster`.
        unique_cluster_ids = self.processor.path_data.get("unique_centroids", {}).keys()

        for unique_id_str in unique_cluster_ids: # unique_id_str is e.g., "0", "1", ...
            if unique_id_str not in id_to_layer_cluster_map: # Should map e.g. "0" to ["layer1", 0, 0]
                print(f"Warning: Unique ID {unique_id_str} not in id_to_layer_cluster map. Skipping profile generation.")
                continue

            layer_name, original_cluster_idx_in_layer, _ = id_to_layer_cluster_map[unique_id_str]
            
            layer_specific_data = all_layer_clusters_data.get(layer_name)
            if not layer_specific_data or "labels" not in layer_specific_data:
                print(f"Warning: No label data for layer {layer_name}. Skipping profile for cluster {unique_id_str}.")
                continue

            # Get all sample indices belonging to this specific cluster in this layer
            # layer_specific_data['labels'] is a list where list[sample_idx] = original_cluster_idx_in_layer
            member_indices_for_this_cluster = [
                i for i, cl_idx in enumerate(layer_specific_data["labels"])
                if cl_idx == original_cluster_idx_in_layer
            ]

            if not member_indices_for_this_cluster:
                # print(f"Warning: No members found for cluster {unique_id_str} ({layer_name}C{original_cluster_idx_in_layer}). Labeling with generic ID.")
                cluster_profiles_for_labeling[unique_id_str] = "Cluster has no assigned samples."
                continue

            # Subset the main demographic dataframe
            try:
                # Ensure indices are within bounds of the demographic_data dataframe
                valid_member_indices = [idx for idx in member_indices_for_this_cluster if idx < len(self.processor.demographic_data)]
                if not valid_member_indices:
                    # print(f"Warning: No valid member indices for cluster {unique_id_str} in demographic data. Labeling with generic ID.")
                    cluster_profiles_for_labeling[unique_id_str] = "Cluster members not found in demographic data."
                    continue
                
                cluster_members_df = self.processor.demographic_data.iloc[valid_member_indices].copy()
            except IndexError as e:
                print(f"Error creating members_df for cluster {unique_id_str}: {e}. Max index: {max(member_indices_for_this_cluster)}, DF len: {len(self.processor.demographic_data)}")
                cluster_profiles_for_labeling[unique_id_str] = "Error retrieving member demographics."
                continue

            if cluster_members_df.empty:
                # print(f"Warning: Demographic data for cluster {unique_id_str} is empty. Labeling with generic ID.")
                cluster_profiles_for_labeling[unique_id_str] = "Cluster has no demographic data."
                continue

            # Generate a textual profile (simplified example)
            profile_parts = []
            dataset = self.processor.dataset_name
            
            if dataset == "titanic":
                if "Survived" in cluster_members_df.columns or "survived" in cluster_members_df.columns:
                    survived_col_name = "Survived" if "Survived" in cluster_members_df.columns else "survived"
                    rate = cluster_members_df[survived_col_name].mean()
                    profile_parts.append(f"Survival Rate: {rate:.1%}")
                if "Age" in cluster_members_df.columns or "age" in cluster_members_df.columns:
                    age_col_name = "Age" if "Age" in cluster_members_df.columns else "age"
                    if not cluster_members_df[age_col_name].dropna().empty:
                        profile_parts.append(f"Mean Age: {cluster_members_df[age_col_name].mean():.1f}")
                if "Sex" in cluster_members_df.columns or "sex" in cluster_members_df.columns:
                    sex_col_name = "Sex" if "Sex" in cluster_members_df.columns else "sex"
                    male_pct = (cluster_members_df[sex_col_name].str.lower() == "male").mean()
                    profile_parts.append(f"Sex: {male_pct:.1%} Male")
                if "Pclass" in cluster_members_df.columns or "pclass" in cluster_members_df.columns:
                    pclass_col_name = "Pclass" if "Pclass" in cluster_members_df.columns else "pclass"
                    # Ensure we use single quotes for keys if f-string uses double, or vice-versa for safety
                    # The .value_counts().to_dict() might produce integer keys that don't need quotes for f-string access if simple.
                    # However, a general string formatting for the dict items is safer.
                    pclass_dist = cluster_members_df[pclass_col_name].value_counts(normalize=True).sort_index()
                    pclass_str_parts = []
                    for k, v in pclass_dist.items():
                        pclass_str_parts.append(f"Class {k}: {v:.1%}")
                    profile_parts.append(f"Distribution: {', '.join(pclass_str_parts)}")
            
            elif dataset == "heart":
                if 'target' in cluster_members_df.columns: # Use single quotes for consistency
                    rate = cluster_members_df['target'].mean()
                    profile_parts.append(f"Disease Rate: {rate:.1%}")
                if 'age' in cluster_members_df.columns and not cluster_members_df['age'].dropna().empty:
                    profile_parts.append(f"Mean Age: {cluster_members_df['age'].mean():.1f}")
                if 'sex' in cluster_members_df.columns: # Assuming 1 for male, 0 for female
                    male_pct = cluster_members_df['sex'].mean()
                    profile_parts.append(f"Sex: {male_pct:.1%} Male")
                for col in ["cp", "trestbps", "chol"]:
                    if col in cluster_members_df.columns and not cluster_members_df[col].dropna().empty:
                        profile_parts.append(f"Mean {col.title()}: {cluster_members_df[col].mean():.1f}")
            
            profile_str = ", ".join(profile_parts) if profile_parts else "No specific demographic profile generated."
            cluster_profiles_for_labeling[unique_id_str] = profile_str

        if not cluster_profiles_for_labeling:
            print("No cluster profiles could be generated for labeling.")
            return {}

        print(f"Generated {len(cluster_profiles_for_labeling)} demographic profiles for cluster labeling.")
        if self.processor.dataset_name == "titanic": # Example print for one profile
            example_id = next(iter(cluster_profiles_for_labeling.keys()), None)
            if example_id:
                print(f"Example profile for cluster {example_id}: {cluster_profiles_for_labeling[example_id]}")

        # Instantiate ClusterAnalysis with appropriate provider/model for labeling
        # The __init__ in ClusterAnalysis now defaults to openai/gpt-4
        ca = ClusterAnalysis(use_cache=True, debug=True) # Use debug=True for labeling prompts
        
        # Call the updated label_clusters_sync with profiles
        generated_labels = ca.label_clusters_sync(cluster_profiles_for_labeling, max_concurrency=3)
        ca.close()
        
        print(f"Generated {len(generated_labels)} semantic labels.")
        return generated_labels


def main():
    parser = argparse.ArgumentParser(description="Generate LLM narratives for neural network paths")
    parser.add_argument("--data_dir", type=str, default="data/cluster_paths",
                      help="Directory containing cluster path data")
    parser.add_argument("--output_dir", type=str, default="results/llm",
                      help="Directory to save output files")
    parser.add_argument("--dataset", type=str, default="titanic",
                      choices=["titanic", "heart", "adult"],
                      help="Dataset to analyze")
    parser.add_argument("--seed", type=int, default=0,
                      help="Random seed for the experiment")
    parser.add_argument("--analysis_type", type=str, default="comprehensive",
                      choices=["technical", "fairness", "narrative", "comprehensive"],
                      help="Type of analysis to perform for per-path mode")
    parser.add_argument("--num_paths", type=int, default=3,
                      help="Number of top paths to analyze (per-path mode) or detail in prompt (holistic mode)")
    parser.add_argument("--all_paths", action="store_true",
                      help="Analyze every unique path instead of just the top-N (per-path mode only)")
    parser.add_argument("--paths", type=str, nargs="+",
                      help="Specific paths to analyze (per-path mode only, e.g., 'L0C0→L1C2→L2C0')")
    parser.add_argument("--outfile", type=str,
                      help="Explicit output JSON filename (default: {dataset}_seed_{seed}_analysis.json)")
    parser.add_argument("--analysis_mode", type=str, default="per_path", choices=["per_path", "holistic"],
                        help="Analysis mode: 'per_path' for individual path narratives, 'holistic' for a single system-wide analysis.")
    
    args = parser.parse_args()
    
    # Construct path to data file
    data_path = os.path.join(args.data_dir, f"{args.dataset}_seed_{args.seed}_paths.json")
    
    if not os.path.exists(data_path):
        # Try with _paths_with_centroids.json suffix
        data_path = os.path.join(args.data_dir, f"{args.dataset}_seed_{args.seed}_paths_with_centroids.json")
        
        if not os.path.exists(data_path):
            print(f"Error: Could not find path data file for {args.dataset} (seed {args.seed})")
            print(f"Tried: {os.path.join(args.data_dir, f'{args.dataset}_seed_{args.seed}_paths.json')}")
            print(f"Tried: {os.path.join(args.data_dir, f'{args.dataset}_seed_{args.seed}_paths_with_centroids.json')}")
            sys.exit(1)
    
    # Initialize processor and analyzer
    print(f"Loading path data from {data_path}")
    processor = PathDataProcessor(data_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = LLMPathAnalyzer(processor, args.output_dir)
    
    # Analyze paths based on mode
    if args.analysis_mode == "holistic":
        print(f"Performing HOLISTIC system analysis, detailing top {args.num_paths} paths in prompt.")
        holistic_results = analyzer.analyze_system_holistically(num_top_archetypes_in_prompt=args.num_paths)
        # Note: current save_results will save this under 'holistic_system_analysis' in the JSON.
        # The .md and .tex reports will still be from any per-path data if it existed or was run before.
    elif args.all_paths:
        print("Analyzing ALL unique paths (per-path mode)")
        results = analyzer.analyze_top_paths(None, args.analysis_type)
    elif args.paths:
        # Analyze specific paths
        print(f"Analyzing specific paths: {args.paths} (per-path mode)")
        results = analyzer.analyze_selected_paths(args.paths, args.analysis_type)
    else:
        # Analyze top paths (per-path mode)
        print(f"Analyzing top {args.num_paths} paths (per-path mode)")
        results = analyzer.analyze_top_paths(args.num_paths, args.analysis_type)
    
    # Save results
    output_path = analyzer.save_results(args.outfile)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()