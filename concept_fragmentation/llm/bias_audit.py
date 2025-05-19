"""
Bias audit module for cluster path analysis.

This module provides functions to analyze potential biases in cluster paths
by examining the relationship between demographic factors and clustering patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import Counter, defaultdict
import scipy.stats as stats


def compute_demographic_distribution(
    paths: Dict[int, List[str]],
    demographic_info: Dict[int, Dict[str, Any]],
    demographic_column: str
) -> Dict[str, Dict[str, float]]:
    """
    Compute distribution of demographic attributes across different paths.
    
    Args:
        paths: Dictionary mapping path IDs to lists of cluster IDs
        demographic_info: Dictionary mapping path IDs to demographic information
        demographic_column: The demographic attribute to analyze
        
    Returns:
        Dictionary mapping path IDs to demographic distributions
    """
    distributions = {}
    
    for path_id, path in paths.items():
        if path_id in demographic_info and demographic_column in demographic_info[path_id]:
            # Extract the demographic distribution for this path
            demo_data = demographic_info[path_id][demographic_column]
            
            # Convert to consistent format for analysis
            if isinstance(demo_data, dict):
                distributions[str(path_id)] = demo_data
            else:
                # If it's a scalar value, create single-category distribution
                distributions[str(path_id)] = {str(demo_data): 1.0}
    
    return distributions


def compute_bias_scores(
    paths: Dict[int, List[str]],
    demographic_info: Dict[int, Dict[str, Any]],
    demographic_columns: List[str],
    baseline_distribution: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute bias scores for each demographic attribute across paths.
    
    Args:
        paths: Dictionary mapping path IDs to lists of cluster IDs
        demographic_info: Dictionary mapping path IDs to demographic information
        demographic_columns: List of demographic attributes to analyze
        baseline_distribution: Optional baseline distribution for comparison
        
    Returns:
        Dictionary mapping demographic attributes to bias scores per path
    """
    bias_scores = {}
    
    for demo_col in demographic_columns:
        # Get distributions across paths
        distributions = compute_demographic_distribution(paths, demographic_info, demo_col)
        
        # Compute average distribution if no baseline provided
        if baseline_distribution is None or demo_col not in baseline_distribution:
            # Create an aggregate distribution for this demographic
            aggregated = defaultdict(float)
            counts = defaultdict(int)
            
            # Sum up values across all paths
            for path_id, dist in distributions.items():
                for category, value in dist.items():
                    aggregated[category] += value
                    counts[category] += 1
            
            # Compute averages
            baseline = {cat: agg/counts[cat] for cat, agg in aggregated.items() if counts[cat] > 0}
        else:
            baseline = baseline_distribution[demo_col]
        
        # Compute deviation from baseline for each path
        path_scores = {}
        for path_id, dist in distributions.items():
            # Measure distance between distributions
            # Using Jensen-Shannon divergence (symmetrized KL divergence)
            
            # Ensure all categories are in both distributions
            all_categories = set(dist.keys()) | set(baseline.keys())
            
            # Convert to probability vectors with same categories
            p = np.array([dist.get(cat, 0.0) for cat in all_categories])
            q = np.array([baseline.get(cat, 0.0) for cat in all_categories])
            
            # Normalize if needed
            if p.sum() > 0:
                p = p / p.sum()
            if q.sum() > 0:
                q = q / q.sum()
            
            # Skip if either distribution is empty
            if p.sum() == 0 or q.sum() == 0:
                path_scores[path_id] = 0.0
                continue
            
            # Calculate Jensen-Shannon divergence
            m = 0.5 * (p + q)
            
            # Handle zeros in distributions to avoid log(0)
            eps = 1e-10
            p_adjusted = np.maximum(p, eps)
            q_adjusted = np.maximum(q, eps)
            m_adjusted = np.maximum(m, eps)
            
            js_divergence = 0.5 * (
                np.sum(p_adjusted * np.log(p_adjusted / m_adjusted)) +
                np.sum(q_adjusted * np.log(q_adjusted / m_adjusted))
            )
            
            # Store the divergence as bias score
            path_scores[path_id] = float(js_divergence)
        
        bias_scores[demo_col] = path_scores
    
    return bias_scores


def generate_bias_report(
    paths: Dict[int, List[str]],
    demographic_info: Dict[int, Dict[str, Any]],
    demographic_columns: List[str],
    cluster_labels: Dict[str, str],
    population_stats: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive bias audit report for cluster paths.
    
    Args:
        paths: Dictionary mapping path IDs to lists of cluster IDs
        demographic_info: Dictionary mapping path IDs to demographic information
        demographic_columns: List of demographic attributes to analyze
        cluster_labels: Dictionary mapping cluster IDs to human-readable labels
        population_stats: Optional overall population statistics for comparison
        
    Returns:
        Dictionary containing bias report sections
    """
    report = {
        "summary": {},
        "demographic_distributions": {},
        "bias_scores": {},
        "high_bias_paths": {},
        "low_bias_paths": {},
        "recommendations": []
    }
    
    # Compute bias scores
    bias_scores = compute_bias_scores(paths, demographic_info, demographic_columns, population_stats)
    report["bias_scores"] = bias_scores
    
    # Get distributions for all demographic columns
    for demo_col in demographic_columns:
        report["demographic_distributions"][demo_col] = compute_demographic_distribution(
            paths, demographic_info, demo_col
        )
    
    # Find paths with highest/lowest bias for each demographic
    for demo_col, scores in bias_scores.items():
        # Sort paths by bias score
        sorted_paths = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Extract high bias paths (top 3)
        high_bias = []
        for path_id, score in sorted_paths[:3]:
            if int(path_id) in paths:
                path_clusters = paths[int(path_id)]
                path_labels = [cluster_labels.get(cid, cid) for cid in path_clusters]
                
                high_bias.append({
                    "path_id": path_id,
                    "bias_score": score,
                    "path": "→".join(path_clusters),
                    "path_with_labels": "→".join([f"{cid} ({label})" for cid, label in zip(path_clusters, path_labels)]),
                    "demographic_distribution": report["demographic_distributions"][demo_col].get(path_id, {})
                })
        
        # Extract low bias paths (bottom 3)
        low_bias = []
        for path_id, score in sorted_paths[-3:]:
            if int(path_id) in paths:
                path_clusters = paths[int(path_id)]
                path_labels = [cluster_labels.get(cid, cid) for cid in path_clusters]
                
                low_bias.append({
                    "path_id": path_id,
                    "bias_score": score,
                    "path": "→".join(path_clusters),
                    "path_with_labels": "→".join([f"{cid} ({label})" for cid, label in zip(path_clusters, path_labels)]),
                    "demographic_distribution": report["demographic_distributions"][demo_col].get(path_id, {})
                })
        
        report["high_bias_paths"][demo_col] = high_bias
        report["low_bias_paths"][demo_col] = low_bias
    
    # Generate overall summary
    avg_bias_per_demographic = {}
    max_bias_per_demographic = {}
    
    for demo_col, scores in bias_scores.items():
        if scores:
            score_values = list(scores.values())
            avg_bias_per_demographic[demo_col] = sum(score_values) / len(score_values)
            max_bias_per_demographic[demo_col] = max(score_values)
    
    report["summary"] = {
        "average_bias_scores": avg_bias_per_demographic,
        "maximum_bias_scores": max_bias_per_demographic,
        "total_paths_analyzed": len(paths),
        "demographics_analyzed": demographic_columns
    }
    
    # Generate recommendations
    recommendations = []
    
    # Find the demographic with highest average bias
    if avg_bias_per_demographic:
        highest_bias_demo = max(avg_bias_per_demographic.items(), key=lambda x: x[1])
        recommendations.append(
            f"The demographic factor '{highest_bias_demo[0]}' shows the highest average bias "
            f"({highest_bias_demo[1]:.4f}). Consider examining how this attribute influences clustering patterns."
        )
    
    # Suggest investigation of high-bias paths
    for demo_col, high_paths in report["high_bias_paths"].items():
        if high_paths:
            top_path = high_paths[0]
            recommendations.append(
                f"Path {top_path['path_id']} shows significant bias for {demo_col} "
                f"(score: {top_path['bias_score']:.4f}). Investigate this path's clusters for potential bias."
            )
    
    report["recommendations"] = recommendations
    
    return report


def generate_llm_bias_prompt(report: Dict[str, Any]) -> str:
    """
    Generate a prompt for LLM bias analysis based on the bias report.
    
    Args:
        report: Bias audit report dictionary
        
    Returns:
        A formatted prompt string for LLM analysis
    """
    prompt = """You are an AI expert analyzing potential biases in neural network clusters and paths.
    
Below is a bias audit report for cluster paths, showing how demographic factors relate to neural network decision patterns.
Please analyze this report and provide insights on potential biases, their implications, and recommendations for mitigation.

"""
    
    # Add summary section
    summary = report["summary"]
    prompt += "## Bias Audit Summary\n"
    prompt += f"- Total paths analyzed: {summary['total_paths_analyzed']}\n"
    prompt += f"- Demographics analyzed: {', '.join(summary['demographics_analyzed'])}\n"
    
    # Add average bias scores
    prompt += "\n### Average Bias Scores by Demographic\n"
    for demo, score in summary["average_bias_scores"].items():
        prompt += f"- {demo}: {score:.4f}\n"
    
    # Add paths with highest bias
    prompt += "\n## High Bias Paths\n"
    for demo_col, high_paths in report["high_bias_paths"].items():
        if high_paths:
            prompt += f"\n### Highest bias paths for '{demo_col}':\n"
            for path_info in high_paths:
                prompt += f"- Path {path_info['path_id']} (bias score: {path_info['bias_score']:.4f})\n"
                prompt += f"  Path with labels: {path_info['path_with_labels']}\n"
                prompt += f"  Demographic distribution: {path_info['demographic_distribution']}\n"
    
    # Add recommendations
    prompt += "\n## Initial Recommendations\n"
    for rec in report["recommendations"]:
        prompt += f"- {rec}\n"
    
    prompt += """
Based on this report, please provide:
1. An assessment of potential bias patterns in the clustering process
2. Implications of these biases for model fairness and interpretability
3. Specific recommendations for mitigating identified biases
4. Suggestions for further analysis to better understand bias patterns

Your analysis should be concise, actionable, and focused on practical implications.
"""
    
    return prompt


def analyze_bias_with_llm(
    analyzer,
    bias_report: Dict[str, Any]
) -> str:
    """
    Use an LLM to analyze a bias report.
    
    Args:
        analyzer: ClusterAnalysis instance with an LLM client
        bias_report: The generated bias audit report
        
    Returns:
        LLM-generated bias analysis
    """
    # Generate the prompt
    prompt = generate_llm_bias_prompt(bias_report)
    
    # Get LLM response
    response = analyzer.generate_with_cache(
        prompt=prompt,
        temperature=0.4,
        max_tokens=500,
        system_prompt="You are an AI assistant specializing in fairness and bias analysis in machine learning systems."
    )
    
    return response.text.strip()