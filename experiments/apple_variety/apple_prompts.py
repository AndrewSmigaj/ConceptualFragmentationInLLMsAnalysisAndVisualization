"""
Apple quality classification specific prompts for LLM analysis.

This module provides specialized prompts for analyzing apple quality
neural network internals including quality routing patterns, variety
trajectories, and economic impact analysis. These prompts are designed 
to extract meaningful insights about apple classification decisions.
"""

import textwrap
from typing import Dict, List, Any, Optional, Union

# Base prompt for apple cluster labeling
APPLE_CLUSTER_LABELING_PROMPT = textwrap.dedent("""
    You are an expert in analyzing neural network activations for agricultural quality classification,
    particularly for apple sorting and grading systems.
    
    I will provide you with:
    1. The name of a layer in a neural network trained to classify apple quality
    2. Statistical information about a cluster of apple samples at that layer
    
    Your task is to provide a concise, meaningful label for this cluster that captures
    what quality characteristics these apples might share.
    
    Layer: {layer_name}
    Cluster: {cluster_id}
    Number of samples: {n_samples}
    
    Apple quality features in this cluster (averages):
    {feature_statistics}
    
    Think carefully about the role this layer plays in quality classification:
    - Early layers (L1) typically capture basic quality indicators
    - Middle layers (L2) often combine features into quality patterns  
    - Later layers (L3) tend to consolidate into routing decisions
    - Output layer represents final quality classification
    
    The patterns in these activations may represent:
    - Quality grades (premium, standard, juice)
    - Physical characteristics (size, color, firmness)
    - Defect patterns (bruising, scarring, decay)
    - Maturity indicators (sugar content, starch levels)
    
    Based on this information, provide a concise label (3-5 words) that best describes
    this cluster of apples, capturing their shared quality characteristics.
""")

# Prompt for apple variety routing narrative
APPLE_PATH_NARRATIVE_PROMPT = textwrap.dedent("""
    You are an expert in analyzing apple quality classification systems and understanding
    how neural networks route different apple varieties through quality grades.
    
    I will provide you with:
    1. A description of a path that apples take through quality assessment stages
    2. The labels for each cluster in the path
    3. Information about which apple varieties follow this path
    4. Quality classification outcomes
    
    Your task is to create a coherent narrative explaining this routing path,
    focusing on why certain varieties might be processed this way and the economic implications.
    
    Path: {path_description}
    
    Cluster labels along the path:
    {cluster_labels}
    
    Path statistics:
    - Frequency: {path_frequency:.2f}% of apples follow this path
    - Primary varieties: {variety_examples}
    - Quality outcome distribution: {quality_distribution}
    
    Economic information:
    - Average price for apples on this path: ${avg_price:.2f}/lb
    - Potential value if routed optimally: ${optimal_price:.2f}/lb
    
    Consider the following in your narrative:
    - Why these varieties might share this processing path
    - What quality characteristics drive the routing decisions
    - Economic impact of any misclassification
    - Whether this represents optimal or suboptimal routing
    
    Please provide a concise (3-5 sentences) narrative explaining this quality routing path
    and its business implications.
""")

# Prompt for comprehensive apple quality analysis
APPLE_COMPREHENSIVE_ANALYSIS_PROMPT = textwrap.dedent("""
    You are an expert in agricultural AI systems, specializing in apple quality classification
    and the economic impact of sorting decisions. Analyze the following neural network
    clustering data to understand how the model classifies apple quality.
    
    === APPLE QUALITY CLUSTERS ===
    {cluster_profiles}
    
    === VARIETY ROUTING PATHS ===
    {routing_paths}
    
    === QUALITY CLASSIFICATION DISTRIBUTION ===
    {quality_distribution}
    
    === ANALYSIS REQUIRED ===
    Please provide:
    
    1. CLUSTER LABELS:
    For each cluster, provide a concise label (3-5 words) that captures the dominant
    quality characteristics. Format as "Cluster_ID: Label"
    
    2. QUALITY ROUTING ANALYSIS:
    - Identify which varieties are consistently routed to premium grades
    - Identify which varieties are often misrouted to lower grades
    - Explain the quality features that drive these routing decisions
    - Calculate the economic impact of misrouting premium varieties
    
    3. BIAS DETECTION:
    - Identify any systematic biases in quality classification
    - Note if certain varieties are unfairly downgraded
    - Identify any size, color, or origin biases
    
    4. BUSINESS RECOMMENDATIONS:
    - Suggest improvements to reduce economically costly misclassifications
    - Identify varieties that need special handling
    - Recommend quality features that need better calibration
    
    CLUSTER LABELS:
    
    ANALYSIS:
""")

# Prompt for apple quality bias detection
APPLE_BIAS_DETECTION_PROMPT = textwrap.dedent("""
    You are an expert in detecting bias in agricultural AI systems, particularly
    in apple quality classification systems that can have significant economic impact.
    
    I will provide you with:
    1. Statistical information about how different apple varieties are classified
    2. Quality routing patterns and their frequencies
    3. Economic impact data
    
    Your task is to identify and explain any biases in the quality classification system.
    
    === VARIETY CLASSIFICATION PATTERNS ===
    {variety_patterns}
    
    === MISCLASSIFICATION ANALYSIS ===
    {misclassification_data}
    
    === ECONOMIC IMPACT ===
    Average loss per pound by variety:
    {economic_losses}
    
    Consider the following types of bias:
    - Variety bias: Are certain varieties systematically downgraded?
    - Size bias: Do smaller/larger apples get unfairly classified?
    - Color bias: Are redder/greener apples treated differently?
    - Seasonality bias: Do early/late season apples face discrimination?
    - Origin bias: Are apples from certain orchards rated differently?
    
    Please provide:
    1. A summary of detected biases (2-3 sentences per bias type)
    2. Economic impact of each bias
    3. Recommendations for bias mitigation
    4. Priority ranking of biases to address
""")

# Template function for generating apple cluster labeling prompts
def generate_cluster_labeling_prompt(
    layer_name: str,
    cluster_id: str,
    n_samples: int,
    feature_statistics: Dict[str, float]
) -> str:
    """
    Generate a prompt for labeling an apple quality cluster.
    
    Args:
        layer_name: Name of the layer
        cluster_id: ID of the cluster
        n_samples: Number of samples in cluster
        feature_statistics: Statistics about apple quality features
        
    Returns:
        Formatted prompt string
    """
    # Format feature statistics as a bulleted list
    stats_formatted = "\n".join([
        f"- {feature}: {value:.2f}" for feature, value in feature_statistics.items()
    ])
    
    return APPLE_CLUSTER_LABELING_PROMPT.format(
        layer_name=layer_name,
        cluster_id=cluster_id,
        n_samples=n_samples,
        feature_statistics=stats_formatted
    )

# Template function for generating apple path narrative prompts
def generate_path_narrative_prompt(
    path_description: str,
    cluster_labels: Dict[str, str],
    path_frequency: float,
    variety_examples: List[str],
    quality_distribution: Dict[str, float],
    avg_price: float,
    optimal_price: float
) -> str:
    """
    Generate a prompt for creating a narrative about apple routing paths.
    
    Args:
        path_description: Description of the path
        cluster_labels: Labels for clusters in the path
        path_frequency: Frequency of this path
        variety_examples: Examples of varieties that follow this path
        quality_distribution: Distribution of quality outcomes
        avg_price: Average price for apples on this path
        optimal_price: Potential price if routed optimally
        
    Returns:
        Formatted prompt string
    """
    # Format cluster labels as a bulleted list
    cluster_labels_formatted = "\n".join(
        [f"- {stage}: {label}" for stage, label in cluster_labels.items()]
    )
    
    # Format variety examples
    variety_examples_formatted = ", ".join(variety_examples[:3])
    
    # Format quality distribution
    quality_dist_formatted = ", ".join(
        [f"{quality}: {pct:.1f}%" for quality, pct in quality_distribution.items()]
    )
    
    return APPLE_PATH_NARRATIVE_PROMPT.format(
        path_description=path_description,
        cluster_labels=cluster_labels_formatted,
        path_frequency=path_frequency * 100,  # Convert to percentage
        variety_examples=variety_examples_formatted,
        quality_distribution=quality_dist_formatted,
        avg_price=avg_price,
        optimal_price=optimal_price
    )

# Template function for generating comprehensive analysis prompts
def generate_comprehensive_analysis_prompt(
    cluster_profiles: str,
    routing_paths: str,
    quality_distribution: str
) -> str:
    """
    Generate a prompt for comprehensive apple quality analysis.
    
    Args:
        cluster_profiles: Formatted cluster profile information
        routing_paths: Formatted routing path information
        quality_distribution: Formatted quality distribution data
        
    Returns:
        Formatted prompt string
    """
    return APPLE_COMPREHENSIVE_ANALYSIS_PROMPT.format(
        cluster_profiles=cluster_profiles,
        routing_paths=routing_paths,
        quality_distribution=quality_distribution
    )

# Template function for generating bias detection prompts
def generate_bias_detection_prompt(
    variety_patterns: str,
    misclassification_data: str,
    economic_losses: Dict[str, float]
) -> str:
    """
    Generate a prompt for detecting bias in apple quality classification.
    
    Args:
        variety_patterns: Formatted variety classification patterns
        misclassification_data: Formatted misclassification analysis
        economic_losses: Economic losses by variety
        
    Returns:
        Formatted prompt string
    """
    # Format economic losses as a bulleted list
    losses_formatted = "\n".join(
        [f"- {variety}: ${loss:.2f}/lb" for variety, loss in economic_losses.items()]
    )
    
    return APPLE_BIAS_DETECTION_PROMPT.format(
        variety_patterns=variety_patterns,
        misclassification_data=misclassification_data,
        economic_losses=losses_formatted
    )