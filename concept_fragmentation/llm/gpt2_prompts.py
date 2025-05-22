"""
GPT-2 specific prompts for LLM analysis.

This module provides specialized prompts for analyzing GPT-2 model
internals including attention patterns, token representations,
and cluster behaviors. These prompts are designed to extract
meaningful insights from transformer-specific metrics.
"""

import textwrap
from typing import Dict, List, Any, Optional, Union

# Base prompt for GPT-2 cluster labeling
GPT2_CLUSTER_LABELING_PROMPT = textwrap.dedent("""
    You are an expert in analyzing and interpreting neural network activations,
    particularly in transformer language models like GPT-2. 
    
    I will provide you with:
    1. The name of a layer in a GPT-2 model
    2. The centroid (average activation) of a cluster of token activations at that layer
    
    Your task is to provide a concise, meaningful label for this cluster that captures
    what concept these tokens might represent.
    
    Layer: {layer_name}
    Cluster: {cluster_id}
    Centroid dimensions: {centroid_shape}
    
    Here are some sample tokens that belong to this cluster:
    {token_examples}
    
    Think carefully about the role this layer plays in GPT-2:
    - Lower layers (0-3) typically capture basic token properties and syntax
    - Middle layers (4-7) often represent semantic and grammatical patterns
    - Upper layers (8-11) tend to focus on higher-level language and contextual meaning
    
    The patterns in these activations may represent:
    - Syntactic categories (nouns, verbs, adjectives, etc.)
    - Semantic groupings (entities, actions, properties, etc.)
    - Contextual roles (subject, object, modifier, etc.)
    - Attention patterns or information flow
    
    Based on this information, provide a concise label (2-4 words) that best describes 
    this cluster of tokens, capturing their shared characteristics or function.
""")

# Prompt for GPT-2 cluster label with attention pattern context
GPT2_ATTENTION_AWARE_CLUSTER_PROMPT = textwrap.dedent("""
    You are an expert in analyzing neural network activations in transformer language models
    like GPT-2, with particular expertise in understanding attention mechanisms.
    
    I will provide you with:
    1. The name of a layer in a GPT-2 model
    2. The centroid (average activation) of a cluster of token activations at that layer
    3. Attention pattern information for tokens in this cluster
    
    Your task is to provide a concise, meaningful label for this cluster that captures
    what concept these tokens might represent, taking into account how attention flows
    to and from these tokens.
    
    Layer: {layer_name}
    Cluster: {cluster_id}
    Centroid dimensions: {centroid_shape}
    
    Sample tokens in this cluster:
    {token_examples}
    
    Attention pattern information:
    - Attention received: {attention_received:.4f} (higher means tokens in this cluster are attended to more)
    - Attention given: {attention_given:.4f} (higher means tokens in this cluster attend to others more)
    - Self-attention: {self_attention:.4f} (higher means tokens attend to themselves more)
    - Primary attention targets: {attention_targets}
    
    Think about how attention patterns relate to token function:
    - Tokens that receive high attention often carry important semantic meaning
    - Tokens that give high attention often need context to be interpreted
    - Tokens with high self-attention often have stand-alone meaning
    
    Based on this information, provide a concise label (2-4 words) that best describes 
    this cluster of tokens, capturing their shared characteristics or function.
""")

# Prompt for generating path narratives in GPT-2
GPT2_PATH_NARRATIVE_PROMPT = textwrap.dedent("""
    You are an expert in analyzing and interpreting the internal representations of
    transformer language models like GPT-2. You specialize in explaining how token
    representations evolve through the layers of the network.
    
    I will provide you with:
    1. A description of a path that tokens take through clusters in different layers
    2. The labels for each cluster in the path
    3. Information about how common or rare this path is
    
    Your task is to create a coherent narrative explaining this path through the network,
    focusing on how the token's representation evolves and what it might mean for the model's
    processing of language.
    
    Path: {path_description}
    
    Cluster labels along the path:
    {cluster_labels}
    
    Path statistics:
    - Frequency: {path_frequency:.2f}% of tokens follow this path
    - Token examples: {token_examples}
    
    Fragmentation information:
    - Fragmentation score: {fragmentation_score:.4f} (higher means tokens with same text follow many different paths)
    
    Consider the following in your narrative:
    - How the representation evolves from lower to higher layers
    - What language features or concepts might be captured at each stage
    - Why the model might process these tokens along this specific path
    - Whether this path represents typical or atypical processing
    
    Please provide a concise (3-5 sentences) narrative explaining this path through the network.
""")

# Prompt for attention pattern narrative generation
GPT2_ATTENTION_PATTERN_PROMPT = textwrap.dedent("""
    You are an expert in analyzing and interpreting attention patterns in transformer
    language models like GPT-2. You specialize in explaining what attention patterns reveal
    about how the model processes language.
    
    I will provide you with:
    1. Attention pattern statistics for a specific layer
    2. Information about which tokens attend to which other tokens
    3. Details about attention head behaviors
    
    Your task is to create a coherent narrative explaining what these attention patterns
    suggest about how the model is processing the input text at this layer.
    
    Layer: {layer_name}
    
    Attention pattern statistics:
    - Average attention entropy: {attention_entropy:.4f} (higher means more diffuse attention)
    - Head agreement score: {head_agreement:.4f} (higher means heads focus on similar tokens)
    - Number of attention heads: {num_heads}
    
    Notable attention patterns:
    {attention_patterns}
    
    Token-to-token attention examples:
    {token_attention_examples}
    
    Consider the following in your narrative:
    - What linguistic phenomena these attention patterns might correspond to
    - How attention is distributed (focused vs. diffuse)
    - Whether different heads specialize in different patterns
    - How this layer's attention relates to its position in the model
    
    Please provide a concise (3-5 sentences) narrative explaining these attention patterns.
""")

# Prompt for token movement narrative generation
GPT2_TOKEN_MOVEMENT_PROMPT = textwrap.dedent("""
    You are an expert in analyzing and interpreting how token representations move through
    the embedding space in transformer language models like GPT-2. You specialize in
    explaining what token movements reveal about the model's processing.
    
    I will provide you with:
    1. Statistics about how a token's representation changes across layers
    2. Information about which clusters the token visits
    3. Metrics about the token's movement pattern
    
    Your task is to create a coherent narrative explaining what this movement pattern
    suggests about how the model is processing this token.
    
    Token: {token_text} (position {token_position})
    
    Path through clusters:
    {cluster_path}
    
    Movement metrics:
    - Path length: {path_length:.4f} (higher means greater movement through embedding space)
    - Cluster changes: {cluster_changes} (number of times the token changes clusters)
    - Mobility score: {mobility_score:.4f} (overall measure of how much the token moves)
    
    Comparative statistics:
    - Average token path length: {avg_path_length:.4f}
    - Average cluster changes: {avg_cluster_changes:.2f}
    - Mobility ranking: {mobility_ranking} out of {total_tokens} tokens
    
    Consider the following in your narrative:
    - Whether this token's movement is typical or unusual
    - What the pattern of movement suggests about how the model processes this token
    - How the sequence of clusters relates to the token's linguistic properties
    - What might be driving changes or stability in the token's representation
    
    Please provide a concise (3-5 sentences) narrative explaining this token's movement pattern.
""")

# Prompt for concept purity analysis
GPT2_CONCEPT_PURITY_PROMPT = textwrap.dedent("""
    You are an expert in analyzing and interpreting the conceptual organization in
    transformer language models like GPT-2. You specialize in explaining what cluster
    structure reveals about how the model organizes information.
    
    I will provide you with:
    1. Metrics about cluster purity and separation at different layers
    2. Information about how stable the clustering is
    3. Comparisons between layers
    
    Your task is to create a coherent narrative explaining what these metrics reveal about
    how the model organizes concepts at different stages of processing.
    
    Layer metrics:
    {layer_metrics}
    
    Stability information:
    - Average silhouette score: {avg_silhouette:.4f} (higher means better defined clusters)
    - Cluster stability: {cluster_stability:.4f} (higher means more consistent clustering)
    
    Layer comparison:
    - Layer with highest concept purity: {best_layer} (purity: {best_purity:.4f})
    - Layer with lowest concept purity: {worst_layer} (purity: {worst_purity:.4f})
    
    Consider the following in your narrative:
    - What the cluster quality metrics suggest about how the model organizes information
    - How concept organization changes from lower to higher layers
    - Whether clusters become more or less distinct at different stages
    - What this organization might reveal about the model's language processing
    
    Please provide a concise (3-5 sentences) narrative explaining the model's conceptual organization.
""")

# Prompt for token path and attention correlation analysis
GPT2_PATH_ATTENTION_CORRELATION_PROMPT = textwrap.dedent("""
    You are an expert in analyzing and interpreting the relationship between token paths
    and attention patterns in transformer language models like GPT-2. You specialize in
    explaining how attention mechanisms influence token representations.
    
    I will provide you with:
    1. Correlation metrics between token paths and attention patterns
    2. Layer-specific correlation information
    3. Examples of tokens with strong or weak correlation
    
    Your task is to create a coherent narrative explaining what these correlations reveal
    about how attention mechanisms guide token representations through the network.
    
    Overall correlation metrics:
    - Average path-attention correlation: {avg_correlation:.4f} (higher means stronger relationship)
    - Attention follows paths score: {attn_follows_paths:.4f} (higher means attention predicts paths)
    - Paths follow attention score: {paths_follow_attn:.4f} (higher means paths follow attention)
    
    Layer transition correlations:
    {layer_correlations}
    
    Token examples:
    - Strongest correlation: {strong_examples}
    - Weakest correlation: {weak_examples}
    
    Consider the following in your narrative:
    - What the correlation patterns suggest about how attention guides token representations
    - Whether attention is more predictive of paths in certain layers
    - What types of tokens show strongest correlation between attention and paths
    - How these correlations might reveal mechanism of information flow in the model
    
    Please provide a concise (3-5 sentences) narrative explaining the relationship between
    attention patterns and token paths in this model.
""")

# Template function for generating GPT-2 cluster labeling prompts
def generate_cluster_labeling_prompt(
    layer_name: str,
    cluster_id: int,
    centroid: Any,
    token_examples: List[str],
    attention_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a prompt for labeling a GPT-2 cluster.
    
    Args:
        layer_name: Name of the layer
        cluster_id: ID of the cluster
        centroid: Centroid of the cluster
        token_examples: Examples of tokens in the cluster
        attention_info: Optional attention pattern information
        
    Returns:
        Formatted prompt string
    """
    # Format token examples as a bulleted list
    token_examples_formatted = "\n".join([f"- {token}" for token in token_examples[:10]])
    
    # Get centroid shape
    centroid_shape = getattr(centroid, "shape", "unknown")
    
    # Use attention-aware prompt if attention info is provided
    if attention_info:
        return GPT2_ATTENTION_AWARE_CLUSTER_PROMPT.format(
            layer_name=layer_name,
            cluster_id=cluster_id,
            centroid_shape=centroid_shape,
            token_examples=token_examples_formatted,
            attention_received=attention_info.get("received", 0.0),
            attention_given=attention_info.get("given", 0.0),
            self_attention=attention_info.get("self", 0.0),
            attention_targets=attention_info.get("targets", "Unknown")
        )
    else:
        return GPT2_CLUSTER_LABELING_PROMPT.format(
            layer_name=layer_name,
            cluster_id=cluster_id,
            centroid_shape=centroid_shape,
            token_examples=token_examples_formatted
        )

# Template function for generating GPT-2 path narrative prompts
def generate_path_narrative_prompt(
    path_description: str,
    cluster_labels: Dict[str, str],
    path_frequency: float,
    token_examples: List[str],
    fragmentation_score: float
) -> str:
    """
    Generate a prompt for creating a narrative about a token path in GPT-2.
    
    Args:
        path_description: Description of the path
        cluster_labels: Labels for clusters in the path
        path_frequency: Frequency of this path
        token_examples: Examples of tokens that follow this path
        fragmentation_score: Path fragmentation score
        
    Returns:
        Formatted prompt string
    """
    # Format cluster labels as a bulleted list
    cluster_labels_formatted = "\n".join(
        [f"- Layer {layer}: {label}" for layer, label in cluster_labels.items()]
    )
    
    # Format token examples as a comma-separated list
    token_examples_formatted = ", ".join(token_examples[:5])
    
    return GPT2_PATH_NARRATIVE_PROMPT.format(
        path_description=path_description,
        cluster_labels=cluster_labels_formatted,
        path_frequency=path_frequency * 100,  # Convert to percentage
        token_examples=token_examples_formatted,
        fragmentation_score=fragmentation_score
    )

# Template function for generating GPT-2 attention pattern prompts
def generate_attention_pattern_prompt(
    layer_name: str,
    attention_entropy: float,
    head_agreement: float,
    num_heads: int,
    attention_patterns: List[str],
    token_attention_examples: List[str]
) -> str:
    """
    Generate a prompt for creating a narrative about attention patterns in GPT-2.
    
    Args:
        layer_name: Name of the layer
        attention_entropy: Entropy of attention distributions
        head_agreement: Agreement score between attention heads
        num_heads: Number of attention heads
        attention_patterns: Descriptions of notable attention patterns
        token_attention_examples: Examples of token-to-token attention
        
    Returns:
        Formatted prompt string
    """
    # Format attention patterns as a bulleted list
    patterns_formatted = "\n".join([f"- {pattern}" for pattern in attention_patterns])
    
    # Format token attention examples as a bulleted list
    examples_formatted = "\n".join([f"- {example}" for example in token_attention_examples])
    
    return GPT2_ATTENTION_PATTERN_PROMPT.format(
        layer_name=layer_name,
        attention_entropy=attention_entropy,
        head_agreement=head_agreement,
        num_heads=num_heads,
        attention_patterns=patterns_formatted,
        token_attention_examples=examples_formatted
    )

# Template function for generating GPT-2 token movement prompts
def generate_token_movement_prompt(
    token_text: str,
    token_position: int,
    cluster_path: List[Tuple[str, int]],
    path_length: float,
    cluster_changes: int,
    mobility_score: float,
    avg_path_length: float,
    avg_cluster_changes: float,
    mobility_ranking: int,
    total_tokens: int
) -> str:
    """
    Generate a prompt for creating a narrative about token movement in GPT-2.
    
    Args:
        token_text: Text of the token
        token_position: Position of the token in the sequence
        cluster_path: Path of the token through clusters
        path_length: Length of the token's path through embedding space
        cluster_changes: Number of times the token changes clusters
        mobility_score: Overall mobility score for the token
        avg_path_length: Average path length across all tokens
        avg_cluster_changes: Average number of cluster changes across all tokens
        mobility_ranking: Rank of the token's mobility (1 = most mobile)
        total_tokens: Total number of tokens analyzed
        
    Returns:
        Formatted prompt string
    """
    # Format cluster path as a bulleted list
    cluster_path_formatted = "\n".join(
        [f"- Layer {layer}: Cluster {cluster_id}" for layer, cluster_id in cluster_path]
    )
    
    return GPT2_TOKEN_MOVEMENT_PROMPT.format(
        token_text=token_text,
        token_position=token_position,
        cluster_path=cluster_path_formatted,
        path_length=path_length,
        cluster_changes=cluster_changes,
        mobility_score=mobility_score,
        avg_path_length=avg_path_length,
        avg_cluster_changes=avg_cluster_changes,
        mobility_ranking=mobility_ranking,
        total_tokens=total_tokens
    )

# Template function for generating GPT-2 concept purity prompts
def generate_concept_purity_prompt(
    layer_metrics: Dict[str, Dict[str, float]],
    avg_silhouette: float,
    cluster_stability: float,
    best_layer: str,
    best_purity: float,
    worst_layer: str,
    worst_purity: float
) -> str:
    """
    Generate a prompt for creating a narrative about concept purity in GPT-2.
    
    Args:
        layer_metrics: Metrics for each layer
        avg_silhouette: Average silhouette score across layers
        cluster_stability: Stability of clustering across layers
        best_layer: Layer with highest concept purity
        best_purity: Purity score of the best layer
        worst_layer: Layer with lowest concept purity
        worst_purity: Purity score of the worst layer
        
    Returns:
        Formatted prompt string
    """
    # Format layer metrics as a bulleted list
    layer_metrics_formatted = "\n".join(
        [f"- Layer {layer}: Purity = {metrics.get('purity', 0.0):.4f}, Silhouette = {metrics.get('silhouette', 0.0):.4f}"
         for layer, metrics in layer_metrics.items()]
    )
    
    return GPT2_CONCEPT_PURITY_PROMPT.format(
        layer_metrics=layer_metrics_formatted,
        avg_silhouette=avg_silhouette,
        cluster_stability=cluster_stability,
        best_layer=best_layer,
        best_purity=best_purity,
        worst_layer=worst_layer,
        worst_purity=worst_purity
    )


# Template function for generating GPT-2 path-attention correlation prompts
def generate_path_attention_correlation_prompt(
    correlation_metrics: Dict[str, Any],
    strong_examples: List[Tuple[str, float]],
    weak_examples: List[Tuple[str, float]]
) -> str:
    """
    Generate a prompt for creating a narrative about path-attention correlation in GPT-2.
    
    Args:
        correlation_metrics: Metrics about correlation between paths and attention
        strong_examples: Examples of tokens with strong correlation
        weak_examples: Examples of tokens with weak correlation
        
    Returns:
        Formatted prompt string
    """
    # Extract global metrics
    global_metrics = correlation_metrics.get("global_metrics", {})
    avg_correlation = global_metrics.get("avg_path_attention_correlation", 0.0)
    attn_follows_paths = global_metrics.get("attention_follows_paths", 0.0)
    paths_follow_attn = global_metrics.get("paths_follow_attention", 0.0)
    
    # Format layer correlation information
    layer_transitions = correlation_metrics.get("layer_transitions", {})
    layer_corr_formatted = "\n".join(
        [f"- {layer1} → {layer2}: Correlation = {data.get('correlation', 0.0):.4f}"
         for (layer1, layer2), data in layer_transitions.items()]
    )
    
    # Format strong examples
    strong_examples_formatted = ", ".join(
        [f"'{token}' ({corr:.3f})" for token, corr in strong_examples[:3]]
    )
    
    # Format weak examples
    weak_examples_formatted = ", ".join(
        [f"'{token}' ({corr:.3f})" for token, corr in weak_examples[:3]]
    )
    
    return GPT2_PATH_ATTENTION_CORRELATION_PROMPT.format(
        avg_correlation=avg_correlation,
        attn_follows_paths=attn_follows_paths,
        paths_follow_attn=paths_follow_attn,
        layer_correlations=layer_corr_formatted,
        strong_examples=strong_examples_formatted,
        weak_examples=weak_examples_formatted
    )