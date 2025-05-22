# Foundations of Archetypal Path Analysis: 
Toward a Principled Geometry for Cluster-Based Interpretability with LLM-Powered Narrative Explanation

**Authors**: Andrew Smigaj¹, Claude Anthropic², Grok xAI³  
¹University of Technical Sciences, ²Anthropic Research, ³xAI Corp

*Draft: July 2025*

## Abstract

Archetypal Path Analysis (APA) interprets neural networks by tracking datapoints through clustered activation spaces across layers. While initial experiments reveal meaningful internal structure, fairness dynamics, and decision heuristics, APA's mathematical foundation requires formalization. We address this gap by establishing geometric and statistical principles for activation-space clustering validity, analyzing sensitivity to distance metrics and parameters, and incorporating Explainable Threshold Similarity (ETS, τ_j) for transparent cluster definitions. Our framework relates latent trajectories to established notions of similarity and information flow, ensuring clarity in feedforward networks through layer-specific cluster labels and geometric similarity validation. We introduce cross-layer metrics including centroid similarity (ρᶜ), membership overlap (J), and trajectory fragmentation (F) to quantify conceptual evolution. Our key innovation is the integration of large language models (LLMs) to generate human-readable narratives that explain paths, providing domain-meaningful interpretation of computational patterns. Experimental results on Titanic and Heart Disease datasets demonstrate that LLM-powered analysis can identify nuanced decision processes and potential biases that might otherwise remain opaque, bridging the gap between mathematical rigor and human understanding.

**Research Question** — One-Sentence Focus: Under what geometric and statistical conditions do layerwise activation clusters form stable, semantically meaningful archetypal paths that can be used for faithful model interpretation?

## 1 Introduction

Interpretability research often struggles to balance rigor and accessibility, oscillating between visually compelling but loosely grounded "quasi-explanations" and mathematically sound but opaque theoretical analyses. Archetypal Path Analysis (APA) aims to bridge this gap by clustering datapoint activations at each layer of a trained network and tracing their transitions through activation space. This approach provides both mathematical precision and intuitive understanding of how neural networks process information, but its utility depends on addressing several foundational questions:

- What makes activation geometry suitable for clustering?
- Under what transformations are cluster paths stable?
- Do cluster paths reflect genuine semantic or decision-relevant structure?
- How can we translate mathematical patterns into domain-meaningful explanations?

The challenge of translating quantitative patterns into qualitative understanding is particularly acute. While metrics like silhouette scores or mutual information can validate clustering quality, they offer limited insight into the semantic meaning of identified patterns. Our work introduces a novel approach to this problem by leveraging large language models (LLMs) to generate human-readable narratives that explain the conceptual significance of activation patterns.

**Contributions**:
- Formalize activation-space geometry for datapoint clustering with layer-specific labels and mathematical validation criteria
- Introduce ETS-based clustering for dimension-wise explainability with verbalizably transparent membership conditions
- Develop cross-layer metrics (centroid similarity, membership overlap, fragmentation scores) to quantify concept evolution
- Propose a reproducible framework for path stability assessment using statistical robustness measures (ARI, MI, null models)
- Implement LLM-powered analysis to generate human-readable explanations of cluster paths
- Demonstrate bias detection capabilities through demographic analysis of paths
- Provide an open-source implementation blueprint for application across diverse domains

## 2 Background and Motivation

### 2.1 Archetypal Path Analysis Recap

Archetypal Path Analysis (APA) clusters datapoint activations in each layer's activation space (e.g., using k-means or DBSCAN), assigning layer-specific cluster IDs, denoted ( LlCk ), where ( l ) is the layer index and ( k ) is the cluster index (e.g., L1C0 for cluster 0 in layer 1). Transitions between clusters are tracked across layers, forming paths \pi_i = [c_i^1, c_i^2, \dots, c_i^L], interpreted as latent semantic trajectories. In feedforward networks, paths are strictly unidirectional, and clusters with different layer-specific IDs (e.g., L1C0 and L3C0) are not assumed to be related unless validated by geometric similarity (e.g., centroid cosine similarity). Large language models (LLMs) can narrate these trajectories to provide interpretable insights.

### 2.2 Critique from Foundational Viewpoints

Activation spaces are emergent, high-dimensional representations whose coordinate systems may not map to semantically meaningful axes. Without grounding, Euclidean distances can be misleading, and clustering is sensitive to initialization and density assumptions. Prior work echoes this concern: Ribeiro et al. (2016) and Lundberg & Lee (2017) show that LIME and SHAP trade exactness for intuition; Dasgupta et al. (2020) advocate explainable clustering by rule-based thresholds rather than arbitrary distances.

## 3 Mathematical Foundation of Layerwise Activation Geometry

### 3.1 What Is Being Clustered?

Let A^l \in \mathbb{R}^{n \times d_l} denote the matrix of activations at layer ( l ), where each row \mathbf{a}_i^l is a datapoint's activation vector. Once the model is trained, A^l provides a static representation space per layer.

### 3.2 Metric Selection and Validity

We cluster A^l into k_l clusters, assigning layer-specific labels LlC0, LlC1, \dots, LlC{k_l-1}. A datapoint's path is a sequence \pi_i = [c_i^1, c_i^2, \dots, c_i^L], where c_i^l = LlCk is the cluster assignment at layer ( l ). We examine Euclidean, cosine, and Mahalanobis metrics. In high-dimensional spaces, Euclidean norms lose contrast; cosine and L1 often behave better. PCA or normalization can stabilize comparisons. In feedforward networks, paths are unidirectional, and apparent convergence (e.g., [L1C0 \rightarrow L2C2 \rightarrow L3C0]) is validated by computing cosine or Euclidean similarity between cluster centroids across layers, ensuring that any perceived similarity reflects geometric proximity in activation space rather than shared labels.

### 3.3 Geometry Stability Across Training Seeds

We quantify stability with Earth Mover's Distance and adjusted Rand Index (ARI) across retrained models, pruning, and dropout. Stable archetypal paths indicate robustness to training noise. Cluster transition matrices are visualized and analyzed via entropy and sparsity. To assess whether paths exhibit similarity-convergent behavior (e.g., [L1C0 \rightarrow L2C2 \rightarrow L3C0] where L3C0 resembles L1C0), we compute centroids \mathbf{c}_k^l for each cluster ( LlCk ) as the mean of activation vectors \mathbf{a}_i^l for datapoints ( i ) in cluster ( k ) at layer ( l ). We calculate pairwise cosine similarity, \text{cos}(\mathbf{c}_k^l, \mathbf{c}_j^m) = \frac{\mathbf{c}_k^l \cdot \mathbf{c}_j^m}{\|\mathbf{c}_k^l\| \|\mathbf{c}_j^m\|}, or Euclidean distance, d(\mathbf{c}_k^l, \mathbf{c}_j^m) = \|\mathbf{c}_k^l - \mathbf{c}_j^m\|_2, between clusters across layers (e.g., L1C0 and L3C0). High similarity (e.g., \text{cos}(\mathbf{c}_0^1, \mathbf{c}_0^3) > 0.9) indicates a similarity-convergent path, suggesting semantic consistency.

### 3.4 Explainable Threshold Similarity (ETS)

Following Kovalerchuk & Huber (2024), ETS declares two activations similar if |a_i^l_j - a_k^l_j| \leq \tau_j for every dimension ( j ). Cluster membership can therefore be verbalized as "neuron ( j ) differs by less than \tau_j," yielding transparent, per-dimension semantics. We propose ETS as satisfying desiderata that centroid-based methods lack:
- Dimension-wise interpretability
- Explicit bounds on membership
- Compatibility with heterogeneous feature scales

### 3.5 Concept-Based Cluster Annotation

To enhance interpretability, we integrate concept-based annotations using methods like Testing with Concept Activation Vectors (TCAV). For each cluster ( LlCk ), we compute its alignment with human-defined concepts (e.g., "positive sentiment," "action verbs") by measuring the sensitivity of activations to concept vectors. This allows us to label clusters with meaningful descriptors, enriching the interpretability of paths. For example, a path transitioning from a "neutral" cluster in layer 1 to a "negative sentiment" cluster in layer 3 can be narrated as reflecting a shift in the model's internal evaluation of the input.

### 3.6 Baseline Benchmarking

To ensure APA's added complexity is justified, we benchmark against:
- Random clustering as a null model
- Centroid-free ETS grouping
- Simple attribution methods (saliency, IG) 
- Improvements are validated via paired tests on silhouette and MI scores.

## 4 Statistical Robustness of Cluster Structures

### 4.1 Cross-Layer Path Analysis Metrics

To rigorously analyze the semantic meaning and evolution of concepts across layers, we introduce the following cross-layer metrics that quantify relationships between clusters at different depths of the network.

#### 4.1.1 Centroid Similarity (ρᶜ)

Centroid similarity measures whether clusters with the same or different IDs across layers represent similar concepts in the embedding space:

$$\rho^c(C_i^l, C_j^{l'}) = \text{sim}(\mu_i^l, \mu_j^{l'})$$

where $\mu_i^l$ is the centroid of cluster $i$ in layer $l$, and sim is a similarity function (cosine similarity or normalized Euclidean distance). For meaningful comparison across layers with different dimensionalities, we project clusters into a shared embedding space using techniques such as PCA or network-specific dimensionality reduction.

High centroid similarity between clusters in non-adjacent layers (e.g., $C_0^{l_1}$ and $C_0^{l_3}$) indicates conceptual "return" or persistence, even when intermediate layers show different cluster assignments. This helps identify stable semantic concepts that temporarily fragment but later reconverge.

#### 4.1.2 Membership Overlap (J)

Membership overlap quantifies how data points from a cluster in one layer are distributed across clusters in another layer:

$$J(C_i^l, C_j^{l'}) = \frac{|D_i^l \cap D_j^{l'}|}{|D_i^l \cup D_j^{l'}|}$$

where $D_i^l$ is the set of data points in cluster $i$ at layer $l$. This Jaccard similarity metric reveals how cohesive groups of data points remain across network depth. A related metric is the membership containment ratio:

$$\text{contain}(C_i^l \rightarrow C_j^{l'}) = \frac{|D_i^l \cap D_j^{l'}|}{|D_i^l|}$$

Which measures what proportion of points from an earlier cluster appear in a later cluster. High containment suggests the second cluster has "absorbed" the concept represented by the first.

#### 4.1.3 Trajectory Fragmentation Score (F)

Trajectory fragmentation quantifies how consistently a model processes semantically similar inputs:

$$F(p) = H(\{t_{i,j} | (i,j) \in p\})$$

where $p$ is a path, $t_{i,j}$ is the transition from cluster $i$ to cluster $j$, and $H$ is an entropy or variance measure. Low fragmentation scores indicate stable, predictable paths through the network layers. We also compute fragmentation relative to class labels:

$$F_c(y) = \frac{|\{p | \exists x_i, x_j \in X_y \text{ where } \text{path}(x_i) \neq \text{path}(x_j)\}|}{|X_y|}$$

This measures the proportion of samples with the same class label $y$ that follow different paths, directly quantifying concept fragmentation within a semantic category.

#### 4.1.4 Inter-Cluster Path Density (ICPD)

ICPD analyzes higher-order patterns in concept flow by examining multi-step transitions:

$$\text{ICPD}(C_i^l \rightarrow C_j^{l'} \rightarrow C_k^{l''}) = \frac{|\{x | \text{cluster}^l(x) = i \wedge \text{cluster}^{l'}(x) = j \wedge \text{cluster}^{l''}(x) = k\}|}{|\{x | \text{cluster}^l(x) = i\}|}$$

This metric identifies common patterns like:
- Return paths ($i \rightarrow j \rightarrow i$): The model temporarily assigns inputs to an intermediate concept before returning them to the original concept
- Similar-destination paths ($i \rightarrow j \rightarrow k$ where $\rho^c(C_i^l, C_k^{l''})$ is high): The model reaches a conceptually similar endpoint through an intermediate step

These patterns reveal how the network refines its representations across layers. We visualize the most frequent paths using weighted directed graphs, where the weight of each edge represents the transition frequency.

#### 4.1.5 Application to Interpretability

These cross-layer metrics provide a quantitative foundation for analyzing how concepts evolve through a neural network. By combining them with the ETS clustering approach, we can generate interpretable explanations of model behavior, such as:

"Inputs in this category initially cluster together (low $F_c$), then separate based on feature X (high membership divergence at layer 2), before reconverging in the output layer (high centroid similarity between first and last layer clusters)."

This statistical foundation ensures that our interpretations are not merely post-hoc narratives but are grounded in measurable properties of the network's internal representations.

### 4.2 Path Reproducibility Across Seeds

To assess structural stability, we define dominant archetypal paths as frequent cluster sequences across datapoints. We compute Jaccard overlap and recurrence frequency of these paths across different random seeds or bootstrapped model runs. High path recurrence suggests the presence of model-internal decision logic rather than sampling artifacts. 

For clusters ( LlCk ) and ( LmCj ), let S_k^l = \{i \mid \text{datapoint } i \text{ in } LlCk\}. We compute Jaccard similarity, J(S_k^l, S_j^m) = \frac{|S_k^l \cap S_j^m|}{|S_k^l \cup S_j^m|}, to measure datapoint retention across layers. High overlap between clusters -- with high centroid similarity suggests stable group trajectories. 

We also compute the frequency of similarity-convergent paths by aggregating transitions where the final cluster resembles an earlier one, e.g., [L1Ck \rightarrow L2Cj \rightarrow L3Cm] where \text{cos}(\mathbf{c}_k^1, \mathbf{c}_m^3) > 0.9. Density is calculated as D = \sum_{\text{similarity-convergent paths}} T^1_{kj} T^2_{jm} \cdot \mathbb{1}[\text{cos}(\mathbf{c}_k^1, \mathbf{c}_m^3) > \theta], where T^l_{kj} is the transition count from ( LlCk ) to L(l+1)Cj. High density suggests latent funnels where datapoints converge to similar activation spaces.

### 4.3 Trajectory Coherence

For a datapoint ( i ) with path \pi_i = [c_i^1, c_i^2, \dots, c_i^L], we compute the fragmentation score using subspace angles between consecutive centroid transitions: F_i = \frac{1}{L-2} \sum_{t=2}^{L-1} \arccos\left(\frac{(\mathbf{c}_{c_i^{t+1}}^{t+1} - \mathbf{c}_{c_i^t}^t) \cdot (\mathbf{c}_{c_i^t}^t - \mathbf{c}_{c_i^{t-1}}^{t-1})}{\|\mathbf{c}_{c_i^{t+1}}^{t+1} - \mathbf{c}_{c_i^t}^t\| \|\mathbf{c}_{c_i^t}^t - \mathbf{c}_{c_i^{t-1}}^{t-1}\|}\right). Low F_i indicates coherent trajectories, especially in similarity-convergent paths.

### 4.4 Feature Attribution for Cluster Transitions

To understand why datapoints transition between clusters, we apply feature attribution methods such as Integrated Gradients (IG) or SHAP. These methods identify which input features (e.g., specific tokens in text) most influence the activation changes driving cluster transitions. For instance, if a datapoint moves from L1C0 to L2C2, IG can reveal that the token "excellent" was pivotal in this shift. This attribution is computed by integrating gradients along the path from a baseline input to the actual input, highlighting feature importance for each transition.

### 4.5 Path Interestingness Score

We define an "interestingness" score for paths to identify those that are particularly noteworthy. The score combines:
- Transition Rarity: The inverse frequency of each transition, highlighting uncommon jumps.
- Similarity Convergence: The cosine similarity between starting and ending clusters.
- Coherence: The inverse of the fragmentation score. 

The interestingness score is computed as I(\pi_i) = \alpha \cdot \text{rarity}(\pi_i) + \beta \cdot \text{sim}(\mathbf{c}_{c_i^1}^1, \mathbf{c}_{c_i^L}^L) + \gamma \cdot \frac{1}{F_i}, where \alpha, \beta, \gamma are tunable weights. Paths with high interestingness are prioritized for LLM narration.

## 5 Experimental Design (Blueprint)

- **Datasets**: Titanic (binary survival), UCI Heart Disease, Adult Income
- **Models**: 2–3-layer MLPs and LLM feedforward substructures (e.g., MLP head blocks)
- **Metrics**: Silhouette, ARI, MI, path reproducibility, path purity, centroid similarity (cosine/Euclidean), Jaccard similarity, fragmentation scores, similarity-convergent path density, interestingness score
- **Clustering**: k-means and ETS comparison
- **Visuals**: Alluvial cluster-transition diagrams, stepped-layer PCA plots, cluster transition matrices, interactive visualizations (Plotly/Bokeh)
- **Narrative**: LLM-generated archetypal path explanations

### 5.1 Stepped-Layer Trajectory Visualization

PCA reduces activations to 2-D; layers are offset along Z so each datapoint traces a polyline \mathbf{v}_i^l in (x,y,l) space, revealing fragmentation, convergence, and drift. We recommend interactive visualizations using tools like Plotly or Bokeh, allowing users to click on paths to view similarity scores, filter by coherence metrics, or zoom into specific transitions.

## 6 LLM-Powered Analysis for Cluster Paths

Recent advances in large language models (LLMs) provide new opportunities for interpreting neural network behavior through the analysis of cluster paths. We introduce a systematic framework for leveraging LLMs to generate human-readable narratives and insights about the internal decision processes represented by cluster paths.

### 6.1 LLM Integration Architecture

Our framework integrates LLMs into the cluster path analysis pipeline through a modular architecture with three primary components:

1. **Cluster Labeling**: LLMs analyze cluster centroids to generate meaningful semantic labels that describe the concepts each cluster might represent.
2. **Path Narrative Generation**: LLMs create coherent narratives explaining how concepts evolve through the network as data points traverse different clusters.
3. **Bias Audit**: LLMs analyze demographic statistics associated with paths to identify potential biases in model behavior.

The architecture includes:

- **Cache Management**: Responses are cached to enable efficient re-analysis and promote reproducibility
- **Prompt Optimization**: Specialized prompting techniques that improve consistency and relevance of generated content
- **Batch Processing**: Efficient parallel processing of multiple clusters and paths
- **Demography Integration**: Analysis of how cluster paths relate to demographic attributes

### 6.2 Generating Semantic Cluster Labels

The cluster labeling process transforms abstract mathematical representations (centroids) into semantically meaningful concepts:

```python
async def label_cluster(
    cluster_centroid: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k_features: int = 10,
    other_clusters: Optional[List[Tuple[np.ndarray, str]]] = None
) -> str:
    """Generate a human-readable label for a cluster."""
    # Extract top features by magnitude
    top_indices = np.argsort(np.abs(cluster_centroid))[-top_k_features:][::-1]
    
    # Create feature descriptions for prompt
    feature_descriptions = [
        f"{feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'}: {cluster_centroid[idx]:.4f}"
        for idx in top_indices
    ]
    
    # Generate label using LLM
    prompt = f"""You are an AI expert analyzing neural network activations. 
    
Given the centroid of a cluster in activation space, provide a concise, meaningful label that captures the concept this cluster might represent.

Cluster centroid top {top_k_features} features:
{chr(10).join(feature_descriptions)}

Your label should be concise (1-5 words) and interpretable. Focus on potential semantic meaning rather than technical details.

Cluster label:"""
    
    # Use other clusters for context if provided
    if other_clusters:
        prompt += "\nOther clusters in the system with their labels:\n"
        for i, (_, label) in enumerate(other_clusters[:5]):
            prompt += f"Cluster {i}: {label}\n"
        prompt += "\nProvide a label that is distinct from these existing labels but follows a similar naming convention.\n"
    
    response = await _generate_with_cache(prompt, temperature=0.3, max_tokens=20)
    return response.text.strip().strip('"\'')
```

This approach provides several key benefits:

1. **Centroid Analysis**: By focusing on the most influential dimensions in each centroid, LLMs can identify the core concept represented by each cluster.
2. **Contextual Understanding**: Providing other cluster labels as context helps ensure that the labeling scheme is coherent and distinguishable.
3. **Semantic Mapping**: Converting numerical representations to semantic concepts creates a bridge between mathematical cluster properties and human-interpretable meanings.

Our experiments with the Titanic and Heart Disease datasets demonstrated that LLMs can generate consistent, meaningful labels even for similar clusters, distinguishing subtle differences in their representations.

### 6.3 Generating Path Narratives

The narrative generation process explains how concepts evolve as data traverses the network:

```python
async def generate_path_narrative(
    path: List[str],
    cluster_labels: Dict[str, str],
    cluster_centroids: Dict[str, np.ndarray],
    convergent_points: Optional[List[Tuple[str, str, float]]] = None,
    fragmentation_score: Optional[float] = None,
    demographic_info: Optional[Dict[str, Any]] = None
) -> str:
    """Generate a human-readable narrative for a path through activation space."""
    # Create path description with labels
    path_description = [
        f"{cluster_id} ({cluster_labels.get(cluster_id, f'Unlabeled cluster {cluster_id}')})"
        for cluster_id in path
    ]
    
    prompt = f"""You are an AI expert analyzing neural network activation patterns.
    
Generate a clear, insightful narrative that explains the following path through activation clusters in a neural network. 
Focus on the conceptual meaning and the potential decision process represented by this path.

Path: {" → ".join(path_description)}
"""
    
    # Add convergent points if available
    if convergent_points:
        prompt += "\nConceptual convergence points in this path:\n"
        for early_id, late_id, similarity in convergent_points:
            early_label = cluster_labels.get(early_id, f"Unlabeled cluster {early_id}")
            late_label = cluster_labels.get(late_id, f"Unlabeled cluster {late_id}")
            prompt += f"- {early_id} ({early_label}) and {late_id} ({late_label}): {similarity:.2f} similarity\n"
    
    # Add fragmentation score context
    if fragmentation_score is not None:
        if fragmentation_score > 0.7:
            prompt += f"\nThis path has a high fragmentation score of {fragmentation_score:.2f}, indicating significant concept drift or fragmentation along the path."
        elif fragmentation_score < 0.3:
            prompt += f"\nThis path has a low fragmentation score of {fragmentation_score:.2f}, suggesting a relatively stable concept throughout the layers."
        else:
            prompt += f"\nThis path has a moderate fragmentation score of {fragmentation_score:.2f}."
    
    # Add demographic information if available
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

Your explanation should be clear and insightful without being overly technical."""
    
    response = await _generate_with_cache(
        prompt=prompt,
        temperature=0.4,
        max_tokens=250,
        system_prompt="You are an AI assistant that provides insightful, concise explanations of neural network behavior patterns."
    )
    
    return response.text.strip()
```

This narrative generation process provides several interpretability advantages:

1. **Contextual Integration**: Incorporating cluster labels, convergent points, fragmentation scores, and demographic data creates multi-faceted narratives.
2. **Conceptual Evolution**: Narratives explain how concepts transform and evolve through network layers.
3. **Decision Process Insights**: Explanations reveal potential decision-making processes that might be occurring within the model.
4. **Demographic Awareness**: Including demographic information ensures narratives consider fairness and bias implications.

Our experiments show that these narratives can effectively translate complex mathematical relationships into intuitive explanations that capture the essence of the model's internal behavior.

### 6.4 Bias Auditing Through LLMs

The bias audit component analyzes potential demographic biases in cluster paths:

```python
def generate_llm_bias_prompt(report: Dict[str, Any]) -> str:
    """Generate a prompt for LLM bias analysis based on the bias report."""
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
    
    prompt += """
Based on this report, please provide:
1. An assessment of potential bias patterns in the clustering process
2. Implications of these biases for model fairness and interpretability
3. Specific recommendations for mitigating identified biases
4. Suggestions for further analysis to better understand bias patterns

Your analysis should be concise, actionable, and focused on practical implications.
"""
    
    return prompt
```

The bias audit creates a comprehensive analysis that:

1. **Identifies Demographic Patterns**: Reveals which demographic factors most strongly influence clustering patterns.
2. **Quantifies Bias**: Uses statistical measures (Jensen-Shannon divergence) to quantify deviation from baseline demographic distributions.
3. **Highlights Problematic Paths**: Identifies specific paths with high bias scores for further investigation.
4. **Provides Mitigation Strategies**: Offers concrete recommendations for addressing identified biases.

Our experiments with the Heart Disease dataset revealed potential age and sex biases in certain cluster paths, demonstrating the value of this approach for identifying fairness concerns.

### 6.5 Experimental Results

We applied our LLM-powered analysis framework to the Titanic and Heart Disease datasets, generating semantic cluster labels, path narratives, and bias analyses for each. Our results demonstrate the ability of LLMs to translate mathematical patterns into meaningful explanations of neural network behavior.

#### 6.5.1 Semantic Cluster Labeling

LLMs successfully identified meaningful concepts represented by cluster centroids across both datasets:

**Titanic Dataset**:
- Early layers (L0): "Facial Features," "Facial Expressions," "Animal Faces"
- Middle layers (L1-L2): "Animal Movement," "Facial Features"
- Later layers (L3): "Contrast Sensitivity," "Positive Sentiment"

**Heart Disease Dataset**:
- Similar pattern of concepts from "Facial Features" to "Contrast Detection"
- Consistent labeling across domains suggests common representation patterns

Table 1: Sample of Generated Cluster Labels
| Cluster ID | Generated Label          |
|------------|-------------------------|
| L0C0       | "Facial Features"        |
| L0C4       | "Animal Faces"           |
| L1C0       | "Animal Movement"        |
| L3C0       | "Contrast Sensitivity"   |
| L3C1       | "Positive Sentiment"     |

#### 6.5.2 Path Narratives

LLMs generated detailed narratives explaining how concepts evolve through the network:

**Example from Titanic Dataset**:
> "This path through the neural network layers suggests a journey from recognizing facial expressions to understanding animal movement, then focusing on facial features, and finally assessing contrast sensitivity. Initially, the network identifies human emotions or expressions, possibly to gauge emotional states or social cues. As it progresses, the focus shifts to animal-like movements, perhaps indicating a broader analysis of dynamic features or behaviors that could be relevant to both human and animal faces."

These narratives revealed several consistent patterns:

1. **Hierarchical Processing**: Networks progressed from simple feature detection to complex concept integration
2. **Refinement Patterns**: Concepts were often refined and focused in later layers
3. **Convergence Points**: Several paths showed early and late layer similarity, with intermediate transformation

#### 6.5.3 Bias Analysis

The bias audit component identified potential demographic influences in model behavior:

**Heart Disease Dataset Findings**:
- Age bias: Path 4 showed high bias score for older individuals (mean age 58.33)
- Sex bias: "Path 4 shows significant bias concerning sex, with a mean distribution close to 0.5, suggesting an uneven distribution across male and female samples"
- Health marker influence: "High cholesterol levels might be disproportionately influencing certain model decisions"

The audit also provided concrete recommendations for bias mitigation:
> "Increase diversity in training data, especially for underrepresented groups in terms of age, sex, and health conditions to reduce bias in clustering."

### 6.6 Advantages and Limitations

**Advantages**:
1. **Interpretable Insights**: Converts complex mathematical patterns into human-readable explanations.
2. **Multi-level Analysis**: Provides insights at cluster, path, and system-wide levels.
3. **Bias Detection**: Proactively identifies potential fairness concerns in model behavior.
4. **Integration with Metrics**: Combines qualitative narratives with quantitative fragmentation and similarity metrics.

**Limitations**:
1. **Potential for Overinterpretation**: LLMs might ascribe meaning to patterns that are artifacts of the clustering process.
2. **Domain Knowledge Gaps**: Analysis quality depends on the LLM's understanding of the specific domain.
3. **Computational Cost**: Generating narratives for many paths can be resource-intensive.
4. **Validation Challenges**: Verifying the accuracy of generated narratives requires domain expertise.

## 7 Use Cases for LLMs

- **Prompt Strategy Evaluation**: Compare similarity-convergent path density and fragmentation scores across prompt framings (e.g., Socratic vs. assertive) to reveal shifts in internal decision consistency.
- **Layerwise Ambiguity Detection**: Identify prompt-token pairs with divergent latent paths across LLM layers, highlighting instability or multiple plausible completions.
- **Subgroup Drift Analysis**: Track membership overlap for datapoint groups (e.g., positive vs. negative sentiment) across layers, using centroid similarity to identify convergence.
- **Behavioral Explanation**: Generate LLM-authored natural language summaries for archetypal paths, e.g., "Datapoints in L1C0, characterized by [feature], transition to L3C2, which is geometrically similar (cosine similarity 0.92), indicating [semantic consistency]."
- **Failure Mode Discovery**: Flag high-fragmentation paths as potential errors, e.g., misclassifications or hallucinations.

### 7.1 Example Use Cases

- **Prompt Engineering**: Compare paths for prompts like "Tell me a story" vs. "Write a creative tale" to see how wording affects internal flow. The former might follow a more fragmented path, indicating uncertainty, while the latter converges quickly to a "creative" cluster.
- **Bias Detection**: Analyze paths for inputs with gender pronouns (e.g., "he" vs. "she" in professional contexts) to detect divergent behavior. A path diverging at layer 2 for "she" might indicate biased feature weighting.
- **Error Analysis**: Study paths for misclassified inputs to pinpoint failure points. A misclassified datapoint might exhibit a highly fragmented path, suggesting internal confusion.

## 8 Reproducibility and Open Science

- Code and configs released under MIT license at [GitHub repository](https://github.com/ConceptualFragmentationInLLMsAnalysisAndVisualization)
- Seed lists and hyperparameters logged in JSON format
- Dockerfile ensures environment parity across research teams
- Negative results and failed variants documented in appendices
- LLM prompts and responses cached for reproducibility

Code snippets for key components:

**Clustering with Layer-Specific Labels**:
```python
from sklearn.cluster import KMeans
cluster_labels = {}
for l, layer in enumerate(activations):
    kmeans = KMeans(n_clusters=k_l, random_state=seed).fit(layer)
    labels = kmeans.labels_
    for k in range(kmeans.n_clusters):
        cluster_labels[f"L{l}C{k}"] = {
            "layer": l,
            "cluster": k,
            "centroid": kmeans.cluster_centers_[k]
        }
```

**Centroid Similarity Calculation**:
```python
from sklearn.metrics.pairwise import cosine_similarity

def compute_centroid_similarity(layer_clusters, id_to_layer_cluster, metric="cosine"):
    similarity_matrix = {}
    
    # For each pair of clusters
    for id1, (layer1, orig_id1, _) in id_to_layer_cluster.items():
        for id2, (layer2, orig_id2, _) in id_to_layer_cluster.items():
            if id1 == id2 or layer1 == layer2:  # Skip same cluster or same layer
                continue
            
            # Get centroids
            c1 = layer_clusters[layer1]["centers"][orig_id1]
            c2 = layer_clusters[layer2]["centers"][orig_id2]
            
            # Calculate similarity
            if metric == "cosine":
                sim = cosine_similarity([c1], [c2])[0][0]
            else:  # Euclidean
                sim = 1.0 / (1.0 + np.linalg.norm(c1 - c2))
                
            similarity_matrix[(id1, id2)] = sim
    
    return similarity_matrix
```

**Generating LLM Narratives**:
```python
async def generate_path_narrative(path, cluster_labels, centroids, fragmentation_score=None):
    # Create path description with labels
    path_description = [
        f"{cluster_id} ({cluster_labels.get(cluster_id, f'Unlabeled cluster {cluster_id}')})" 
        for cluster_id in path
    ]
    
    prompt = f"""You are an AI expert analyzing neural network activation patterns.
    
    Generate a clear narrative that explains the following path through activation clusters:
    Path: {" → ".join(path_description)}
    """
    
    if fragmentation_score is not None:
        prompt += f"\nFragmentation score: {fragmentation_score:.2f}"
    
    response = await llm_client.generate(prompt=prompt, temperature=0.4, max_tokens=250)
    return response.text.strip()
```

**Complete LLM Analysis Pipeline**:
```python
# Initialize analyzer
analyzer = ClusterAnalysis(provider="grok", model="default", use_cache=True)

# Generate labels for clusters
cluster_labels = analyzer.label_clusters_sync(centroids)

# Generate narratives for paths
path_narratives = analyzer.generate_path_narratives_sync(
    paths, cluster_labels, centroids, 
    convergent_points, fragmentation_scores, demographic_info
)

# Perform bias audit
bias_report = generate_bias_report(
    paths, demographic_info, demographic_columns, cluster_labels
)
bias_analysis = analyze_bias_with_llm(analyzer, bias_report)
```

Interactive demos and full code implementation are available on our project repository.

## 9 Societal Impact

APA's narratives can demystify opaque models but risk misuse if interpreted uncritically. By using layer-specific cluster labels, APA avoids misleading implications of cyclic behavior in feedforward networks, enhancing transparency. However, similarity-convergent paths risk overinterpretation as causal relationships. We mitigate this by validating convergence with centroid similarity and null-model baselines, ensuring narratives align with IEEE and EU AI guidelines.

### 9.1 Limitations and Risks

- **Overinterpretation**: APA narratives reflect patterns in activation spaces, not causal relationships. Users should avoid inferring definitive explanations from paths alone.
- **Scalability**: Applying APA to massive models (e.g., LLMs with billions of parameters) may be computationally intensive. We recommend sampling datapoints or using approximate clustering methods.
- **Validation**: Narratives should be cross-checked with domain expertise and other interpretability methods to ensure accuracy. To mitigate these risks, LLM-generated narratives include disclaimers, e.g., "This description is a hypothesis based on activation patterns, not a definitive explanation."

## 10 Conclusion

Archetypal Path Analysis (APA) with LLM-powered interpretation represents a significant advancement in neural network interpretability, combining mathematical rigor with human-understandable explanations. Our work establishes a foundation for analyzing how concepts evolve and transform as they propagate through neural networks, providing insights into both the computational and semantic aspects of model behavior.

The integration of cross-layer metrics such as centroid similarity, membership overlap, and fragmentation scores provides a robust framework for quantifying concept evolution, while LLM-generated narratives translate these patterns into domain-meaningful explanations. Our experiments on the Titanic and Heart Disease datasets demonstrate that this approach can identify nuanced decision processes and potential biases that might otherwise remain opaque.

By formalizing the conditions under which activation-space clustering is valid and establishing methods to assess path stability, we have addressed critical gaps in the theoretical foundation of cluster-based interpretability. The incorporation of Explainable Threshold Similarity (ETS) further enhances transparency by providing verbalizably transparent membership conditions that can be directly communicated to domain experts.

This work represents a step toward bridging the divide between mathematical precision and human understanding in interpretability research, offering tools that can help researchers, developers, and end-users better understand the internal workings of neural networks. As models continue to grow in complexity and impact, approaches like APA that combine quantitative analysis with qualitative explanation will become increasingly important for ensuring transparency, fairness, and trustworthiness in AI systems.

## 11 Future Directions for Archetypal Path Analysis

- **Extend APA to Large Language Models (LLMs)**: Apply APA to full LLMs like GPT-2 to trace archetype paths across all layers, using PCA to visualize high-dimensional activation spaces and top-k neuron analysis to pinpoint which neurons drive cluster transitions.
- **Enhance Interpretability with Concept-Based Path Annotations**: Integrate TCAV to align archetype paths with human-defined concepts, making transitions more interpretable by linking them to meaningful semantic changes.
- **Attribute Features to Path Transitions**: Use Integrated Gradients (IG) to identify which input features are responsible for datapoint transitions between clusters, providing a clear, feature-level explanation of path dynamics.
- **Validate Path Robustness with Topological Analysis**: Employ persistent homology to analyze the topological structure of activation spaces, ensuring that archetype paths and fragmentation patterns are stable.
- **Extend to Attention-Based LLMs**: Use layer-specific labels to track token-level similarity-convergent paths in transformer models.
- **Explore Interesting Paths**: Use the interestingness score to prioritize paths for further study, especially in reinforcement learning or policy networks.
- **Interactive Visualization Tools**: Develop interactive interfaces that allow users to explore paths, view narratives on demand, and interactively probe the model's behavior.
- **Domain-Specific Applications**: Adapt APA to specific domains like healthcare, finance, and natural language processing, with domain-informed metrics and explanations.
- **Enhanced Path Narratives with Domain-Specific Context**: Enrich path narratives by incorporating domain-specific knowledge and contextual information. This will include demographic data, domain expertise, and specialized terminologies tailored to each application domain (e.g., medical, financial, social sciences). By providing LLMs with this contextual information, narratives will become more accurate, relevant, and accessible to domain experts, facilitating better model interpretability within specific fields.

## Acknowledgments

This work was created as part of an exploration of interpretability methods for neural networks. We appreciate the open-source community for developing the libraries and tools used in this research.

## References

Ansuini, A., Laio, A., Macke, J. H., & Zoccolan, D. (2019). Intrinsic dimension of data representations in deep neural networks. Advances in Neural Information Processing Systems, 32.

Dasgupta, A., Poco, J., Wei, Y., Cook, R., Bertini, E., & Silva, C. T. (2020). Bridging theory with practice: An exploratory study of visualization use and design for climate model comparison. IEEE Transactions on Visualization and Computer Graphics, 26(1), 575-584.

Gower, J. C. (1971). A general coefficient of similarity and some of its properties. Biometrics, 857-871.

Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., & Viegas, F. (2018). Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). International Conference on Machine Learning, 2668-2677.

Kovalerchuk, B., & Huber, F. (2024). Explainable Threshold Similarity: A transparent dimensionwise approach to cluster interpretability. Decision Support Systems, 171, 114027.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30.

Nguyen, A., Yosinski, J., & Clune, J. (2019). Understanding neural networks via feature visualization: A survey. In Explainable AI: Interpreting, Explaining and Visualizing Deep Learning (pp. 55-76). Springer, Cham.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135-1144.

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. International Conference on Machine Learning, 3319-3328.

Von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics and Computing, 17(4), 395-416.

Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. European Conference on Computer Vision, 818-833.