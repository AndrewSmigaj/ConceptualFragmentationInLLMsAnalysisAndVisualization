

Foundations of Archetypal Path Analysis: Toward a Principled Geometry for Cluster-Based Interpretability
Author: Andrew Smigaj, Gpt-4o, Grok, o3, Claude (Draft)
Abstract
Archetypal Path Analysis (APA) interprets neural networks by tracking datapoints through clustered activation spaces across layers. While initial experiments reveal meaningful internal structure, fairness dynamics, and decision heuristics, APA’s mathematical foundation remains underdeveloped. We formalize the conditions under which activation-space clustering is valid, analyze sensitivity to distance metrics and parameters, incorporate Explainable Threshold Similarity (ETS, τ_j) for transparent cluster definitions, and relate latent trajectories to established notions of similarity and information flow. By using layer-specific cluster labels and validating path convergence with geometric similarity, we ensure clarity in feedforward networks. Large language models (LLMs) generate human-readable narratives for these paths, enhancing interpretability.
Research Question — One-Sentence Focus: Under what geometric and statistical conditions do layerwise activation clusters form stable, semantically meaningful archetypal paths that can be used for faithful model interpretation?
1 Introduction
Interpretability research oscillates between visually compelling but loosely grounded “quasi-explanations” and rigorously defined but narrow theoretical analyses. APA aims for a middle path by clustering datapoint activations at each layer of a trained network and tracing their transitions. Its long-term utility depends on addressing foundational questions:
What makes activation geometry suitable for clustering?
Under what transformations are cluster paths stable?
Do cluster paths reflect genuine semantic or decision-relevant structure?
Contributions:
Formalize activation-space geometry for datapoint clustering with layer-specific labels.
Introduce ETS-based clustering for dimension-wise explainability.
Propose robustness metrics (silhouette, ARI, null models, MI, interestingness) for validating archetypal paths.
Benchmark against simpler attribution and clustering baselines.
Provide a lightweight empirical blueprint for reproducible implementation.
Enhance interpretability with concept-based annotations, feature attribution, and LLM-narrated paths.
2 Background and Motivation
2.1 Archetypal Path Analysis Recap
Archetypal Path Analysis (APA) clusters datapoint activations in each layer’s activation space (e.g., using k-means or DBSCAN), assigning layer-specific cluster IDs, denoted ( LlCk ), where ( l ) is the layer index and ( k ) is the cluster index (e.g., L1C0 for cluster 0 in layer 1). Transitions between clusters are tracked across layers, forming paths \pi_i = [c_i^1, c_i^2, \dots, c_i^L], interpreted as latent semantic trajectories. In feedforward networks, paths are strictly unidirectional, and clusters with different layer-specific IDs (e.g., L1C0 and L3C0) are not assumed to be related unless validated by geometric similarity (e.g., centroid cosine similarity). Large language models (LLMs) can narrate these trajectories to provide interpretable insights.
2.2 Critique from Foundational Viewpoints
Activation spaces are emergent, high-dimensional representations whose coordinate systems may not map to semantically meaningful axes. Without grounding, Euclidean distances can be misleading, and clustering is sensitive to initialization and density assumptions. Prior work echoes this concern: Ribeiro et al. (2016) and Lundberg & Lee (2017) show that LIME and SHAP trade exactness for intuition; Dasgupta et al. (2020) advocate explainable clustering by rule-based thresholds rather than arbitrary distances.
3 Mathematical Foundation of Layerwise Activation Geometry
3.1 What Is Being Clustered?
Let A^l \in \mathbb{R}^{n \times d_l} denote the matrix of activations at layer ( l ), where each row \mathbf{a}_i^l is a datapoint’s activation vector. Once the model is trained, A^l provides a static representation space per layer.
3.2 Metric Selection and Validity
We cluster A^l into k_l clusters, assigning layer-specific labels LlC0, LlC1, \dots, LlC{k_l-1}. A datapoint’s path is a sequence \pi_i = [c_i^1, c_i^2, \dots, c_i^L], where c_i^l = LlCk is the cluster assignment at layer ( l ). We examine Euclidean, cosine, and Mahalanobis metrics. In high-dimensional spaces, Euclidean norms lose contrast; cosine and L1 often behave better. PCA or normalization can stabilize comparisons. In feedforward networks, paths are unidirectional, and apparent convergence (e.g., [L1C0 \rightarrow L2C2 \rightarrow L3C0]) is validated by computing cosine or Euclidean similarity between cluster centroids across layers, ensuring that any perceived similarity reflects geometric proximity in activation space rather than shared labels.
3.3 Geometry Stability Across Training Seeds
We quantify stability with Earth Mover’s Distance and adjusted Rand Index (ARI) across retrained models, pruning, and dropout. Stable archetypal paths indicate robustness to training noise. Cluster transition matrices are visualized and analyzed via entropy and sparsity. To assess whether paths exhibit similarity-convergent behavior (e.g., [L1C0 \rightarrow L2C2 \rightarrow L3C0] where L3C0 resembles L1C0), we compute centroids \mathbf{c}_k^l for each cluster ( LlCk ) as the mean of activation vectors \mathbf{a}_i^l for datapoints ( i ) in cluster ( k ) at layer ( l ). We calculate pairwise cosine similarity, \text{cos}(\mathbf{c}_k^l, \mathbf{c}_j^m) = \frac{\mathbf{c}_k^l \cdot \mathbf{c}_j^m}{\|\mathbf{c}_k^l\| \|\mathbf{c}_j^m\|}, or Euclidean distance, d(\mathbf{c}_k^l, \mathbf{c}_j^m) = \|\mathbf{c}_k^l - \mathbf{c}_j^m\|_2, between clusters across layers (e.g., L1C0 and L3C0). High similarity (e.g., \text{cos}(\mathbf{c}_0^1, \mathbf{c}_0^3) > 0.9) indicates a similarity-convergent path, suggesting semantic consistency.
3.4 Explainable Threshold Similarity (ETS)
Following Kovalerchuk & Huber (2024), ETS declares two activations similar if |a_i^l_j - a_k^l_j| \leq \tau_j for every dimension ( j ). Cluster membership can therefore be verbalized as “neuron ( j ) differs by less than \tau_j,” yielding transparent, per-dimension semantics. We propose ETS as satisfying desiderata that centroid-based methods lack:
Dimension-wise interpretability
Explicit bounds on membership
Compatibility with heterogeneous feature scales
3.5 Concept-Based Cluster Annotation
To enhance interpretability, we integrate concept-based annotations using methods like Testing with Concept Activation Vectors (TCAV). For each cluster ( LlCk ), we compute its alignment with human-defined concepts (e.g., “positive sentiment,” “action verbs”) by measuring the sensitivity of activations to concept vectors. This allows us to label clusters with meaningful descriptors, enriching the interpretability of paths. For example, a path transitioning from a “neutral” cluster in layer 1 to a “negative sentiment” cluster in layer 3 can be narrated as reflecting a shift in the model’s internal evaluation of the input.
3.6 Baseline Benchmarking
To ensure APA’s added complexity is justified, we benchmark against:
Random clustering as a null model
Centroid-free ETS grouping
Simple attribution methods (saliency, IG) Improvements are validated via paired tests on silhouette and MI scores.
4 Statistical Robustness of Cluster Structures
We evaluate cluster coherence and transition stability using:
Silhouette Score (distance-based cohesion/separation)
Adjusted Rand Index (ARI) across runs/layers
Mutual Information (MI) with ground-truth labels
Null-model baselines (shuffled activations)
Dimension-wise ETS agreement ratio
Path purity scores relative to label distribution
4.1 Path Reproducibility Across Seeds
To assess structural stability, we define dominant archetypal paths as frequent cluster sequences across datapoints. We compute Jaccard overlap and recurrence frequency of these paths across different random seeds or bootstrapped model runs. High path recurrence suggests the presence of model-internal decision logic rather than sampling artifacts. For clusters ( LlCk ) and ( LmCj ), let S_k^l = \{i \mid \text{datapoint } i \text{ in } LlCk\}. We compute Jaccard similarity, J(S_k^l, S_j^m) = \frac{|S_k^l \cap S_j^m|}{|S_k^l \cup S_j^m|}, to measure datapoint retention across layers. High overlap between clusters -- with high centroid similarity suggests stable group trajectories. We also compute the frequency of similarity-convergent paths by aggregating transitions where the final cluster resembles an earlier one, e.g., [L1Ck \rightarrow L2Cj \rightarrow L3Cm] where \text{cos}(\mathbf{c}_k^1, \mathbf{c}_m^3) > 0.9. Density is calculated as D = \sum_{\text{similarity-convergent paths}} T^1_{kj} T^2_{jm} \cdot \mathbb{1}[\text{cos}(\mathbf{c}_k^1, \mathbf{c}_m^3) > \theta], where T^l_{kj} is the transition count from ( LlCk ) to L(l+1)Cj. High density suggests latent funnels where datapoints converge to similar activation spaces.
4.2 Trajectory Coherence
For a datapoint ( i ) with path \pi_i = [c_i^1, c_i^2, \dots, c_i^L], we compute the fragmentation score using subspace angles between consecutive centroid transitions: F_i = \frac{1}{L-2} \sum_{t=2}^{L-1} \arccos\left(\frac{(\mathbf{c}_{c_i^{t+1}}^{t+1} - \mathbf{c}_{c_i^t}^t) \cdot (\mathbf{c}_{c_i^t}^t - \mathbf{c}_{c_i^{t-1}}^{t-1})}{\|\mathbf{c}_{c_i^{t+1}}^{t+1} - \mathbf{c}_{c_i^t}^t\| \|\mathbf{c}_{c_i^t}^t - \mathbf{c}_{c_i^{t-1}}^{t-1}\|}\right). Low F_i indicates coherent trajectories, especially in similarity-convergent paths.
4.3 Feature Attribution for Cluster Transitions
To understand why datapoints transition between clusters, we apply feature attribution methods such as Integrated Gradients (IG) or SHAP. These methods identify which input features (e.g., specific tokens in text) most influence the activation changes driving cluster transitions. For instance, if a datapoint moves from L1C0 to L2C2, IG can reveal that the token “excellent” was pivotal in this shift. This attribution is computed by integrating gradients along the path from a baseline input to the actual input, highlighting feature importance for each transition.
4.4 Path Interestingness Score
We define an “interestingness” score for paths to identify those that are particularly noteworthy. The score combines:
Transition Rarity: The inverse frequency of each transition, highlighting uncommon jumps.
Similarity Convergence: The cosine similarity between starting and ending clusters.
Coherence: The inverse of the fragmentation score. The interestingness score is computed as I(\pi_i) = \alpha \cdot \text{rarity}(\pi_i) + \beta \cdot \text{sim}(\mathbf{c}_{c_i^1}^1, \mathbf{c}_{c_i^L}^L) + \gamma \cdot \frac{1}{F_i}, where \alpha, \beta, \gamma are tunable weights. Paths with high interestingness are prioritized for LLM narration.
5 Experimental Design (Blueprint)
Datasets: Titanic (binary survival), UCI Heart Disease, Adult Income
Models: 2–3-layer MLPs and LLM feedforward substructures (e.g., MLP head blocks)
Metrics: Silhouette, ARI, MI, path reproducibility, path purity, centroid similarity (cosine/Euclidean), Jaccard similarity, fragmentation scores, similarity-convergent path density, interestingness score
Clustering: k-means and ETS comparison
Visuals: Alluvial cluster-transition diagrams, stepped-layer PCA plots, cluster transition matrices, interactive visualizations (Plotly/Bokeh)
Narrative: LLM-generated archetypal path explanations
5.1 Stepped-Layer Trajectory Visualization
PCA reduces activations to 2-D; layers are offset along Z so each datapoint traces a polyline \mathbf{v}_i^l in (x,y,l) space, revealing fragmentation, convergence, and drift. We recommend interactive visualizations using tools like Plotly or Bokeh, allowing users to click on paths to view similarity scores, filter by coherence metrics, or zoom into specific transitions.
6 Use Cases for LLMs
Prompt Strategy Evaluation: Compare similarity-convergent path density and fragmentation scores across prompt framings (e.g., Socratic vs. assertive) to reveal shifts in internal decision consistency.
Layerwise Ambiguity Detection: Identify prompt-token pairs with divergent latent paths across LLM layers, highlighting instability or multiple plausible completions.
Subgroup Drift Analysis: Track membership overlap for datapoint groups (e.g., positive vs. negative sentiment) across layers, using centroid similarity to identify convergence.
Behavioral Explanation: Generate LLM-authored natural language summaries for archetypal paths, e.g., “Datapoints in L1C0, characterized by [feature], transition to L3C2, which is geometrically similar (cosine similarity 0.92), indicating [semantic consistency].”
Failure Mode Discovery: Flag high-fragmentation paths as potential errors, e.g., misclassifications or hallucinations.
6.1 Example Use Cases
Prompt Engineering: Compare paths for prompts like “Tell me a story” vs. “Write a creative tale” to see how wording affects internal flow. The former might follow a more fragmented path, indicating uncertainty, while the latter converges quickly to a “creative” cluster.
Bias Detection: Analyze paths for inputs with gender pronouns (e.g., “he” vs. “she” in professional contexts) to detect divergent behavior. A path diverging at layer 2 for “she” might indicate biased feature weighting.
Error Analysis: Study paths for misclassified inputs to pinpoint failure points. A misclassified datapoint might exhibit a highly fragmented path, suggesting internal confusion.
7 Reproducibility and Open Science
Code and configs released under MIT license.
Seed lists and hyperparameters logged in JSON.
Dockerfile ensures environment parity.
Negative results and failed variants documented.
Code snippets for key components:
Clustering with Layer-Specific Labels:
python
from sklearn.cluster import KMeans
cluster_labels = [f"L{l}C{k}" for l, layer in enumerate(activations) for k in KMeans(n_clusters=k).fit(layer).labels_]
Centroid Similarity Calculation:
python
from sklearn.metrics.pairwise import cosine_similarity
centroids = [np.mean(layer[labels == k], axis=0) for layer, labels in zip(activations, cluster_labels)]
sim_matrix = cosine_similarity(centroids)
Generating LLM Narratives:
python
narrative = f"Datapoint {i} follows path {path}, indicating a shift from {concept_start} to {concept_end}."
Interactive demos available at [link].
8 Societal Impact
APA’s narratives can demystify opaque models but risk misuse if interpreted uncritically. By using layer-specific cluster labels, APA avoids misleading implications of cyclic behavior in feedforward networks, enhancing transparency. However, similarity-convergent paths risk overinterpretation as causal relationships. We mitigate this by validating convergence with centroid similarity and null-model baselines, ensuring narratives align with IEEE and EU AI guidelines.
8.1 Limitations and Risks
Overinterpretation: APA narratives reflect patterns in activation spaces, not causal relationships. Users should avoid inferring definitive explanations from paths alone.
Scalability: Applying APA to massive models (e.g., LLMs with billions of parameters) may be computationally intensive. We recommend sampling datapoints or using approximate clustering methods.
Validation: Narratives should be cross-checked with domain expertise and other interpretability methods to ensure accuracy. To mitigate these risks, LLM-generated narratives include disclaimers, e.g., “This description is a hypothesis based on activation patterns, not a definitive explanation.”
9 Future Directions for Archetypal Path Analysis
Extend APA to Large Language Models (LLMs): Apply APA to full LLMs like GPT-2 to trace archetype paths across all layers, using PCA to visualize high-dimensional activation spaces and top-k neuron analysis to pinpoint which neurons drive cluster transitions.
Enhance Interpretability with Concept-Based Path Annotations: Integrate TCAV to align archetype paths with human-defined concepts, making transitions more interpretable by linking them to meaningful semantic changes.
Attribute Features to Path Transitions: Use Integrated Gradients (IG) to identify which input features are responsible for datapoint transitions between clusters, providing a clear, feature-level explanation of path dynamics.
Validate Path Robustness with Topological Analysis: Employ persistent homology to analyze the topological structure of activation spaces, ensuring that archetype paths and fragmentation patterns are stable.
Develop LLM-Narrated Path Explanations: Leverage LLMs to generate human-readable narratives for archetype paths (e.g., “this text shifted clusters due to stronger negative sentiment”), validating narratives against cluster properties.
Extend to Attention-Based LLMs: Use layer-specific labels to track token-level similarity-convergent paths in transformer models.
Explore Interesting Paths: Use the interestingness score to prioritize paths for further study, especially in reinforcement learning or policy networks.
References
(To be added: Kovalerchuk & Huber (2024), Ribeiro et al., Lundberg & Lee, Dasgupta et al., Sundararajan et al., Kim et al., Gower, von Luxburg, Ansuini, etc.)

