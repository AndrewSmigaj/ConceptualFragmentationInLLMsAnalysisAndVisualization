Title: Foundations of Archetypal Path Analysis: Toward a Principled Geometry for Cluster-Based Interpretability Author: Andrew Smigaj (Draft)
Abstract Archetypal Path Analysis (APA) interprets neural networks by tracking datapoints through clustered activation spaces across layers. While initial experiments reveal meaningful internal structure, fairness dynamics, and decision heuristics, APA’s mathematical foundation remains underdeveloped. We formalize the conditions under which activation‑space clustering is valid, analyze sensitivity to distance metrics and parameters, incorporate Explainable Threshold Similarity (ETS, τ_j) for transparent cluster definitions, and relate latent trajectories to established notions of similarity and information flow.
Research Question — One‑Sentence Focus Under what geometric and statistical conditions do layerwise activation clusters form stable, semantically meaningful archetypal paths that can be used for faithful model interpretation?

1  Introduction
Interpretability research oscillates between visually compelling but loosely grounded “quasi‑explanations” and rigorously defined but narrow theoretical analyses. APA aims for a middle path by clustering datapoint activations at each layer of a trained network and tracing their transitions. Its long‑term utility depends on addressing foundational questions: 1. What makes activation geometry suitable for clustering? 2. Under what transformations are cluster paths stable? 3. Do cluster paths reflect genuine semantic or decision‑relevant structure?
Contributions
Formalize activation-space geometry for datapoint clustering
Introduce ETS-based clustering for dimension-wise explainability
Propose robustness metrics (silhouette, ARI, null models, MI) for validating archetypal paths
Benchmark against simpler attribution and clustering baselines
Provide a lightweight empirical blueprint for reproducible implementation

2  Background and Motivation
2.1 Archetypal Path Analysis Recap
APA clusters datapoints in activation space (e.g., k‑means, DBSCAN) at each layer, then tracks cluster transitions layer‑to‑layer. These transitions are interpreted as latent semantic trajectories. LLMs can narrate these paths.
2.2 Critique from Foundational Viewpoints
Activation spaces are emergent, high‑dimensional representations whose coordinate systems may not map to semantically meaningful axes. Without grounding, Euclidean distances can be misleading, and clustering is sensitive to initialization and density assumptions.
Prior work echoes this concern: Ribeiro et al. (2016) and Lundberg & Lee (2017) show that LIME and SHAP trade exactness for intuition; Dasgupta et al. (2020) advocate explainable clustering by rule‑based thresholds rather than arbitrary distances.

3  Mathematical Foundation of Layerwise Activation Geometry
3.1 What Is Being Clustered?
Let denote the matrix of activations at layer , where each row is a datapoint's activation vector. Once the model is trained, provides a static representation space per layer.
3.2 Metric Selection and Validity
We examine Euclidean, cosine, and Mahalanobis metrics. In high‑dimensional spaces, Euclidean norms lose contrast; cosine and L1 often behave better. PCA or normalization can stabilize comparisons. We define paths as sequences:
 where is the cluster assignment of datapoint at layer .
3.3 Geometry Stability Across Training Seeds
We quantify stability with Earth Mover’s Distance and adjusted Rand Index (ARI) across retrained models, pruning, and dropout. Stable archetypal paths indicate robustness to training noise. Cluster transition matrices are visualized and analyzed via entropy and sparsity.
3.4 Explainable Threshold Similarity (ETS)
Following Kovalerchuk & Huber (2024), ETS declares two activations similar if |aᵢˡⱼ − aₖˡⱼ| ≤ τⱼ for every dimension j. Cluster membership can therefore be verbalized as “neuron j differs by less than τⱼ,” yielding transparent, per‑dimension semantics. We propose ETS as satisfying desiderata that centroid-based methods lack:
Dimension-wise interpretability
Explicit bounds on membership
Compatibility with heterogeneous feature scales
3.5 Baseline Benchmarking
To ensure APA’s added complexity is justified, we benchmark against:
Random clustering as a null model
Centroid-free ETS grouping
Simple attribution methods (saliency, IG) Improvements are validated via paired tests on silhouette and MI scores.

4  Statistical Robustness of Cluster Structures
We evaluate cluster coherence and transition stability using:
Silhouette Score (distance‑based cohesion/separation)
Adjusted Rand Index (ARI) across runs/layers
Mutual Information (MI) with ground‑truth labels
Null‑model baselines (shuffled activations)
Dimension-wise ETS agreement ratio
Path purity scores relative to label distribution
4.1 Path Reproducibility Across Seeds
To assess structural stability, we define dominant archetypal paths as frequent cluster sequences across datapoints. We compute Jaccard overlap and recurrence frequency of these paths across different random seeds or bootstrapped model runs. High path recurrence suggests the presence of model-internal decision logic rather than sampling artifacts.

5  Experimental Design (Blueprint)
Datasets: Titanic (binary survival), UCI Heart Disease, Adult Income
Models: 2–3‑layer MLPs and LLM feedforward substructures (e.g. MLP head blocks)
Metrics: Silhouette, ARI, MI, path reproducibility, path purity
Clustering: k‑means and ETS comparison
Visuals: Alluvial cluster‑transition diagrams, stepped‑layer PCA plots, cluster transition matrices
Narrative: LLM‑generated archetypal path explanations
5.1 Stepped‑Layer Trajectory Visualization
PCA reduces activations to 2‑D; layers are offset along Z so each datapoint traces a polyline 𝐯ᵢˡ in (x,y,l) space, revealing fragmentation, convergence, and drift.

6  Use Cases for LLMs
Prompt Strategy Evaluation: Compare archetypal paths under different prompt framings (e.g., Socratic vs. assertive) to reveal shifts in internal decision flow
Layerwise Ambiguity Detection: Identify prompt-token pairs with divergent latent paths across LLM layers, highlighting instability or multiple plausible completions
Subgroup Drift Analysis: Track how datapoints (e.g., questions of different tone or subject) move through latent clusters—useful for auditing model bias or representation collapse
Behavioral Explanation: Generate LLM-authored natural language summaries for archetypal paths, anchoring observed behaviors in structural transitions
Failure Mode Discovery: Flag latent path outliers corresponding to nonsensical or hallucinated generations

xxx

8  Reproducibility and Open Science
Code and configs released under MIT license
Seed lists and hyperparameters logged in JSON
Dockerfile ensures environment parity
Negative results and failed variants documented

9  Societal Impact

9.1 Societal Impact
APA’s narratives can demystify opaque models but risk misuse if interpreted uncritically. We discuss benefits (transparency) and risks (over‑trust, exposure of sensitive feature interactions) in alignment with IEEE and EU AI guidelines.

10  Future Directions for Archetypal Path Analysis
Extend APA to Large Language Models (LLMs)
Apply APA to full LLMs like GPT-2 to trace archetype paths across all layers, using PCA to visualize high-dimensional activation spaces and top-k neuron analysis to pinpoint which neurons drive cluster transitions, emphasizing where fragmentation (e.g., cluster divergence) occurs.
Enhance Interpretability with Concept-Based Path Annotations
Integrate Testing with Concept Activation Vectors (TCAV) to align archetype paths with human-defined concepts (e.g., sentiment or topic shifts in text), making transitions more interpretable by linking them to meaningful semantic changes.
Attribute Features to Path Transitions
Use Integrated Gradients (IG) to identify which input features (e.g., tokens or embeddings) are responsible for datapoint transitions between clusters, providing a clear, feature-level explanation of fragmentation and path dynamics.
Validate Path Robustness with Topological Analysis
Employ persistent homology to analyze the topological structure of activation spaces, ensuring that archetype paths and fragmentation patterns are stable and not artifacts of clustering or noise.
Develop LLM-Narrated Path Explanations
Leverage LLMs to generate human-readable narratives for archetype paths (e.g., “this text shifted clusters due to stronger negative sentiment”), enhancing interpretability while rigorously validating narratives against cluster properties.


References
(To be added: Kovalerchuk & Huber (2024), Ribeiro et al., Lundberg & Lee, Dasgupta et al., Sundararajan et al., Kim et al., Gower, von Luxburg, Ansuini, etc.)


