\documentclass{article}
\usepackage{iclr2025_conference,times}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{enumitem}
\usepackage{url}
\graphicspath{{figures/}}
\title{Concept Fragmentation in Neural Networks: Visualizing and Measuring Intra-Class Dispersion in Feedforward Models}

\author{Anonymous Submission}

\begin{document}

\maketitle

\begin{abstract}
Neural networks often encode class representations across disjoint latent regions, a phenomenon we term \textit{concept fragmentation}. This paper introduces a framework to quantify and interpret fragmentation in feedforward neural networks using metrics (cluster entropy, subspace angles, intra-class pairwise distance), trajectory visualizations, and a novel interpretability approach: large language models (LLMs) that generate human-readable narratives from computed archetype paths. In a Titanic dataset case study, we compute archetype paths to reveal how passengers traverse latent space, with LLMs analyzing these paths to produce insightful stories about model behavior. Our results uncover conceptual roles, fairness implications, and decision boundaries, demonstrating the power of combining quantitative analysis with narrative synthesis. Future work will extend this to large language models by sampling top-k activations, offering scalable insights into complex representations.
\end{abstract}

\section{Introduction}
Neural networks excel at pattern recognition, yet their internal representations remain elusive. While interpretability methods like neuron activation visualization or concept direction analysis have advanced, we identify a pervasive issue: \textit{concept fragmentation}, where datapoints of the same class (e.g., "survivors") are scattered across disjoint latent subspaces. This dispersion complicates auditing, obscures fairness, and may reveal meaningful subgroups or biases.

We propose a comprehensive framework to address this: quantitative metrics to measure fragmentation, visualizations to track activation trajectories, and a groundbreaking use of large language models (LLMs) to generate human-readable narratives from computed archetype paths. Unlike prior work, we compute these paths explicitly through clustering and trajectory analysis, with LLMs analyzing the resulting data to produce interpretive stories. Our Titanic dataset case study demonstrates how this approach unveils model logic, fairness concerns, and conceptual roles, paving the way for transparent, equitable AI systems.

\section{Related Work}
\textbf{Concept Activation Vectors (CAVs)} \citep{kim2018} identify linear concept directions but ignore intra-class cohesion. \textbf{Network Dissection} \citep{bau2017} maps neurons to concepts without analyzing datapoint dispersion. \textbf{Polysemantic Neurons} \citep{anthropic2022} explore multi-role units, while we focus on class-level fragmentation. \textbf{Subspace Offsets} \citep{zhang2024} track inter-class drifts, not intra-class scatter. Similarity metrics like \textbf{CKA} \citep{kornblith2019} and \textbf{SVCCA} \citep{raghu2017} assess global layer alignment, missing fine-grained dispersion.

Clustering-based methods \citep{zhou2018} group activations but rarely track trajectories. Our archetype path analysis, where paths are computed and then narrated by LLMs, is novel. Recent LLM applications \citep{bills2023} annotate neuron behavior; we extend this by synthesizing narratives from cluster paths, creating a human-centric interpretability paradigm.

\section{Methodology}

\subsection{Architecture and Activation Tracing}
We analyze fully connected feedforward networks with three layers (input, hidden, output). Activations are collected per layer, reduced via PCA or UMAP for visualization, and clustered using k-means to identify latent subgroups. We compute archetype paths by tracking datapoint transitions across clusters layer by layer, capturing how representations evolve.

\subsection{Fragmentation Metrics}
- \textbf{Cluster Entropy}: Applies k-means clustering to activations and computes normalized entropy of class assignments:
  \[ H(c) = -\sum_{k=1}^{K} p_k(c) \log_2 p_k(c) / \log_2 K, \]
  where $p_k(c)$ is the proportion of class $c$ in cluster $k$. High entropy indicates fragmentation.
- \textbf{Subspace Angle}: Measures pairwise principal angles between class-conditioned PCA subspaces across bootstrap samples, reflecting structural divergence.
- \textbf{Intra-Class Pairwise Distance (ICPD)}: Calculates average Euclidean distance between same-class datapoints, quantifying spatial dispersion.
- \textbf{K-star ($k^*$)}: Determines optimal cluster count per layer using silhouette scores, revealing natural grouping tendencies.

\subsection{LLM-Based Narrative Synthesis}
We compute archetype paths (e.g., cluster transitions like 0→2→0) and their statistics (e.g., demographics, survival rates). These are formatted into structured prompts and passed to an LLM (e.g., GPT-4), which generates narratives describing cluster behavior, fairness implications, and model logic. The LLM does not create paths but analyzes our computed data, ensuring fidelity to the underlying patterns (Appendix~\ref{app:prompts}).

\section{Titanic Case Study}

\subsection{Dataset and Setup}
We train a three-layer feedforward network on the Titanic dataset (891 passengers, predicting survival from age, fare, sex, class). Activations are collected from input, hidden, and output layers, reduced via UMAP, and clustered. We compute archetype paths to track how passengers move through clusters, with LLMs analyzing these paths to produce interpretive narratives.

\subsection{Quantitative Results}
Our metrics reveal fragmentation dynamics:
- \textbf{Cluster Entropy}: Drops from 0.86 (input) to 0.79 (layer 3), indicating increasing cohesion.
- \textbf{Subspace Angles}: Decrease from 45° to 30°, showing subspace alignment.
- \textbf{ICPD}: Rises from 2.1 to 2.8, suggesting subgroup differentiation.
- \textbf{K-star}: Peaks at 6 clusters in layer 1, consolidates to 3 in layer 3.

\subsection{Archetype Path Analysis}
We computed archetype paths by tracking passenger transitions across clusters, revealing dominant trajectories. The top three paths, covering 106 passengers, are:

\begin{itemize}
  \item \textbf{Path 0→2→0 (55 passengers, 6.17\% of dataset)}:
    \begin{itemize}
      \item \textbf{Traits}: 85\% survival, high fare, 45\% first-class, balanced gender.
      \item \textbf{LLM Narrative}: \textit{“These passengers, marked by wealth and privilege, are swiftly identified as survivors. From a diffuse input cluster (0), they sharpen into a cohesive survivor prototype by layer 3 (0), with layer 2 (2) refining their high-fare signature. The network’s confidence is clear—low entropy and tight subspace angles confirm a stable, favored group. Yet, their first-class dominance raises questions: is this prediction or prejudice?”}
      \item \textbf{Insight}: This path reflects a privileged subgroup, consistently prioritized by the model, with 55 passengers (indices 0, 3, 9, etc.) following a stable trajectory.
    \end{itemize}
  \item \textbf{Path 1→1→1 (29 passengers, 3.25\% of dataset)}:
    \begin{itemize}
      \item \textbf{Traits}: 37\% survival, young, male, 80\% third-class.
      \item \textbf{LLM Narrative}: \textit{“Trapped in a static cluster, these young third-class men are dismissed from the start. Across all layers, they remain in cluster 1—a fragmented, low-survival limbo. High ICPD and persistent entropy suggest the network barely reconsiders them. Their story is one of neglect, echoing historical biases where class and gender dictated fate.”}
      \item \textbf{Insight}: This path, followed by 29 passengers (indices 7, 8, 27, etc.), highlights a fairness concern, with the model marginalizing a disadvantaged group.
    \end{itemize}
  \item \textbf{Path 2→0→1 (22 passengers, 2.47\% of dataset)}:
    \begin{itemize}
      \item \textbf{Traits}: 52\% survival, moderate fare, middle-aged, slightly male-heavy.
      \item \textbf{LLM Narrative}: \textit{“These passengers begin with promise, aligned with survivors in cluster 2. But layer 2 shifts them to cluster 0, and by layer 3, they’re demoted to an ambiguous cluster 1. Their moderate traits—neither rich nor poor—confuse the network, reflected in rising ICPD. This trajectory reveals the cost of fragmentation: edge cases slip through, their fates muddled.”}
      \item \textbf{Insight}: This path, followed by 22 passengers (indices 2, 10, 13, etc.), exposes the model’s indecision, fragmenting a mixed group.
    \end{itemize}
\end{itemize}

These paths were computed using our trajectory analysis pipeline, not generated by the LLM. The LLM’s role was to analyze the path data (e.g., cluster assignments, demographics) and produce narratives, which we validated against ground-truth statistics. The paths cover 106 passengers, with 0→2→0 dominating (55 passengers), suggesting a strong survivor archetype, while 1→1→1 and 2→0→1 reveal marginalized or ambiguous groups.

\subsection{Fairness Implications}
The archetype paths highlight fairness concerns. Third-class passengers (60\% of the dataset) are overrepresented in fragmented, low-survival paths like 1→1→1 (80\% third-class), while first-class passengers dominate cohesive survivor paths like 0→2→0 (45\% first-class). Demographic parity analysis shows a 20\% gap in cluster assignment fairness between classes, indicating bias amplification. These findings, supported by LLM narratives, underscore how fragmentation can expose inequities, informing interventions like regularization to reduce unfair splits.

\subsection{Visualizations}
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.8\textwidth]{optimal_clusters.png}
  \caption{Optimal number of clusters ($k^*$) by layer, peaking at 6 in layer 1 before consolidating to 3 in layer 3.}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.8\textwidth]{intra_class_distance.png}
  \caption{Intra-class pairwise distance (ICPD) by layer, showing increasing divergence within classes.}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.8\textwidth]{subspace_angle.png}
  \caption{Subspace angles by layer, decreasing as class-conditioned subspaces align.}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.8\textwidth]{cluster_entropy.png}
  \caption{Cluster entropy by layer, declining as the network consolidates representations.}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{trajectory_basic.png}
  \caption{UMAP projection of activation trajectories, showing distinct paths for survivors and non-survivors.}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{trajectory_annotated.png}
  \caption{UMAP trajectories annotated with LLM-derived labels, highlighting archetype paths like 0→2→0.}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=0.9\textwidth]{trajectory_by_endpoint_cluster.png}
  \caption{Layer 3 clustering, showing convergence into three dominant archetypes.}
\end{figure}

\section{Future Directions: Extending to Large Language Models}
Our framework, while validated on feedforward networks, holds immense potential for large language models (LLMs). LLMs encode complex semantic relationships in high-dimensional spaces, where fragmentation may obscure trust, safety, and fairness. Extending our approach could involve sampling top-k activations and clustering them across diverse text inputs (e.g., varying topics or styles) to reveal how LLMs organize concepts like sentiment or bias. This scalable method could uncover latent representation patterns in billion-parameter models.

\textbf{Adapted Metrics}: Cluster entropy could measure dispersion in token embeddings, subspace angles could analyze attention head divergence, and ICPD could assess semantic cluster consistency (e.g., synonyms). Dimensionality reduction (e.g., UMAP) would manage scale.

\textbf{Self-Interpretation}: LLMs could narrate their own cluster paths, producing stories about processing sensitive topics (e.g., hate speech). This recursive interpretability could enable real-time auditing, though preventing hallucination is key.

\textbf{Fairness Applications}: Fragmentation metrics could detect biased representations of demographic attributes, with narratives translating findings into actionable insights for regulators and developers.

\section{Conclusion}
We present a framework to measure and interpret concept fragmentation, combining metrics, trajectory visualizations, and LLM-driven narratives. Our Titanic case study, with computed archetype paths analyzed by an LLM, reveals model logic, fairness issues, and conceptual roles. By clarifying that paths are derived from our analysis, not LLM-generated, we highlight the synergy of quantitative and narrative methods. Future extensions to LLMs promise transformative insights into complex AI systems.

\section*{Acknowledgments}
[Redacted for anonymity.]

\bibliographystyle{iclr2025_conference}
\bibliography{references}

\appendix

\section{LLM Prompt for Narrative Synthesis}
\label{app:prompts}
\small
\begin{verbatim}
You are an interpretability analyst narrating a neural network’s processing of Titanic passenger data. You are given computed archetype paths (e.g., cluster transitions 0->2->0) and statistics (e.g., survival rate, demographics). Your task:
1. Describe each path’s journey through the network’s latent space.
2. Highlight passenger traits (age, fare, class) and their implications.
3. Identify fairness concerns or biases in cluster assignments.
4. Craft a human-readable story reflecting the network’s logic.
Use vivid, precise language, grounding narratives in the provided data. Avoid generating paths; analyze only the given trajectories.
\end{verbatim}

\end{document}
