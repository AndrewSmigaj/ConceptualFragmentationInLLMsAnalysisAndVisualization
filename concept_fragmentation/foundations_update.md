## 4.2 Cross-Layer Path Analysis Metrics

To rigorously analyze the semantic meaning and evolution of concepts across layers, we introduce the following cross-layer metrics that quantify relationships between clusters at different depths of the network.

### 4.2.1 Centroid Similarity (ρᶜ)

Centroid similarity measures whether clusters with the same or different IDs across layers represent similar concepts in the embedding space:

$$\rho^c(C_i^l, C_j^{l'}) = \text{sim}(\mu_i^l, \mu_j^{l'})$$

where $\mu_i^l$ is the centroid of cluster $i$ in layer $l$, and sim is a similarity function (cosine similarity or normalized Euclidean distance). For meaningful comparison across layers with different dimensionalities, we project clusters into a shared embedding space using techniques such as PCA or network-specific dimensionality reduction.

High centroid similarity between clusters in non-adjacent layers (e.g., $C_0^{l_1}$ and $C_0^{l_3}$) indicates conceptual "return" or persistence, even when intermediate layers show different cluster assignments. This helps identify stable semantic concepts that temporarily fragment but later reconverge.

### 4.2.2 Membership Overlap (J)

Membership overlap quantifies how data points from a cluster in one layer are distributed across clusters in another layer:

$$J(C_i^l, C_j^{l'}) = \frac{|D_i^l \cap D_j^{l'}|}{|D_i^l \cup D_j^{l'}|}$$

where $D_i^l$ is the set of data points in cluster $i$ at layer $l$. This Jaccard similarity metric reveals how cohesive groups of data points remain across network depth. A related metric is the membership containment ratio:

$$\text{contain}(C_i^l \rightarrow C_j^{l'}) = \frac{|D_i^l \cap D_j^{l'}|}{|D_i^l|}$$

Which measures what proportion of points from an earlier cluster appear in a later cluster. High containment suggests the second cluster has "absorbed" the concept represented by the first.

### 4.2.3 Trajectory Fragmentation Score (F)

Trajectory fragmentation quantifies how consistently a model processes semantically similar inputs:

$$F(p) = H(\{t_{i,j} | (i,j) \in p\})$$

where $p$ is a path, $t_{i,j}$ is the transition from cluster $i$ to cluster $j$, and $H$ is an entropy or variance measure. Low fragmentation scores indicate stable, predictable paths through the network layers. We also compute fragmentation relative to class labels:

$$F_c(y) = \frac{|\{p | \exists x_i, x_j \in X_y \text{ where } \text{path}(x_i) \neq \text{path}(x_j)\}|}{|X_y|}$$

This measures the proportion of samples with the same class label $y$ that follow different paths, directly quantifying concept fragmentation within a semantic category.

### 4.2.4 Inter-Cluster Path Density (ICPD)

ICPD analyzes higher-order patterns in concept flow by examining multi-step transitions:

$$\text{ICPD}(C_i^l \rightarrow C_j^{l'} \rightarrow C_k^{l''}) = \frac{|\{x | \text{cluster}^l(x) = i \wedge \text{cluster}^{l'}(x) = j \wedge \text{cluster}^{l''}(x) = k\}|}{|\{x | \text{cluster}^l(x) = i\}|}$$

This metric identifies common patterns like:
- Return paths ($i \rightarrow j \rightarrow i$): The model temporarily assigns inputs to an intermediate concept before returning them to the original concept
- Similar-destination paths ($i \rightarrow j \rightarrow k$ where $\rho^c(C_i^l, C_k^{l''})$ is high): The model reaches a conceptually similar endpoint through an intermediate step

These patterns reveal how the network refines its representations across layers. We visualize the most frequent paths using weighted directed graphs, where the weight of each edge represents the transition frequency.

### 4.2.5 Application to Interpretability

These cross-layer metrics provide a quantitative foundation for analyzing how concepts evolve through a neural network. By combining them with the ETS clustering approach, we can generate interpretable explanations of model behavior, such as:

"Inputs in this category initially cluster together (low $F_c$), then separate based on feature X (high membership divergence at layer 2), before reconverging in the output layer (high centroid similarity between first and last layer clusters)."

This statistical foundation ensures that our interpretations are not merely post-hoc narratives but are grounded in measurable properties of the network's internal representations.