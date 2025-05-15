Title: Concept Fragmentation in Neural Networks: Visualizing and Measuring Intra-Class Dispersion in Feedforward Models

Abstract:
Neural networks learn internal representations that ideally encode human-interpretable concepts cohesively. However, we observe that representations of a single concept may fragment across disjoint subspaces, a phenomenon we call concept fragmentation. We define this as the spatial dispersion of datapoints sharing a semantic label into multiple, disconnected regions in latent space. This paper introduces visualization and metric-based techniques to detect, quantify, and analyze concept fragmentation in feedforward neural networks. Using datasets such as Titanic, Adult Income, and Fashion MNIST, we examine whether fragmentation correlates with model misclassifications, subgroup separation, and training dynamics. [Placeholder: Insert quantitative summary result here.] Fragmentation is quantified via (i) intra-class clustering entropy and (ii) pairwise subspace angles between class-conditioned principal components. Concept fragmentation is presented as an interpretability phenomenon—a structural property that may be problematic in some cases (e.g., audit obfuscation, fairness gaps) but benign or even informative in others (e.g., valid subpopulation distinction).

1. Introduction

Neural networks are known to form complex internal representations of input data. While these representations enable high performance, they are often opaque. One desirable property is conceptual coherence—that instances of the same concept (e.g., "female" or "survivor") cluster together in the latent space. However, networks often split these instances across separate regions. We term this phenomenon concept fragmentation, defined as the spatial dispersion of a shared conceptual label across disjoint subregions of a neural representation space.

We present concept fragmentation as an interpretability phenomenon—a pattern of internal organization that can obscure transparency and auditing, especially when related to fairness or reliability, but which may also reflect true substructure in the data. Fragmentation may occur when a network fails to unify semantically related inputs, often due to conflicting correlations or entangled features. This paper contributes:

A formal definition and metric for measuring concept fragmentation

A method for visualizing representation trajectories layer by layer

Empirical evidence across tabular and visual datasets

A proposed regularization strategy to mitigate fragmentation

Discussion of when fragmentation is problematic vs. neutral

2. Related Work

Polysemantic Neurons (Anthropic): Neurons encoding multiple meanings; fragmentation may be a complementary issue across datapoints.

Concept Activation Vectors (CAVs): Directional metrics of concept representation; our focus is on spatial cohesion instead.

Network Dissection: Identifies interpretable units; we analyze structure across datapoints, not per unit.

Subspace alignment (Heinze-Deml et al.): Cross-model or cross-class comparison; our focus is on intra-class dispersion.

Centered Kernel Alignment (CKA) and SVCCA: Global similarity metrics; we analyze spatial fragmentation within a class.

Subspace Offsets (Zhang & Morcos, 2024): Measures class drift; our use of subspace angles operates within a class.

Disentanglement and Feature Entanglement: Focus on feature separability; we focus on concept-level spatial fragmentation.

Class Separability: Inter-class distinction; our concern is intra-class scattering.

3. Methodology

3.1 Architecture and Notation
We evaluate networks of varying sizes, ranging from small models with 3 neurons per hidden layer to larger feedforward networks with greater width and depth. The input layer is treated as Layer 0, and activations from all subsequent layers are included in our analysis. All models are trained with fixed seeds and identical initialization for consistency.

3.2 Activation Tracing
For networks with very few neurons per layer (e.g., 3), we manually trace full activation trajectories for each datapoint across layers. For larger networks, we apply dimensionality reduction using PCA to embed activations into a low-dimensional trajectory space. This allows us to capture representational paths without requiring sparsity-based filtering or selection of top activations.

3.3 Fragmentation Metrics

Cluster Entropy: For each layer we fit *k*-means **once** on the complete set of activations, obtaining global clusters.  The number of clusters $K$ is **selected automatically** via the silhouette criterion (Rousseeuw, 1987): we search $K\in\{2,\dots,K_{\max}\}$ (with $K_{\max}=\min(12,\lceil\sqrt{N}\rceil)$) and pick the value that maximises the mean silhouette score.  For each class $c$ we then compute the distribution $\mathbf{p}(c)=[p_1(c),\dots,p_K(c)]$ where $p_k(c)$ is the fraction of $c$’s samples assigned to cluster $k$.  The fragmentation score is the normalised entropy
\[
\hat H(c)= -\sum_{k=1}^K p_k(c)\,\log_2 p_k(c) / \log_2 K\in[0,1].
\]
A high value indicates that the class is scattered through many global clusters; a low value indicates cohesion.

Subspace Angle: For each class, compute the first  principal components (explaining 90% variance) and calculate pairwise principal angles  between class-conditioned subspaces:



Where  and  are subspace bases from different bootstrap samples ( total) of class .

3.4 Visualization
We use PCA and UMAP to reduce high-dimensional activation vectors into 2D or 3D spaces for visualization. This dimensionality reduction enables us to extend fragmentation analysis beyond small 3×3×3 networks to arbitrarily wide or deep models. Trajectories are color-coded by class and optionally by sub-cluster. Our interactive Dash visualization tool displays the same k-means cluster centers that are used for the fragmentation metrics, providing a cohesive view between the quantitative metrics and qualitative exploration. All visualizations use deterministic seeds to ensure reproducibility.

3.5 Cohesion Regularization
We experiment with a simple contrastive loss term that pulls latent activations of same-class datapoints closer:



This term is only applied when a minimum threshold of same-class pairs is met within the minibatch to avoid instability. We also test hyperparameter sensitivity for , explained variance thresholds, and clustering methods.

4. Experiments

Datasets:

Titanic: Survival prediction. Fragmentation expected due to feature interactions (e.g., gender × class).

Adult Income: Predict >50K income. Labels span multiple demographic groups with different feature patterns.

Heart Disease: Binary medical outcome. Observe whether diagnostic criteria fragment across patient subtypes. Class imbalance addressed by stratified subsampling.

Fashion MNIST (shirt vs. pullover): Binary subset to isolate visual fragmentation.

Setup:

Standard preprocessing (e.g., one-hot encoding, normalization)

Fully connected networks of varying size (from 3 neurons per layer to deeper, wider architectures)

Track fragmentation metrics across layers and training epochs

Compare baseline model with and without cohesion regularization

Fragmentation measured separately within protected-attribute strata to detect intra-group splits

5. Results and Discussion

5.1 Visualization Results
[Placeholder: Describe observed trajectory patterns and any visible fragmentation clusters. Include examples of both problematic and benign fragmentation.]

5.2 Metric Results
[Placeholder: Report fragmentation metric trends, correlation coefficients, and statistical significance. Include 95% CIs via bootstrapping and permutation tests for significance.]

5.3 Ablations
[Placeholder: Compare performance and fragmentation with/without regularization. Include sensitivity tests from empirical datasets.]

5.4 Training Dynamics
[Placeholder: Insert plots of fragmentation vs. training epoch and discuss observed trends.]

6. Implications and Future Work

Auditing and Fairness: Fragmentation may obscure auditing when subclusters correlate with protected attributes

Robustness: Fragmented regions may lie closer to decision boundaries or exhibit greater adversarial sensitivity

Regularization: Penalizing fragmentation may improve generalization, especially in overparameterized models

Neutral Fragmentation: Fragmentation may also reflect valid subgroup structure (e.g., education level in Adult Income)

Transferability: Fragmentation may explain failure modes in domain adaptation or fine-tuning

Extension to Transformers: Fragmentation may appear in token embeddings or attention heads; future work will apply metrics to small transformer models on text classification tasks

7. Conclusion

Concept fragmentation is a tractable, measurable, and insightful interpretability phenomenon. We provide a framework to visualize and quantify it, showing that even simple networks—and especially larger ones—can learn disjoint representations of the same concept. Understanding when fragmentation is harmful, neutral, or informative is a promising direction for improving model transparency, auditing, and generalization.

Appendix

Fragmentation metric code

Dataset statistics and preprocessing steps

Full-resolution visualizations

Reproducibility checklist (model seeds, training configs, plotting scripts)

Dataset licenses: Titanic (public domain), Adult Income (UCI ML repository), Heart Disease (UCI ML repository), Fashion MNIST (MIT license)

