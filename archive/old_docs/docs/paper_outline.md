# Concept Fragmentation in Neural Networks: Paper Outline

## Title
"Concept Fragmentation in Neural Networks: Visualizing and Measuring Intra-Class Dispersion in Feedforward Models"

## Abstract
- Introduce concept fragmentation (datapoints of same class scattered across disjoint regions)
- Outline our novel framework: metrics + trajectory visualization + LLM narratives
- Highlight key findings from Titanic case study
- Indicate future applications to larger models, including LLMs

## 1. Introduction
- **Problem Statement**: Neural network interpretability hampered by concept fragmentation
- **Definition**: Concept fragmentation occurs when datapoints of the same class are scattered across disjoint regions in latent space
- **Consequences**: Complicates interpretability, obscures fairness issues, and may reveal meaningful subgroups
- **Our Approach**: Comprehensive framework combining:
  - Quantitative metrics (cluster entropy, subspace angles, etc.)
  - Trajectory visualizations to track activation patterns
  - Cross-layer metrics for analyzing relationships between representations
  - LLM-based narrative synthesis from computed archetypal paths
- **Contributions**: Highlight innovations in measurement, visualization, and LLM-based interpretability

## 2. Related Work
- **Neural Network Interpretability**:
  - Feature attribution methods (LIME, SHAP, IG)
  - Concept-based approaches (CAVs, network dissection)
  - Activation visualization techniques
- **Clustering-Based Analysis**:
  - Prior work on activation clustering
  - Limitations of existing approaches
- **LLMs for Explanation**:
  - Recent use of LLMs in model explanation
  - Contrast with our approach (LLMs analyze computed paths rather than generating them)
- **Cross-Layer Analysis**:
  - Prior work on relationships between layer representations
  - Similarity metrics for neural representations (CKA, SVCCA)

## 3. Methodology: Concept Fragmentation Framework
- **Metrics**:
  - **Cluster Entropy**: Formulation, interpretation, relationship to fragmentation
  - **Subspace Angle**: Measuring principal angles between class subspaces
  - **Intra-Class Pairwise Distance (ICPD)**: Capturing spatial dispersion
  - **K-star (k*)**: Optimal cluster count analysis
- **Visualization Approach**:
  - Dimensionality reduction techniques (UMAP, PCA)
  - Trajectory plotting across layers
  - Color-coding and highlighting strategies

## 4. Methodology: Cross-Layer Analysis
- **Centroid Similarity (ρᶜ)**:
  - Mathematical formulation
  - Implementation details
  - Interpretation guidelines
- **Membership Overlap (J)**:
  - Jaccard similarity and containment metrics
  - Measuring cohesion across layers
- **Trajectory Fragmentation (F)**:
  - Entropy-based measures for path consistency
  - Class-based fragmentation measures
- **Inter-Cluster Path Density (ICPD)**:
  - Multi-step transition analysis
  - Return paths and convergence patterns

## 5. Methodology: Archetypal Path Analysis
- **Path Computation**:
  - Tracking datapoints through layer-specific clusters
  - Identifying dominant paths and archetype patterns
- **Explainable Threshold Similarity (ETS)**:
  - Dimension-wise thresholds for transparent clustering
  - Advantages over centroid-based methods
- **Transition Analysis**:
  - Matrix representation of cluster transitions
  - Entropy and sparsity metrics
- **Path Archetypes**:
  - Computing demographic statistics for paths
  - Identifying characteristic patterns

## 6. Methodology: LLM-Based Narrative Generation
- **System Architecture**:
  - Multi-provider integration (Grok, Claude, OpenAI, Gemini)
  - Prompt engineering for interpretability
- **Cluster Labeling**:
  - Generating human-readable labels from centroids
  - Enhancing cluster understanding
- **Path Narratives**:
  - Creating coherent stories from path statistics
  - Incorporating demographic and fragmentation information
- **Quality Assurance**:
  - Verification against ground truth
  - Prompt optimization for accuracy

## 7. Experimental Setup
- **Datasets**:
  - Titanic passenger survival
  - Heart disease prediction
- **Model Architecture**:
  - 3-layer feedforward network design
  - Activation functions and layer dimensions
- **Regularization Approaches**:
  - Baseline (no regularization)
  - Cohesion regularization variations
- **Evaluation Methodology**:
  - Metrics computation
  - Statistical significance testing
  - LLM evaluation protocol

## 8. Results: Titanic Case Study
- **Quantitative Metrics**:
  - Cluster entropy, subspace angle, ICPD, and k* across layers
  - Comparison between baseline and regularized models
- **Trajectory Visualization**:
  - 3D UMAP embedding analysis
  - Layer-wise progression and key patterns
- **Archetype Path Analysis**:
  - Top paths (0→2→0, 1→1→1, 2→0→1)
  - Demographic characteristics and survival rates
- **LLM-Generated Narratives**:
  - Analysis of key paths and their implications
  - Fairness insights and demographic patterns

## 9. Results: Regularization Effects
- **Entropy Reduction**:
  - Impact of cohesion regularization on fragmentation
  - Statistical significance of improvements
- **Subspace Alignment**:
  - Changes in angle metrics with regularization
  - Mixed effects and potential explanations
- **Accuracy Impact**:
  - Effect on model performance
  - Balancing cohesion and generalization

## 10. Discussion
- **Key Findings**:
  - Fragmentation increases in deeper layers
  - Regularization can reduce fragmentation without hurting performance
  - Paths reveal meaningful subgroups and potential biases
- **Implications for Interpretability**:
  - How reducing fragmentation enhances understanding
  - Role of narratives in communicating model behavior
- **Fairness Considerations**:
  - Fragmentation's relationship to bias
  - Using paths to identify inequitable treatment

## 11. Future Work
- **Extending to Large Language Models**:
  - Adapting metrics for billion-parameter models
  - Sampling strategies for efficiency
  - Self-interpretation possibilities
- **Enhanced Regularization**:
  - Combined approaches targeting both entropy and angle
  - Layer-specific regularization strategies
- **Interactive Tools**:
  - Expanding the dashboard for real-time exploration
  - Integration with model development workflows

## 12. Conclusion
- Recap of contributions
- Importance of addressing concept fragmentation
- Vision for more interpretable and fair neural networks

## Acknowledgments

## References

## Appendices
- **A. Mathematical Formulations**
- **B. LLM Prompt Examples**
- **C. Additional Visualizations**