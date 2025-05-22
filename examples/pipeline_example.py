"""
Example of using the pipeline architecture for neural network analysis.

This example demonstrates how to build and use pipelines for analyzing neural
networks, including activation collection, clustering, path tracking, and
visualization preparation.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import argparse
import os

from concept_fragmentation.activation import (
    ActivationCollector, ActivationProcessor, ActivationStorage,
    ActivationFormat, CollectionConfig
)

from concept_fragmentation.pipeline import (
    Pipeline, PipelineConfig, StreamingMode,
    ActivationCollectionStage, ActivationProcessingStage,
    ClusteringStage, ClusterPathStage, PathArchetypeStage,
    PersistenceStage, LLMAnalysisStage
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Create a simple model for demonstration
class SimpleModel(nn.Module):
    """Simple MLP model for demonstration."""
    
    def __init__(self, input_dim=10, hidden_dims=[128, 64, 32], output_dim=2):
        super().__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i+1])
            for i in range(len(hidden_dims)-1)
        ])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        return self.output_layer(x)


def generate_synthetic_data(n_samples=100, input_dim=10, binary=True, seed=42):
    """Generate synthetic data for demonstration."""
    np.random.seed(seed)
    
    X = np.random.randn(n_samples, input_dim)
    
    # Generate target values based on some non-linear function of inputs
    if binary:
        logits = np.sum(X[:, :3], axis=1) - np.sum(X[:, 3:6], axis=1)
        y = (logits > 0).astype(int)
    else:
        y = np.sin(np.sum(X[:, :3], axis=1)) + np.cos(np.sum(X[:, 3:6], axis=1))
    
    # Create a dataframe with some demographic features
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(input_dim)])
    df['target'] = y
    
    # Add synthetic demographic features
    df['age'] = np.random.randint(18, 80, size=n_samples)
    df['income'] = 30000 + 1000 * df['age'] + np.random.randn(n_samples) * 10000
    df['gender'] = np.random.choice(['male', 'female'], size=n_samples)
    df['education'] = np.random.choice(
        ['high_school', 'bachelors', 'masters', 'phd'],
        size=n_samples,
        p=[0.3, 0.4, 0.2, 0.1]
    )
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), df


def build_standard_pipeline(
    model, 
    output_dir: str,
    llm_provider: str = None,
    streaming: bool = False,
    compute_archetypes: bool = True,
    demographic_columns: List[str] = None
):
    """Build a standard pipeline for neural network analysis."""
    # Configure pipeline
    config = PipelineConfig(
        streaming_mode=StreamingMode.FORCE if streaming else StreamingMode.DISABLE,
        use_context=True,
        log_progress=True
    )
    
    # Create pipeline
    pipeline = Pipeline(config)
    
    # Add activation collection stage
    collector_stage = ActivationCollectionStage(
        streaming=streaming,
        store_to_disk=False,
        name="ActivationCollection"
    )
    pipeline.add_stage(collector_stage)
    
    # Add activation processing stage
    processor_stage = ActivationProcessingStage(
        dimensionality_reduction=True,
        n_components=10,
        normalization=True,
        normalization_method='standard',
        name="ActivationProcessing"
    )
    pipeline.add_stage(processor_stage)
    
    # Add clustering stage
    clustering_stage = ClusteringStage(
        max_k=10,
        compute_silhouette=True,
        name="Clustering"
    )
    pipeline.add_stage(clustering_stage)
    
    # Add cluster path stage
    path_stage = ClusterPathStage(
        compute_similarity=True,
        name="ClusterPaths"
    )
    pipeline.add_stage(path_stage)
    
    # Add archetype stage if requested
    if compute_archetypes:
        archetype_stage = PathArchetypeStage(
            target_column='target',
            demographic_columns=demographic_columns or ['age', 'gender', 'income', 'education'],
            top_k=5,
            name="PathArchetypes"
        )
        pipeline.add_stage(archetype_stage)
    
    # Add LLM analysis stage if provider specified
    if llm_provider:
        llm_stage = LLMAnalysisStage(
            provider=llm_provider,
            label_clusters=True,
            generate_narratives=True,
            top_k_paths=3,
            name="LLMAnalysis"
        )
        pipeline.add_stage(llm_stage)
    
    # Add persistence stage
    persistence_stage = PersistenceStage(
        output_path=os.path.join(output_dir, "analysis_results.json"),
        format='json',
        save_activations=False,
        name="Persistence"
    )
    pipeline.add_stage(persistence_stage)
    
    return pipeline


def run_pipeline_example(args):
    """Run the pipeline example."""
    # Generate synthetic data
    logger.info("Generating synthetic data...")
    X, y, df = generate_synthetic_data(
        n_samples=args.samples,
        input_dim=args.input_dim,
        binary=args.binary,
        seed=args.seed
    )
    
    # Create model
    logger.info("Creating model...")
    model = SimpleModel(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=1 if args.binary else 1
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build pipeline
    logger.info("Building pipeline...")
    pipeline = build_standard_pipeline(
        model,
        output_dir=args.output_dir,
        llm_provider=args.llm_provider,
        streaming=args.streaming,
        compute_archetypes=not args.skip_archetypes,
        demographic_columns=['age', 'gender', 'income', 'education']
    )
    
    # Validate pipeline
    logger.info("Validating pipeline...")
    validation_errors = pipeline.validate()
    if validation_errors:
        for error in validation_errors:
            logger.warning(f"Validation error: {error}")
    
    # Draw pipeline diagram if graphviz is available
    try:
        diagram_path = os.path.join(args.output_dir, "pipeline_diagram")
        pipeline.plot(diagram_path)
        logger.info(f"Pipeline diagram saved to {diagram_path}.pdf")
    except Exception as e:
        logger.warning(f"Could not generate pipeline diagram: {e}")
    
    # Prepare input data for the pipeline
    pipeline_input = {
        'model': model,
        'inputs': X,
        'labels': y,
        'df': df,
        'dataset_name': 'synthetic',
        'metadata': {
            'samples': args.samples,
            'input_dim': args.input_dim,
            'hidden_dims': args.hidden_dims,
            'binary': args.binary,
            'seed': args.seed
        }
    }
    
    # Execute pipeline
    logger.info("Executing pipeline...")
    result = pipeline.execute(pipeline_input)
    
    # Display results summary
    logger.info("Pipeline execution completed.")
    
    if 'saved_path' in result:
        logger.info(f"Results saved to: {result['saved_path']}")
    
    # Display some statistics
    if 'layer_clusters' in result:
        logger.info(f"Clusters computed for {len(result['layer_clusters'])} layers")
    
    if 'paths' in result:
        logger.info(f"Computed paths for {len(result['paths'].get('unique_paths', []))} samples")
    
    if 'path_archetypes' in result:
        logger.info(f"Computed {len(result['path_archetypes'])} path archetypes")
    
    if 'cluster_labels' in result:
        logger.info(f"Generated {len(result['cluster_labels'])} cluster labels")
    
    if 'path_narratives' in result:
        logger.info(f"Generated {len(result['path_narratives'])} path narratives")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Architecture Example")
    
    # Data and model parameters
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--input-dim", type=int, default=10, help="Input dimension")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64, 32], help="Hidden dimensions")
    parser.add_argument("--binary", action="store_true", help="Generate binary classification data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Pipeline parameters
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    parser.add_argument("--skip-archetypes", action="store_true", help="Skip computing archetypes")
    parser.add_argument("--llm-provider", type=str, help="LLM provider to use (e.g., openai, claude)")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="pipeline_example_output", help="Output directory")
    
    args = parser.parse_args()
    run_pipeline_example(args)