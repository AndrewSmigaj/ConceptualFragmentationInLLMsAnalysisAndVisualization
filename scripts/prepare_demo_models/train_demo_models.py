#!/usr/bin/env python3
"""
Main script to train demo models for Concept MRI.
Supports various model variants including optimal (via hyperparameter search),
bottleneck, overfit, underfit, and custom configurations.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, Any
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_trainer import ModelTrainer
from model_architectures import create_model
from model_presets import get_preset_config, validate_config
from hyperparameter_search import HyperparameterOptimizer


def setup_logging(verbose: bool = True):
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def train_optimal_model(dataset: str, n_trials: int = 50, verbose: bool = True) -> Dict[str, Any]:
    """
    Train optimal model using hyperparameter search.
    
    Args:
        dataset: Dataset name
        n_trials: Number of optimization trials
        verbose: Whether to print progress
        
    Returns:
        Training results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting hyperparameter optimization for {dataset}")
    
    # Run optimization
    optimizer = HyperparameterOptimizer(
        dataset_name=dataset,
        n_trials=n_trials,
        metric='val_acc'
    )
    
    optimization_results = optimizer.optimize()
    best_params = optimization_results['best_params']
    
    # Train final model with best parameters
    logger.info("Training final model with best parameters...")
    trainer = ModelTrainer(dataset)
    
    # Create model with best architecture
    model_config = {
        'hidden_sizes': best_params.get('hidden_sizes', [64, 32]),
        'activation': best_params.get('activation', 'relu'),
        'dropout': best_params.get('dropout', 0.0),
        'batch_norm': best_params.get('batch_norm', False)
    }
    
    model = create_model(
        model_type='standard',
        input_size=trainer.input_size,
        output_size=trainer.output_size,
        config=model_config
    )
    
    # Train with best parameters
    results = trainer.train_model(model, best_params, verbose=verbose)
    
    # Save model
    trainer.save_model_for_concept_mri(model, results, best_params, 'optimal')
    
    # Add optimization info to results
    results['optimization_info'] = optimization_results
    
    return results


def train_preset_model(dataset: str, variant: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Train a model with preset configuration.
    
    Args:
        dataset: Dataset name
        variant: Model variant name
        verbose: Whether to print progress
        
    Returns:
        Training results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Training {variant} model for {dataset}")
    
    # Get preset configuration
    config = get_preset_config(dataset, variant)
    if config is None:
        raise ValueError(f"No preset config for variant '{variant}'")
    
    # Create trainer
    trainer = ModelTrainer(dataset)
    
    # Create model based on config
    model_type = config.get('model_type', 'standard')
    model = create_model(
        model_type=model_type,
        input_size=trainer.input_size,
        output_size=trainer.output_size,
        config=config
    )
    
    # Train model
    results = trainer.train_model(model, config, verbose=verbose)
    
    # Save model
    trainer.save_model_for_concept_mri(model, results, config, variant)
    
    return results


def train_custom_model(dataset: str, config_path: str, output_name: str, 
                      verbose: bool = True) -> Dict[str, Any]:
    """
    Train a model with custom configuration.
    
    Args:
        dataset: Dataset name
        config_path: Path to configuration JSON file
        output_name: Name for output files
        verbose: Whether to print progress
        
    Returns:
        Training results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Training custom model for {dataset} with config {config_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate configuration
    validate_config(config)
    
    # Create trainer
    trainer = ModelTrainer(dataset)
    
    # Create model
    model_type = config.get('model_type', 'standard')
    model = create_model(
        model_type=model_type,
        input_size=trainer.input_size,
        output_size=trainer.output_size,
        config=config
    )
    
    # Train model
    results = trainer.train_model(model, config, verbose=verbose)
    
    # Save model
    trainer.save_model_for_concept_mri(model, results, config, output_name)
    
    return results


def train_all_variants(dataset: str, verbose: bool = True):
    """
    Train all preset variants for a dataset.
    
    Args:
        dataset: Dataset name
        verbose: Whether to print progress
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Training all variants for {dataset}")
    
    # Get all variants from presets
    from model_presets import PRESET_CONFIGS
    
    if dataset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    variants = list(PRESET_CONFIGS[dataset].keys())
    
    results_summary = {}
    
    for variant in variants:
        try:
            if variant == 'optimal':
                # Run hyperparameter search for optimal
                logger.info("Training optimal model with hyperparameter search...")
                results = train_optimal_model(dataset, n_trials=30, verbose=verbose)
            else:
                # Use preset config
                results = train_preset_model(dataset, variant, verbose=verbose)
            
            results_summary[variant] = {
                'test_accuracy': results['final_test_acc'],
                'epochs_trained': results['epochs_trained']
            }
            
            logger.info(f"Completed {variant}: test_acc={results['final_test_acc']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to train {variant}: {e}")
            results_summary[variant] = {'error': str(e)}
    
    # Save summary
    summary_path = Path(__file__).parent.parent.parent / 'concept_mri' / 'demos' / dataset / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Training complete! Summary saved to {summary_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train feedforward neural networks for Concept MRI demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train optimal model using hyperparameter search
  python train_demo_models.py --dataset titanic --variant optimal
  
  # Train specific variant
  python train_demo_models.py --dataset heart_disease --variant bottleneck
  
  # Train with custom configuration
  python train_demo_models.py --dataset titanic --variant custom --config my_config.json
  
  # Train all variants
  python train_demo_models.py --dataset titanic --all
        """
    )
    
    parser.add_argument(
        '--dataset',
        required=True,
        choices=['titanic', 'heart_disease'],
        help='Dataset to use for training'
    )
    
    parser.add_argument(
        '--variant',
        choices=['optimal', 'bottleneck', 'overfit', 'underfit', 'unstable', 
                'fragmented', 'regularized', 'multipath', 'sparse', 'custom'],
        help='Model variant to train'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Train all variants for the dataset'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration JSON (required for custom variant)'
    )
    
    parser.add_argument(
        '--output-name',
        type=str,
        default='custom',
        help='Output name for custom models'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of trials for hyperparameter optimization (optimal variant)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=not args.quiet)
    
    # Validate arguments
    if not args.all and not args.variant:
        parser.error("Either --variant or --all must be specified")
    
    if args.variant == 'custom' and not args.config:
        parser.error("--config is required when using custom variant")
    
    # Log device info
    logger.info(f"Using device: {args.device}")
    if args.device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        if args.all:
            # Train all variants
            train_all_variants(args.dataset, verbose=not args.quiet)
        elif args.variant == 'optimal':
            # Train optimal model
            train_optimal_model(args.dataset, n_trials=args.n_trials, 
                              verbose=not args.quiet)
        elif args.variant == 'custom':
            # Train custom model
            train_custom_model(args.dataset, args.config, args.output_name,
                             verbose=not args.quiet)
        else:
            # Train preset variant
            train_preset_model(args.dataset, args.variant, verbose=not args.quiet)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()