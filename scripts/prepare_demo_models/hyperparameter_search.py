"""
Hyperparameter optimization using Optuna.
"""
import optuna
from optuna import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import logging

from model_trainer import ModelTrainer
from model_architectures import create_model
from model_presets import get_search_space


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for neural networks using Optuna.
    """
    
    def __init__(
        self,
        dataset_name: str,
        n_trials: int = 50,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        direction: str = 'maximize',
        metric: str = 'val_acc'
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            dataset_name: Name of dataset to optimize for
            n_trials: Number of optimization trials
            study_name: Name for the Optuna study
            storage: Database URL for study persistence (optional)
            direction: 'maximize' or 'minimize'
            metric: Metric to optimize ('val_acc', 'val_loss')
        """
        self.dataset_name = dataset_name
        self.n_trials = n_trials
        self.study_name = study_name or f"{dataset_name}_optimization"
        self.storage = storage
        self.direction = direction
        self.metric = metric
        
        # Initialize trainer
        self.trainer = ModelTrainer(dataset_name)
        
        # Get search space
        self.search_space = get_search_space(dataset_name)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def create_objective(self) -> Callable[[Trial], float]:
        """
        Create the objective function for Optuna.
        
        Returns:
            Objective function that takes a trial and returns the metric value
        """
        def objective(trial: Trial) -> float:
            # Sample hyperparameters
            params = self._sample_hyperparameters(trial)
            
            # Create model
            model = self._create_model_from_params(params)
            
            # Train model
            try:
                results = self.trainer.train_model(
                    model=model,
                    params=params,
                    verbose=False  # Quiet during optimization
                )
                
                # Get metric value
                if self.metric == 'val_acc':
                    metric_value = results['best_val_acc']
                elif self.metric == 'val_loss':
                    # Get minimum validation loss
                    metric_value = min(results['history']['val_loss'])
                else:
                    raise ValueError(f"Unknown metric: {self.metric}")
                
                # Report intermediate values for pruning
                for epoch, val_acc in enumerate(results['history']['val_acc']):
                    trial.report(val_acc, epoch)
                    
                    # Handle pruning
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                return metric_value
                
            except Exception as e:
                self.logger.warning(f"Trial failed with error: {e}")
                raise optuna.TrialPruned()
        
        return objective
    
    def _sample_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters based on search space.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of sampled parameters
        """
        params = {}
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config['type']
            
            if param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        # Special handling for layer sizes
        if 'n_layers' in params:
            hidden_sizes = []
            layer_size_choices = self.search_space.get('layer_sizes', {}).get('choices', [64])
            
            for i in range(params['n_layers']):
                size = trial.suggest_categorical(f'layer_{i}_size', layer_size_choices)
                hidden_sizes.append(size)
            
            params['hidden_sizes'] = hidden_sizes
        
        # Add fixed parameters
        params['seed'] = 42  # Fixed seed for reproducibility
        params['epochs'] = 100
        params['early_stopping_patience'] = 10
        
        return params
    
    def _create_model_from_params(self, params: Dict[str, Any]) -> torch.nn.Module:
        """
        Create a model based on sampled parameters.
        
        Args:
            params: Hyperparameters
            
        Returns:
            PyTorch model
        """
        model_config = {
            'hidden_sizes': params.get('hidden_sizes', [64, 32]),
            'activation': params.get('activation', 'relu'),
            'dropout': params.get('dropout', 0.0),
            'batch_norm': params.get('batch_norm', False),
            'init_method': params.get('init_method', 'xavier')
        }
        
        return create_model(
            model_type='standard',
            input_size=self.trainer.input_size,
            output_size=self.trainer.output_size,
            config=model_config
        )
    
    def optimize(self, resume: bool = False) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            resume: Whether to resume from existing study
            
        Returns:
            Dictionary with best parameters and results
        """
        # Create or load study
        if resume and self.storage:
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self.storage,
                sampler=TPESampler(seed=42),
                pruner=MedianPruner()
            )
            print(f"Resuming study '{self.study_name}' with {len(study.trials)} existing trials")
        else:
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(),
                direction=self.direction
            )
        
        # Create objective
        objective = self.create_objective()
        
        # Run optimization
        print(f"Starting optimization with {self.n_trials} trials...")
        study.optimize(objective, n_trials=self.n_trials, n_jobs=1)
        
        # Get results
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"\nOptimization completed!")
        print(f"Best {self.metric}: {best_value:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Save study results
        results = {
            'best_params': best_params,
            'best_value': float(best_value),
            'n_trials': len(study.trials),
            'study_name': self.study_name,
            'dataset': self.dataset_name,
            'metric': self.metric
        }
        
        # Save to file
        output_dir = Path(__file__).parent / 'optimization_results'
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / f'{self.dataset_name}_best_params.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_file}")
        
        # Also save full study statistics
        self._save_study_statistics(study, output_dir)
        
        return results
    
    def _save_study_statistics(self, study: optuna.Study, output_dir: Path):
        """Save detailed study statistics."""
        stats = {
            'n_trials': len(study.trials),
            'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'best_trial': {
                'number': study.best_trial.number,
                'value': float(study.best_trial.value),
                'params': study.best_trial.params
            }
        }
        
        # Get top 5 trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=(self.direction == 'maximize'))
        
        stats['top_5_trials'] = [
            {
                'number': t.number,
                'value': float(t.value),
                'params': t.params
            }
            for t in sorted_trials[:5]
        ]
        
        stats_file = output_dir / f'{self.dataset_name}_study_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    @staticmethod
    def load_best_params(dataset_name: str) -> Dict[str, Any]:
        """
        Load best parameters from a previous optimization.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Dictionary of best parameters
        """
        results_file = Path(__file__).parent / 'optimization_results' / f'{dataset_name}_best_params.json'
        
        if not results_file.exists():
            raise FileNotFoundError(f"No optimization results found for {dataset_name}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        return results['best_params']