"""Apple quality routing experiment with Concept Trajectory Analysis.

This experiment demonstrates how neural networks classify apple quality (routing)
and tracks how different varieties flow through these quality predictions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# Add parent directories to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from concept_fragmentation.experiments.base import BaseExperiment
from concept_fragmentation.experiments.config import ExperimentConfig
from concept_fragmentation.models.feedforward import FeedforwardNetwork
from concept_fragmentation.activation.collector import ActivationCollector, CollectionConfig
from concept_fragmentation.clustering import select_optimal_k
from concept_fragmentation.analysis.cross_layer_metrics import (
    extract_paths, 
    compute_trajectory_fragmentation,
    analyze_cross_layer_metrics
)
from concept_fragmentation.visualization.sankey import SankeyGenerator
from concept_fragmentation.visualization.trajectory import TrajectoryVisualizer
from concept_fragmentation.visualization.configs import SankeyConfig, TrajectoryConfig
from concept_fragmentation.llm.analysis import ClusterAnalysis
from concept_fragmentation.llm.bias_audit import generate_bias_report

from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class AppleVarietyExperiment(BaseExperiment):
    """Experiment for apple quality routing with variety trajectory analysis."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the experiment.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create ExperimentConfig
        self.experiment_config = ExperimentConfig(
            name=config_dict['experiment']['name'],
            output_dir=config_dict['experiment']['output_dir'],
            random_seed=config_dict['experiment']['random_seed']
        )
        
        super().__init__(self.experiment_config)
        
        # Store full config
        self.full_config = config_dict
        
        # Initialize attributes
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.activation_collector = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.variety_encoder = LabelEncoder()
        self.routing_encoder = LabelEncoder()
        
    def setup(self) -> None:
        """Set up the experiment - load data, initialize model."""
        logger.info("Setting up Apple Quality Routing experiment...")
        
        # Load data
        import pandas as pd
        df = pd.read_csv(self.full_config['dataset']['data_path'])
        
        # Filter to known routing classes
        known_routing = ['fresh_premium', 'fresh_standard', 'juice']
        df = df[df['routing'].isin(known_routing)].copy()
        
        logger.info(f"Routing distribution: {df['routing'].value_counts().to_dict()}")
        
        # Get features
        feature_cols = self.full_config['dataset']['features']
        
        # Add derived features
        df['sweetness_ratio'] = df['brix_numeric'] / (df['firmness_numeric'] + 1e-6)
        df['quality_index'] = (
            0.3 * df['brix_numeric'] / df['brix_numeric'].max() +
            0.3 * df['firmness_numeric'] / df['firmness_numeric'].max() +
            0.2 * df['size_numeric'] / df['size_numeric'].max() +
            0.2 * (1 - df['starch_numeric'] / df['starch_numeric'].max())
        )
        feature_cols.extend(['sweetness_ratio', 'quality_index'])
        
        # Prepare features and labels
        X = df[feature_cols].fillna(df[feature_cols].mean()).values
        y_routing = self.routing_encoder.fit_transform(df['routing'])
        y_variety = self.variety_encoder.fit_transform(df['variety'])
        
        # Store routing classes and variety names
        self.routing_classes = self.routing_encoder.classes_
        self.variety_names = self.variety_encoder.classes_
        self.n_routing_classes = len(self.routing_classes)
        
        logger.info(f"Routing classes: {self.routing_classes}")
        logger.info(f"Number of varieties: {len(self.variety_names)}")
        
        # Train/test split
        X_train, X_test, y_train, y_test, variety_train, variety_test = train_test_split(
            X, y_routing, y_variety, 
            test_size=self.full_config['dataset']['test_size'],
            random_state=self.full_config['experiment']['random_seed'],
            stratify=y_routing
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Store variety labels for analysis
        self.variety_train = variety_train
        self.variety_test = variety_test
        
        # Store original unscaled features for interpretable cluster descriptions
        self.X_train_original = X_train
        self.y_train_routing = y_train
        self.feature_names = feature_cols
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        batch_size = self.full_config['training']['batch_size']
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model_config = self.full_config['model']['architecture']
        self.model = FeedforwardNetwork(
            input_dim=len(feature_cols),
            output_dim=self.n_routing_classes,  # Predict quality routing
            hidden_layer_sizes=model_config['hidden_dims'],
            dropout_rate=model_config['dropout_rate'],
            activation=model_config['activation'],
            seed=self.full_config['experiment']['random_seed']
        )
        self.model.to(self.device)
        
        # Set up activation collector
        collector_config = CollectionConfig(
            device=str(self.device),
            log_dimensions=True
        )
        self.activation_collector = ActivationCollector(collector_config)
        
        # Register model with collector - capture all layers
        # The feedforward model has fc1, fc2, fc3, and output layers
        layer_names = ['fc1', 'fc2', 'fc3', 'output']
        self.activation_collector.register_model(
            self.model,
            model_id='apple_model',
            activation_points=layer_names
        )
        
        # Store metadata
        self.n_features = len(feature_cols)
        self.n_train = len(X_train)
        self.n_test = len(X_test)
        
        # Initialize LLM analyzer
        self.llm_analyzer = None
        try:
            # Try to import API key
            from local_config import OPENAI_KEY
            
            # Initialize with configuration from full_config
            llm_config = self.full_config.get('llm', {})
            provider = llm_config.get('provider', 'openai')
            model = llm_config.get('model', 'gpt-4')
            use_cache = llm_config.get('use_cache', True)
            debug = llm_config.get('debug', False)
            
            self.llm_analyzer = ClusterAnalysis(
                provider=provider,
                model=model,
                api_key=OPENAI_KEY if provider == 'openai' else None,
                use_cache=use_cache,
                debug=debug
            )
            logger.info(f"LLM analyzer initialized: {provider}/{model}")
        except ImportError:
            logger.warning("local_config.py not found - LLM analysis will be skipped")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM analyzer: {e}")
        
        logger.info(f"Setup complete: {self.n_features} features, {self.n_routing_classes} quality classes")
        logger.info(f"Training samples: {self.n_train}, Test samples: {self.n_test}")
        
    def execute(self) -> Dict[str, Any]:
        """Execute the experiment - train model and collect activations."""
        logger.info("Training model...")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.full_config['training']['learning_rate'],
            weight_decay=self.full_config['training']['weight_decay']
        )
        
        # Training loop
        n_epochs = self.full_config['training']['epochs']
        train_losses = []
        train_accuracies = []
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            accuracy = 100 * correct / total
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            if (epoch + 1) % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Evaluate on test set
        self.model.eval()
        test_correct = 0
        test_total = 0
        test_predictions = []
        
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                test_predictions.extend(predicted.cpu().numpy())
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
        
        # Collect activations
        logger.info("Collecting activations...")
        
        # Prepare full datasets for activation collection  
        # Use the already scaled tensors we created in setup
        X_train_full = torch.cat([x for x, _ in self.train_loader], dim=0)
        X_test_full = torch.cat([x for x, _ in self.test_loader], dim=0)
        
        # Collect train activations
        train_activations = self.activation_collector.collect(
            self.model,
            X_train_full.to(self.device),
            model_id='apple_model'
        )
        
        # Collect test activations
        test_activations = self.activation_collector.collect(
            self.model,
            X_test_full.to(self.device),
            model_id='apple_model'
        )
        
        # Save model
        model_path = self.output_dir / "model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'routing_classes': self.routing_classes,
            'variety_names': self.variety_names,
            'routing_encoder': self.routing_encoder,
            'variety_encoder': self.variety_encoder,
            'config': self.full_config
        }, model_path)
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracy': test_accuracy,
            'test_predictions': test_predictions,
            'train_activations': train_activations,
            'test_activations': test_activations,
            'n_train': self.n_train,
            'n_test': self.n_test
        }
        
    def analyze(self) -> Dict[str, Any]:
        """Analyze results - clustering, path extraction, fragmentation."""
        logger.info("Analyzing trajectories...")
        
        # Get activations from results
        train_activations = self.results['train_activations']
        test_activations = self.results['test_activations']
        
        # Store for visualization
        self._train_activations = train_activations
        self._test_activations = test_activations
        
        # CRITICAL FIX: Combine all data for CTA analysis
        # CTA principle: analyze ALL data together, not separate train/test
        logger.info("Combining train and test data for CTA analysis...")
        
        layer_names = list(train_activations.keys())
        all_activations = {}
        
        # Combine activations for each layer
        for layer_name in layer_names:
            all_activations[layer_name] = np.vstack([
                train_activations[layer_name],
                test_activations[layer_name]
            ])
        
        # Combine labels and variety information
        y_train = torch.cat([y for _, y in self.train_loader], dim=0).numpy()
        y_test = torch.cat([y for _, y in self.test_loader], dim=0).numpy()
        self.all_labels = np.hstack([y_train, y_test])
        self.all_varieties = np.hstack([self.variety_train, self.variety_test])
        
        # Store for output routing calculation
        self.y_train_routing = y_train  # Keep for compatibility
        self.all_routing_labels = self.all_labels
        
        # Cluster ALL data together
        all_clusters = {}
        cluster_info = {}
        
        clustering_config = self.full_config['clustering']
        
        for layer_name in layer_names:
            logger.info(f"Clustering {layer_name} with ALL data...")
            
            # Select optimal k using gap statistic on combined data
            optimal_k, scores = select_optimal_k(
                all_activations[layer_name],
                k_range=(clustering_config['k_min'], clustering_config['k_max']),
                method=clustering_config['k_selection'],
                return_scores=True,
                random_state=clustering_config['random_state']
            )
            
            logger.info(f"  Optimal k for {layer_name}: {optimal_k}")
            
            # Fit clustering on ALL data
            kmeans = KMeans(n_clusters=optimal_k, random_state=clustering_config['random_state'])
            all_clusters[layer_name] = kmeans.fit_predict(all_activations[layer_name])
            
            # Store cluster info
            cluster_info[layer_name] = {
                'n_clusters': optimal_k,
                'gap_scores': scores,
                'cluster_centers': kmeans.cluster_centers_
            }
        
        # Store clusters for later access
        self.all_clusters = all_clusters
        
        # Split back for compatibility with existing analysis functions
        n_train = self.n_train
        train_clusters = {}
        test_clusters = {}
        
        for layer_name in layer_names:
            train_clusters[layer_name] = all_clusters[layer_name][:n_train]
            test_clusters[layer_name] = all_clusters[layer_name][n_train:]
        
        # Get labels from loaders
        y_train = torch.cat([y for _, y in self.train_loader], dim=0).numpy()
        y_test = torch.cat([y for _, y in self.test_loader], dim=0).numpy()
        
        # Analyze trajectories and fragmentation (core CTA without cross-layer similarity)
        train_metrics = self._analyze_trajectories(train_clusters, y_train)
        test_metrics = self._analyze_trajectories(test_clusters, y_test)
        
        # Analyze variety-specific routing
        variety_routing_analysis = self._analyze_variety_routing(
            train_clusters, test_clusters, 
            y_train, y_test,
            self.variety_train, self.variety_test
        )
        
        # Store original metrics for visualization
        self._train_metrics_original = train_metrics
        self._test_metrics_original = test_metrics
        
        # Generate LLM-powered analysis if available
        llm_analysis = {}
        if self.llm_analyzer is not None:
            logger.info("Generating LLM-powered cluster analysis...")
            try:
                llm_analysis = self._generate_llm_analysis(
                    train_clusters, train_metrics, 
                    y_train, self.variety_train
                )
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
                llm_analysis = {'error': str(e)}
        
        # Convert metrics to JSON-serializable format
        train_metrics_clean = self._clean_metrics_for_json(train_metrics)
        test_metrics_clean = self._clean_metrics_for_json(test_metrics)
        
        return {
            'cluster_info': self._clean_metrics_for_json(cluster_info),
            'train_metrics': train_metrics_clean,
            'test_metrics': test_metrics_clean,
            'variety_routing': variety_routing_analysis,
            'llm_analysis': llm_analysis
        }
    
    def _clean_metrics_for_json(self, metrics: Dict) -> Dict:
        """Convert metrics to JSON-serializable format."""
        clean_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                # Handle dictionaries with tuple keys
                if any(isinstance(k, tuple) for k in value.keys()):
                    # Convert tuple keys to strings
                    clean_dict = {}
                    for k, v in value.items():
                        if isinstance(k, tuple):
                            key_str = "_".join(str(x) for x in k)
                            clean_dict[key_str] = self._clean_value_for_json(v)
                        else:
                            clean_dict[k] = self._clean_value_for_json(v)
                    clean_metrics[key] = clean_dict
                else:
                    clean_metrics[key] = self._clean_metrics_for_json(value)
            elif isinstance(value, np.ndarray):
                clean_metrics[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                clean_metrics[key] = float(value)
            else:
                clean_metrics[key] = value
                
        return clean_metrics
    
    def _clean_value_for_json(self, value: Any) -> Any:
        """Convert a single value to JSON-serializable format."""
        if isinstance(value, dict):
            # Recursively clean dictionary
            return self._clean_metrics_for_json(value)
        elif isinstance(value, (list, tuple)):
            # Clean each element in list/tuple
            return [self._clean_value_for_json(v) for v in value]
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, set):
            return list(value)
        else:
            # Return as-is for strings, ints, floats, bools, None
            return value
    
    def _analyze_variety_routing(
        self, 
        train_clusters: Dict,
        test_clusters: Dict,
        y_train_quality: np.ndarray,
        y_test_quality: np.ndarray,
        variety_train: np.ndarray,
        variety_test: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze how each variety gets routed through quality predictions."""
        
        # Extract paths
        train_paths, layer_names = extract_paths(train_clusters)
        test_paths, _ = extract_paths(test_clusters)
        
        # For each variety, track its routing patterns
        variety_results = {}
        
        for variety_idx, variety_name in enumerate(self.variety_names):
            # Get samples for this variety
            train_mask = variety_train == variety_idx
            test_mask = variety_test == variety_idx
            
            if np.sum(train_mask) == 0 and np.sum(test_mask) == 0:
                continue
            
            variety_data = {
                'name': variety_name,
                'train_samples': int(np.sum(train_mask)),
                'test_samples': int(np.sum(test_mask))
            }
            
            # Analyze quality routing for this variety
            if np.sum(train_mask) > 0:
                variety_quality = y_train_quality[train_mask]
                quality_dist = {}
                for q_idx, q_name in enumerate(self.routing_classes):
                    count = np.sum(variety_quality == q_idx)
                    quality_dist[q_name] = {
                        'count': int(count),
                        'percentage': float(count / len(variety_quality) * 100)
                    }
                variety_data['train_quality_distribution'] = quality_dist
                
                # Track unique paths for this variety
                variety_paths = train_paths[train_mask]
                unique_paths = np.unique(variety_paths, axis=0)
                variety_data['n_unique_paths'] = len(unique_paths)
                
                # Fragmentation score
                if len(variety_paths) > 1:
                    variety_data['fragmentation'] = len(unique_paths) / len(variety_paths)
                else:
                    variety_data['fragmentation'] = 1.0
            
            # Test set routing
            if np.sum(test_mask) > 0:
                variety_quality_test = y_test_quality[test_mask]
                test_predictions = self.results['test_predictions']
                variety_predictions = [test_predictions[i] for i, m in enumerate(test_mask) if m]
                
                # Misrouting analysis
                misrouted = 0
                for true_q, pred_q in zip(variety_quality_test[test_mask[test_mask]], variety_predictions):
                    if true_q != pred_q:
                        misrouted += 1
                
                variety_data['test_misrouting_rate'] = misrouted / len(variety_predictions) if variety_predictions else 0
            
            variety_results[variety_name] = variety_data
        
        # Identify problematic varieties (high fragmentation or misrouting)
        problematic_varieties = []
        for variety, data in variety_results.items():
            if 'fragmentation' in data and data['fragmentation'] > 0.7:
                problematic_varieties.append({
                    'variety': variety,
                    'issue': 'high_fragmentation',
                    'score': data['fragmentation']
                })
            if 'test_misrouting_rate' in data and data['test_misrouting_rate'] > 0.3:
                problematic_varieties.append({
                    'variety': variety,
                    'issue': 'high_misrouting',
                    'score': data['test_misrouting_rate']
                })
        
        return {
            'by_variety': variety_results,
            'problematic_varieties': sorted(problematic_varieties, key=lambda x: x['score'], reverse=True),
            'routing_classes': self.routing_classes.tolist()
        }
    
    def _generate_llm_analysis(
        self,
        train_clusters: Dict[str, np.ndarray],
        train_metrics: Dict[str, Any],
        y_train: np.ndarray,
        variety_train: np.ndarray
    ) -> Dict[str, Any]:
        """Generate LLM-powered analysis of apple quality routing clusters."""
        
        # Generate cluster profiles for LLM analysis
        cluster_profiles = self._generate_cluster_profiles(
            train_clusters, y_train, variety_train
        )
        
        # Extract paths for analysis
        paths = train_metrics['paths']
        layer_names = train_metrics['layer_names']
        
        # Convert paths to dictionary format expected by LLM analyzer
        paths_dict = {}
        for i, path in enumerate(paths):
            # Convert path array to cluster IDs with layer names
            path_clusters = []
            for j, cluster_idx in enumerate(path):
                if j < len(layer_names):
                    # Use layer index for CTA standard naming
                    cluster_id = f"L{j}_C{cluster_idx}"
                    path_clusters.append(cluster_id)
            
            if len(path_clusters) > 0:
                paths_dict[i] = path_clusters
        
        # Take top archetypal paths to avoid token limits
        unique_paths, counts = np.unique(paths, axis=0, return_counts=True)
        top_indices = np.argsort(counts)[::-1][:20]  # Top 20 paths
        
        archetypal_paths = {}
        for idx_rank, path_idx in enumerate(top_indices):
            path = unique_paths[path_idx]
            path_clusters = []
            for j, cluster_idx in enumerate(path):
                if j < len(layer_names):
                    # Use layer index for CTA standard naming
                    cluster_id = f"L{j}_C{cluster_idx}"
                    path_clusters.append(cluster_id)
            
            if len(path_clusters) > 0:
                archetypal_paths[idx_rank] = path_clusters
        
        # Generate path demographic info
        path_demographic_info = self._generate_path_demographics(
            archetypal_paths, unique_paths[top_indices], 
            paths, y_train, variety_train
        )
        
        # Call LLM analyzer with comprehensive analysis
        analysis_categories = ['interpretation', 'bias']
        
        try:
            llm_results = self.llm_analyzer.analyze_and_label_clusters_sync(
                paths=archetypal_paths,
                cluster_stats=cluster_profiles,
                path_demographic_info=path_demographic_info,
                analysis_categories=analysis_categories
            )
            
            # Generate bias report
            if len(path_demographic_info) > 0:
                bias_report = generate_bias_report(
                    paths=archetypal_paths,
                    demographic_info=path_demographic_info,
                    demographic_columns=['variety', 'routing_quality'],
                    cluster_labels=llm_results.get('cluster_labels', {})
                )
                llm_results['bias_report'] = bias_report
            
            return llm_results
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {'error': str(e), 'cluster_profiles': cluster_profiles}
    
    def _generate_cluster_profiles(
        self, 
        train_clusters: Dict[str, np.ndarray], 
        y_train: np.ndarray, 
        variety_train: np.ndarray
    ) -> Dict[str, str]:
        """Generate textual profiles for each cluster using original interpretable values."""
        
        cluster_profiles = {}
        
        # Get layer names in order for indexing
        layer_names = list(train_clusters.keys())
        
        for layer_idx, (layer_name, clusters) in enumerate(train_clusters.items()):
            for cluster_id in np.unique(clusters):
                cluster_mask = clusters == cluster_id
                
                if np.sum(cluster_mask) == 0:
                    continue
                
                # Get samples in this cluster using stored original data
                cluster_features = self.X_train_original[cluster_mask]
                cluster_routing = self.y_train_routing[cluster_mask]
                cluster_varieties = self.variety_train[cluster_mask]
                
                # Calculate feature statistics on ORIGINAL unscaled values
                feature_means = np.mean(cluster_features, axis=0)
                
                # Routing distribution
                routing_dist = {}
                for routing_idx, routing_name in enumerate(self.routing_classes):
                    count = np.sum(cluster_routing == routing_idx)
                    routing_dist[routing_name] = count / len(cluster_routing) * 100
                
                # Variety distribution (top 3)
                variety_counts = np.bincount(cluster_varieties, minlength=len(self.variety_names))
                top_variety_indices = np.argsort(variety_counts)[::-1][:3]
                
                # Create profile text using interpretable original feature values
                # Map feature indices to names for clarity
                brix_idx = self.feature_names.index('brix_numeric')
                firmness_idx = self.feature_names.index('firmness_numeric')
                size_idx = self.feature_names.index('size_numeric')
                sweetness_idx = len(self.feature_names) - 2  # sweetness_ratio is second to last
                quality_idx = len(self.feature_names) - 1    # quality_index is last
                
                profile_lines = [
                    f"Quality routing: {max(routing_dist, key=routing_dist.get)} ({routing_dist[max(routing_dist, key=routing_dist.get)]:.1f}%)",
                    f"Brix: {feature_means[brix_idx]:.2f}, Firmness: {feature_means[firmness_idx]:.2f}, Size: {feature_means[size_idx]:.2f}",
                    f"Top varieties: {', '.join([self.variety_names[i] for i in top_variety_indices if variety_counts[i] > 0])}",
                    f"Sweetness ratio: {feature_means[sweetness_idx]:.2f}, Quality index: {feature_means[quality_idx]:.2f}"
                ]
                
                cluster_key = f"L{layer_idx}_C{cluster_id}"
                cluster_profiles[cluster_key] = "; ".join(profile_lines)
        
        return cluster_profiles
    
    def _generate_path_demographics(
        self,
        archetypal_paths: Dict[int, List[str]],
        unique_paths: np.ndarray,
        all_paths: np.ndarray,
        y_train: np.ndarray,
        variety_train: np.ndarray
    ) -> Dict[int, Dict[str, Any]]:
        """Generate demographic information for archetypal paths."""
        
        path_demographics = {}
        
        for path_idx, path_clusters in archetypal_paths.items():
            if path_idx >= len(unique_paths):
                continue
                
            # Find samples that follow this exact path
            target_path = unique_paths[path_idx]
            path_mask = np.all(all_paths == target_path, axis=1)
            
            if np.sum(path_mask) == 0:
                continue
            
            # Get demographics for samples following this path
            path_routing = y_train[path_mask]
            path_varieties = variety_train[path_mask]
            
            # Routing distribution
            routing_dist = {}
            for routing_idx, routing_name in enumerate(self.routing_classes):
                count = np.sum(path_routing == routing_idx)
                routing_dist[routing_name] = count / len(path_routing) if len(path_routing) > 0 else 0
            
            # Variety distribution
            variety_dist = {}
            for variety_idx, variety_name in enumerate(self.variety_names):
                count = np.sum(path_varieties == variety_idx)
                variety_dist[variety_name] = count / len(path_varieties) if len(path_varieties) > 0 else 0
            
            # Get dominant characteristics
            dominant_routing = max(routing_dist, key=routing_dist.get)
            dominant_variety = max(variety_dist, key=variety_dist.get)
            
            path_demographics[path_idx] = {
                'routing_quality': routing_dist,
                'variety': variety_dist,
                'dominant_routing': dominant_routing,
                'dominant_variety': dominant_variety,
                'sample_count': int(np.sum(path_mask)),
                'routing_purity': routing_dist[dominant_routing],
                'variety_purity': variety_dist[dominant_variety]
            }
        
        return path_demographics
    
    def _analyze_trajectories(self, clusters: Dict[str, np.ndarray], labels: np.ndarray) -> Dict[str, Any]:
        """Analyze trajectories and fragmentation without cross-layer similarity."""
        from concept_fragmentation.analysis.cross_layer_metrics import extract_paths, compute_trajectory_fragmentation
        
        # Extract paths through clusters
        paths, layer_names = extract_paths(clusters)
        
        # Compute trajectory fragmentation
        fragmentation = compute_trajectory_fragmentation(
            paths=paths,
            layer_names=layer_names,
            labels=labels
        )
        
        # Basic path statistics
        unique_paths, counts = np.unique(paths, axis=0, return_counts=True)
        path_frequencies = counts / len(paths)
        
        # Most common paths
        top_indices = np.argsort(counts)[::-1][:20]  # Top 20 paths
        archetypal_paths = []
        for idx in top_indices:
            path = unique_paths[idx]
            count = counts[idx]
            frequency = count / len(paths)
            
            archetypal_paths.append({
                'path': path.tolist(),
                'count': int(count),
                'frequency': float(frequency),
                'percentage': float(frequency * 100)
            })
        
        return {
            'paths': paths,
            'layer_names': layer_names,
            'unique_paths': unique_paths,
            'path_counts': counts,
            'path_frequencies': path_frequencies,
            'archetypal_paths': archetypal_paths,
            'fragmentation': fragmentation,
            'total_paths': len(paths),
            'unique_path_count': len(unique_paths)
        }
    
    
    def visualize(self) -> Dict[str, str]:
        """Create visualizations - Sankey diagrams and trajectory plots."""
        logger.info("Creating visualizations...")
        
        viz_paths = {}
        
        # Prepare data for visualization
        # Use original metrics with tuple keys for visualization
        train_metrics = self._train_metrics_original
        test_metrics = self._test_metrics_original
        variety_routing = self.results['analysis']['variety_routing']
        
        # Get labels from loaders
        y_train = torch.cat([y for _, y in self.train_loader], dim=0).numpy()
        y_test = torch.cat([y for _, y in self.test_loader], dim=0).numpy()
        
        # 1. Create Sankey diagram for quality routing
        sankey_data = self._prepare_routing_sankey_data(
            train_metrics, y_train, self.variety_train
        )
        
        # Add LLM labels and path narratives if available
        if 'llm_analysis' in self.results['analysis']:
            llm_analysis = self.results['analysis']['llm_analysis']
            
            # Add cluster labels
            if 'cluster_labels' in llm_analysis:
                llm_labels = llm_analysis['cluster_labels']
                sankey_data = self._add_llm_labels_to_sankey(sankey_data, llm_labels)
            
            # Add path narratives to archetypal paths
            if 'archetypal_paths' in llm_analysis:
                sankey_data = self._add_path_narratives_to_sankey(sankey_data, llm_analysis['archetypal_paths'])
        
        sankey_config = SankeyConfig(
            height=self.full_config['visualization']['sankey']['height'],
            width=self.full_config['visualization']['sankey']['width'],
            last_layer_labels_position='inline'  # Keep labels inline for consistency
        )
        
        sankey_gen = SankeyGenerator(sankey_config)
        
        # Create single Sankey for full network
        if 'full' in sankey_data['windowed_analysis']:
            fig = sankey_gen.create_figure(
                sankey_data,
                window='full',
                top_n_paths=self.full_config['visualization']['sankey']['top_n_paths']
            )
            
            # Save as HTML
            html_path = self.output_dir / "sankey_full_network.html"
            fig.write_html(str(html_path))
            viz_paths['sankey_full_network'] = str(html_path)
            
            # Save as PNG if requested
            if 'png' in self.full_config['visualization']['save_format']:
                png_path = self.output_dir / "sankey_full_network.png"
                fig.write_image(str(png_path))
                viz_paths['sankey_full_network_png'] = str(png_path)
        
        # 2. Create 3D trajectory visualization
        # Skip trajectory visualization for now - requires same-sized layers
        # TODO: Implement proper trajectory visualization with varying layer sizes
        logger.info("Skipping 3D trajectory visualization (requires uniform layer sizes)")
        
        # traj_config = TrajectoryConfig(
        #     reduction_method=self.full_config['visualization']['trajectory']['method'],
        #     dimensions=self.full_config['visualization']['trajectory']['n_components'],
        #     min_dist=self.full_config['visualization']['trajectory']['min_dist'],
        #     n_neighbors=self.full_config['visualization']['trajectory']['n_neighbors']
        # )
        # 
        # traj_viz = TrajectoryVisualizer(traj_config)
        # traj_fig = traj_viz.create_figure(traj_data)
        # 
        # # Save trajectory visualization
        # traj_path = self.output_dir / "trajectories_3d.html"
        # traj_fig.write_html(str(traj_path))
        # viz_paths['trajectories_3d'] = str(traj_path)
        
        # 3. Create variety routing analysis visualization
        self._create_variety_routing_plot(variety_routing)
        viz_paths['variety_routing'] = str(self.output_dir / "variety_routing_analysis.png")
        
        # 4. Create economic impact visualization
        self._create_economic_impact_plot(variety_routing)
        viz_paths['economic_impact'] = str(self.output_dir / "economic_impact.png")
        
        return viz_paths
    
    def _prepare_routing_sankey_data(
        self, 
        metrics: Dict,
        quality_labels: np.ndarray,
        variety_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Prepare Sankey data showing variety routing through quality predictions."""
        paths = metrics['paths']
        layer_names = metrics['layer_names']
        
        # For small 4-layer network, just use full network view
        windows = {
            'full': [0, 1, 2, 3]   # All layers L0->L1->L2->L3
        }
        
        windowed_analysis = {}
        
        for window_name, window_layers in windows.items():
            # Extract paths for this window
            start_idx = window_layers[0]
            end_idx = window_layers[-1]
            window_paths = paths[:, start_idx:end_idx+1]
            
            # Count unique paths
            unique_paths, counts = np.unique(
                window_paths, axis=0, return_counts=True
            )
            
            # Get top paths
            top_indices = np.argsort(counts)[::-1][:15]  # Top 15 paths
            
            archetypal_paths = []
            for idx in top_indices:
                path = unique_paths[idx]
                count = counts[idx]
                
                # Find varieties that follow this path
                path_mask = np.all(window_paths == path, axis=1)
                path_varieties = variety_labels[path_mask]
                variety_counts = np.bincount(path_varieties, minlength=len(self.variety_names))
                top_variety_idx = np.argmax(variety_counts)
                
                # Get quality routing for this path
                path_quality = quality_labels[path_mask]
                quality_counts = np.bincount(path_quality, minlength=len(self.routing_classes))
                dominant_quality_idx = np.argmax(quality_counts)
                dominant_quality = self.routing_classes[dominant_quality_idx]
                
                archetypal_paths.append({
                    'path': path.tolist(),
                    'frequency': int(count),
                    'percentage': float(count / len(window_paths) * 100),
                    'representative_words': [self.variety_names[top_variety_idx]],
                    'dominant_quality': dominant_quality,
                    'quality_purity': float(quality_counts[dominant_quality_idx] / np.sum(quality_counts)),
                    'semantic_labels': None  # Will be filled by LLM if available
                })
            
            windowed_analysis[window_name] = {
                'layers': list(range(start_idx, end_idx+1)),
                'total_paths': len(window_paths),
                'unique_paths': len(unique_paths),
                'archetypal_paths': archetypal_paths
            }
        
        # Create labels dict in expected format using CTA naming
        labels_dict = {}
        for i, layer_name in enumerate(layer_names):
            # Use CTA layer naming (L0, L1, L2, L3)
            layer_key = f"L{i}"
            labels_dict[layer_key] = {
                f"L{i}_C{cluster_id}": {'words': [], 'semantic_label': f'C{cluster_id}', 'label': f'C{cluster_id}'}
                for cluster_id in np.unique(paths[:, i])
            }
        
        return {
            'windowed_analysis': windowed_analysis,
            'labels': labels_dict,
            'purity_data': None
        }
    
    def _create_fragmentation_heatmap(self, variety_metrics: Dict) -> Path:
        """Create heatmap showing fragmentation by variety."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Extract fragmentation data
        varieties = []
        train_frag = []
        test_frag = []
        
        for variety_name, data in variety_metrics['by_variety'].items():
            if 'train_fragmentation' in data:
                varieties.append(variety_name)
                train_frag.append(data['train_fragmentation'])
                test_frag.append(data.get('test_fragmentation', 0))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Train fragmentation
        ax1.barh(varieties, train_frag)
        ax1.set_xlabel('Fragmentation Rate')
        ax1.set_title('Training Set Fragmentation by Variety')
        ax1.set_xlim(0, 1)
        
        # Test fragmentation
        ax2.barh(varieties, test_frag)
        ax2.set_xlabel('Fragmentation Rate')
        ax2.set_title('Test Set Fragmentation by Variety')
        ax2.set_xlim(0, 1)
        
        plt.tight_layout()
        
        # Save
        frag_path = self.output_dir / "fragmentation_by_variety.png"
        plt.savefig(frag_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return frag_path
    
    def _add_llm_labels_to_sankey(self, sankey_data: Dict, llm_labels: Dict[str, str]) -> Dict:
        """Add LLM-generated labels to sankey visualization data."""
        
        # Update labels in sankey data with LLM semantic labels
        for layer_key, layer_clusters in sankey_data['labels'].items():
            for cluster_key, cluster_data in layer_clusters.items():
                # cluster_key is already in CTA format (L0_C0, L1_C1, etc.)
                if cluster_key in llm_labels:
                    # For output layer (L3), add routing distribution to label
                    if cluster_key.startswith('L3'):
                        # Get routing distribution for this cluster
                        routing_info = self._get_output_cluster_routing(cluster_key)
                        if routing_info:
                            # Combine semantic label with routing distribution
                            semantic_label = llm_labels[cluster_key]
                            cluster_data['label'] = f"{semantic_label}\n→ {routing_info}"
                        else:
                            cluster_data['label'] = llm_labels[cluster_key]
                    else:
                        # Non-output layers just use semantic label
                        cluster_data['label'] = llm_labels[cluster_key]
                    
                    # Also keep as semantic_label for compatibility
                    cluster_data['semantic_label'] = llm_labels[cluster_key]
        
        return sankey_data
    
    def _get_output_cluster_routing(self, cluster_key: str) -> Optional[str]:
        """Get routing distribution for an output layer cluster."""
        # Extract cluster ID from key (e.g., "L3_C2" -> 2)
        try:
            cluster_id = int(cluster_key.split('_C')[1])
        except:
            return None
        
        # Use stored cluster assignments from CTA analysis
        if not hasattr(self, 'all_clusters'):
            logger.warning("No cluster assignments found")
            return None
        
        # Handle both 'output' and index-based naming (layer 3 = output)
        if 'output' in self.all_clusters:
            output_clusters = self.all_clusters['output']
        else:
            # Get the last layer (output layer) by index
            layer_names = list(self.all_clusters.keys())
            if len(layer_names) >= 4:
                output_layer_name = layer_names[3]  # 4th layer (0-indexed)
                output_clusters = self.all_clusters[output_layer_name]
            else:
                logger.warning("Could not find output layer in cluster assignments")
                return None
        cluster_mask = output_clusters == cluster_id
        
        if np.sum(cluster_mask) == 0:
            return None
        
        # Get routing labels for samples in this cluster
        cluster_routing = self.all_routing_labels[cluster_mask]
        
        # Calculate distribution
        routing_counts = {}
        for routing_idx, routing_name in enumerate(self.routing_classes):
            count = np.sum(cluster_routing == routing_idx)
            pct = count / len(cluster_routing) * 100
            if pct > 0:
                routing_counts[routing_name] = pct
        
        # Format as string (show all routes with >5%)
        sorted_routes = sorted(routing_counts.items(), key=lambda x: x[1], reverse=True)
        route_strs = [f"{name} {pct:.0f}%" for name, pct in sorted_routes if pct > 5]
        
        # If no significant routes, show the top one
        if not route_strs and sorted_routes:
            name, pct = sorted_routes[0]
            route_strs = [f"{name} {pct:.0f}%"]
        
        return ", ".join(route_strs)
    
    def _add_path_narratives_to_sankey(self, sankey_data: Dict, archetypal_paths: Dict) -> Dict:
        """Add LLM-generated path narratives to sankey visualization data."""
        
        # Add narratives to windowed analysis paths
        for window_name, window_data in sankey_data['windowed_analysis'].items():
            if 'archetypal_paths' in window_data:
                for path_info in window_data['archetypal_paths']:
                    # Try to find matching path in LLM analysis
                    path_key = tuple(path_info['path'])
                    
                    # Look through archetypal paths for matching path
                    for path_id, llm_path_info in archetypal_paths.items():
                        if 'path' in llm_path_info:
                            # Extract cluster indices from path strings like "L0_C1"
                            llm_path_indices = []
                            for cluster_id in llm_path_info['path']:
                                if '_C' in cluster_id:
                                    cluster_num = int(cluster_id.split('_C')[1])
                                    llm_path_indices.append(cluster_num)
                            
                            # Check if this matches our current path
                            if tuple(llm_path_indices[:len(path_key)]) == path_key:
                                # Add narrative information
                                if 'labeled_path_string' in llm_path_info:
                                    path_info['semantic_labels'] = llm_path_info['labeled_path_string']
                                if 'dominant_routing' in llm_path_info:
                                    path_info['llm_routing'] = llm_path_info['dominant_routing']
                                if 'routing_purity' in llm_path_info:
                                    path_info['llm_purity'] = llm_path_info['routing_purity']
                                break
        
        return sankey_data
    
    def _create_variety_routing_plot(self, variety_routing: Dict) -> None:
        """Create visualization of variety routing patterns."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Extract data for plotting
        varieties = []
        fresh_premium_pct = []
        fresh_standard_pct = []
        juice_pct = []
        fragmentation_scores = []
        
        for variety_name, data in variety_routing['by_variety'].items():
            if 'train_quality_distribution' in data:
                varieties.append(variety_name)
                
                dist = data['train_quality_distribution']
                fresh_premium_pct.append(dist.get('fresh_premium', {}).get('percentage', 0))
                fresh_standard_pct.append(dist.get('fresh_standard', {}).get('percentage', 0))
                juice_pct.append(dist.get('juice', {}).get('percentage', 0))
                fragmentation_scores.append(data.get('fragmentation', 0))
        
        # Create stacked bar chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Quality routing distribution
        x = np.arange(len(varieties))
        width = 0.8
        
        ax1.bar(x, fresh_premium_pct, width, label='Fresh Premium', color='darkgreen', alpha=0.8)
        ax1.bar(x, fresh_standard_pct, width, bottom=fresh_premium_pct, label='Fresh Standard', color='lightgreen', alpha=0.8)
        ax1.bar(x, juice_pct, width, bottom=np.array(fresh_premium_pct) + np.array(fresh_standard_pct), label='Juice', color='orange', alpha=0.8)
        
        ax1.set_xlabel('Apple Variety')
        ax1.set_ylabel('Quality Routing Distribution (%)')
        ax1.set_title('Apple Variety Quality Routing Distribution')
        ax1.set_xticks(x)
        ax1.set_xticklabels(varieties, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 100)
        
        # Fragmentation scores
        bars = ax2.bar(x, fragmentation_scores, color='red', alpha=0.7)
        ax2.set_xlabel('Apple Variety')
        ax2.set_ylabel('Fragmentation Score')
        ax2.set_title('Variety Routing Fragmentation (Higher = More Inconsistent Routing)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(varieties, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        
        # Highlight problematic varieties
        for i, score in enumerate(fragmentation_scores):
            if score > 0.7:
                bars[i].set_color('darkred')
                ax2.text(i, score + 0.02, f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        variety_path = self.output_dir / "variety_routing_analysis.png"
        plt.savefig(variety_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_economic_impact_plot(self, variety_routing: Dict) -> None:
        """Create visualization of economic impact from quality misrouting."""
        import matplotlib.pyplot as plt
        
        # Economic values per routing class ($/lb)
        economic_values = {
            'fresh_premium': 2.50,
            'fresh_standard': 1.50,
            'juice': 0.50
        }
        
        # Calculate economic impact for each variety
        varieties = []
        current_values = []
        potential_values = []
        economic_losses = []
        
        for variety_name, data in variety_routing['by_variety'].items():
            if 'train_quality_distribution' in data and data['train_samples'] > 10:
                varieties.append(variety_name)
                
                dist = data['train_quality_distribution']
                
                # Calculate current weighted value
                current_value = (
                    dist.get('fresh_premium', {}).get('percentage', 0) / 100 * economic_values['fresh_premium'] +
                    dist.get('fresh_standard', {}).get('percentage', 0) / 100 * economic_values['fresh_standard'] +
                    dist.get('juice', {}).get('percentage', 0) / 100 * economic_values['juice']
                )
                
                # Assume optimal routing would put premium varieties in premium category
                # This is a simplified assumption - in practice would need variety quality data
                premium_varieties = ['Honeycrisp', 'Fuji', 'Gala', 'Red Delicious', 'Granny Smith']
                if variety_name in premium_varieties:
                    potential_value = economic_values['fresh_premium']
                else:
                    potential_value = economic_values['fresh_standard']
                
                current_values.append(current_value)
                potential_values.append(potential_value)
                economic_losses.append(potential_value - current_value)
        
        # Create economic impact visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Current vs potential value
        x = np.arange(len(varieties))
        width = 0.35
        
        ax1.bar(x - width/2, current_values, width, label='Current Value', color='orange', alpha=0.8)
        ax1.bar(x + width/2, potential_values, width, label='Potential Value', color='green', alpha=0.8)
        
        ax1.set_xlabel('Apple Variety')
        ax1.set_ylabel('Value ($/lb)')
        ax1.set_title('Current vs Potential Economic Value by Variety')
        ax1.set_xticks(x)
        ax1.set_xticklabels(varieties, rotation=45, ha='right')
        ax1.legend()
        
        # Economic losses
        colors = ['red' if loss > 0.5 else 'orange' if loss > 0.2 else 'green' for loss in economic_losses]
        bars = ax2.bar(x, economic_losses, color=colors, alpha=0.8)
        
        ax2.set_xlabel('Apple Variety')
        ax2.set_ylabel('Economic Loss ($/lb)')
        ax2.set_title('Economic Loss from Quality Misrouting')
        ax2.set_xticks(x)
        ax2.set_xticklabels(varieties, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels for significant losses
        for i, loss in enumerate(economic_losses):
            if loss > 0.1:
                ax2.text(i, loss + 0.02, f'${loss:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        economic_path = self.output_dir / "economic_impact.png"
        plt.savefig(economic_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Run the apple variety experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run apple variety CTA experiment")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Create and run experiment
    experiment = AppleVarietyExperiment(args.config)
    results = experiment.run()
    
    print(f"Experiment complete. Results saved to: {experiment.output_dir}")


if __name__ == "__main__":
    main()