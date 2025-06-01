"""
Test data generators for Concept MRI testing.
Creates realistic mock data for testing all components.
"""
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime


class TestDataGenerator:
    """Generate test data for Concept MRI components."""
    
    @staticmethod
    def generate_model_data(
        num_layers: int = 12,
        layer_size: int = 768,
        num_samples: int = 100,
        model_type: str = "feedforward"
    ) -> Dict[str, Any]:
        """Generate mock model data with activations."""
        activations = {}
        
        for i in range(num_layers):
            layer_name = f"layer_{i}"
            # Generate random activations with some structure
            base_activations = np.random.randn(num_samples, layer_size) * 0.1
            
            # Add some cluster structure
            for j in range(5):  # 5 natural clusters
                cluster_mask = np.random.choice(num_samples, size=num_samples//5, replace=False)
                cluster_center = np.random.randn(layer_size) * 0.5
                base_activations[cluster_mask] += cluster_center
            
            activations[layer_name] = base_activations
        
        return {
            'model_id': f'{model_type}_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'model_loaded': True,
            'model_type': model_type,
            'num_layers': num_layers,
            'layer_sizes': {f'layer_{i}': layer_size for i in range(num_layers)},
            'activations': activations,
            'dataset': {
                'name': 'test_dataset',
                'num_samples': num_samples,
                'num_features': 100
            },
            'analysis_configured': True
        }
    
    @staticmethod
    def generate_clustering_data(
        model_data: Dict[str, Any],
        algorithm: str = "kmeans",
        hierarchy_level: str = "meso"
    ) -> Dict[str, Any]:
        """Generate mock clustering results."""
        num_layers = model_data['num_layers']
        num_samples = model_data['dataset']['num_samples']
        
        # Determine number of clusters based on hierarchy
        k_values = {
            'macro': 5,
            'meso': 10,
            'micro': 20
        }
        k = k_values.get(hierarchy_level, 10)
        
        clusters_per_layer = {}
        
        for i in range(num_layers):
            layer_name = f'layer_{i}'
            
            # Generate cluster labels with some continuity
            if i == 0:
                labels = np.random.randint(0, k, num_samples)
            else:
                # Make labels somewhat continuous across layers
                prev_labels = clusters_per_layer[f'layer_{i-1}']['labels']
                labels = prev_labels.copy()
                # Random transitions
                transition_mask = np.random.random(num_samples) < 0.2
                labels[transition_mask] = np.random.randint(0, k, np.sum(transition_mask))
            
            # Calculate mock statistics
            unique_labels = np.unique(labels)
            cluster_sizes = np.bincount(labels)
            
            clusters_per_layer[layer_name] = {
                'labels': labels.tolist(),
                'n_clusters': len(unique_labels),
                'algorithm': algorithm,
                'silhouette_score': np.random.uniform(0.6, 0.9),
                'cluster_sizes': cluster_sizes.tolist()
            }
            
            # Add ETS-specific data if using ETS
            if algorithm == 'ets':
                clusters_per_layer[layer_name]['thresholds'] = np.random.uniform(0.01, 0.1, 768).tolist()
                clusters_per_layer[layer_name]['statistics'] = {
                    'n_clusters': len(unique_labels),
                    'cluster_sizes': {
                        'min': int(np.min(cluster_sizes)),
                        'max': int(np.max(cluster_sizes)),
                        'mean': float(np.mean(cluster_sizes))
                    },
                    'active_dimensions': {
                        'per_cluster': np.random.randint(5, 20, len(unique_labels)).tolist()
                    },
                    'dimension_importance': {
                        str(j): np.random.random() for j in range(20)
                    }
                }
        
        return {
            'algorithm': algorithm,
            'hierarchy': hierarchy_level,
            'completed': True,
            'timestamp': datetime.now().isoformat(),
            'clusters_per_layer': clusters_per_layer,
            'metrics': {
                'total_clusters': k * num_layers,
                'unique_paths': 150,
                'fragmentation': 0.23,
                'stability': 0.87
            }
        }
    
    @staticmethod
    def generate_window_config(num_layers: int) -> Dict[str, Any]:
        """Generate window configuration."""
        if num_layers >= 12:
            # GPT-2 style windows
            windows = {
                'Early': {'start': 0, 'end': 3, 'color': '#1f77b4'},
                'Middle': {'start': 4, 'end': 7, 'color': '#ff7f0e'},
                'Late': {'start': 8, 'end': 11, 'color': '#2ca02c'}
            }
        elif num_layers >= 6:
            # Halves
            mid = num_layers // 2
            windows = {
                'First Half': {'start': 0, 'end': mid - 1, 'color': '#1f77b4'},
                'Second Half': {'start': mid, 'end': num_layers - 1, 'color': '#ff7f0e'}
            }
        else:
            # Single window
            windows = {
                'All': {'start': 0, 'end': num_layers - 1, 'color': '#1f77b4'}
            }
        
        return {
            'windows': windows,
            'mode': 'preset',
            'preset': 'gpt2' if num_layers >= 12 else 'halves'
        }
    
    @staticmethod
    def generate_path_analysis(
        clustering_data: Dict[str, Any],
        top_n: int = 50
    ) -> Dict[str, Any]:
        """Generate mock path analysis results."""
        paths = []
        
        # Generate some common paths
        for i in range(top_n):
            path_length = np.random.randint(3, 8)
            path = []
            
            for j in range(path_length):
                layer_idx = min(j * 2, 11)  # Sample layers
                cluster_id = np.random.randint(0, 10)
                path.append({
                    'layer': f'layer_{layer_idx}',
                    'cluster': cluster_id
                })
            
            paths.append({
                'id': f'path_{i}',
                'transitions': path,
                'frequency': np.random.randint(5, 50),
                'stability': np.random.uniform(0.7, 0.95),
                'samples': list(np.random.choice(100, size=np.random.randint(5, 20), replace=False))
            })
        
        return {
            'paths': paths,
            'analysis_date': datetime.now().isoformat(),
            'parameters': {
                'min_frequency': 5,
                'min_stability': 0.7
            }
        }
    
    @staticmethod
    def generate_cluster_labels(
        clustering_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate mock LLM-generated cluster labels."""
        labels = {}
        
        # Generate labels for each layer
        for layer_name, layer_data in clustering_data['clusters_per_layer'].items():
            layer_labels = {}
            n_clusters = layer_data['n_clusters']
            
            # Create semantic labels
            label_templates = [
                "Syntactic structure",
                "Semantic content", 
                "Positional encoding",
                "Entity recognition",
                "Relationship mapping",
                "Abstract concepts",
                "Concrete objects",
                "Action verbs",
                "Descriptive attributes",
                "Numerical processing"
            ]
            
            for i in range(n_clusters):
                base_label = label_templates[i % len(label_templates)]
                layer_labels[str(i)] = f"{base_label} (L{layer_name.split('_')[1]})"
            
            labels[layer_name] = layer_labels
        
        return {
            'labels': labels,
            'generation_date': datetime.now().isoformat(),
            'model': 'gpt-4',
            'confidence': 0.85
        }
    
    @staticmethod
    def generate_hierarchy_results(
        model_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate hierarchical clustering results."""
        results = {}
        
        for level in ['macro', 'meso', 'micro']:
            results[level] = TestDataGenerator.generate_clustering_data(
                model_data,
                algorithm='kmeans',
                hierarchy_level=level
            )
        
        return results
    
    @staticmethod
    def generate_complete_test_state(
        num_layers: int = 12,
        include_ets: bool = True
    ) -> Dict[str, Any]:
        """Generate a complete test state with all data."""
        # Generate base data
        model_data = TestDataGenerator.generate_model_data(num_layers=num_layers)
        clustering_data = TestDataGenerator.generate_clustering_data(
            model_data,
            algorithm='ets' if include_ets else 'kmeans'
        )
        window_config = TestDataGenerator.generate_window_config(num_layers)
        path_analysis = TestDataGenerator.generate_path_analysis(clustering_data)
        cluster_labels = TestDataGenerator.generate_cluster_labels(clustering_data)
        hierarchy_results = TestDataGenerator.generate_hierarchy_results(model_data)
        
        return {
            'model-store': model_data,
            'clustering-store': clustering_data,
            'window-config-store': window_config,
            'path-analysis-store': path_analysis,
            'cluster-labels-store': cluster_labels,
            'hierarchy-results-store': hierarchy_results
        }


def save_test_data(filename: str = "test_data.json"):
    """Save test data to file for manual testing."""
    test_state = TestDataGenerator.generate_complete_test_state()
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_arrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item) for item in obj]
        return obj
    
    test_state = convert_arrays(test_state)
    
    with open(filename, 'w') as f:
        json.dump(test_state, f, indent=2)
    
    print(f"Test data saved to {filename}")


if __name__ == "__main__":
    # Generate and save test data
    save_test_data("concept_mri_test_data.json")