"""Generate publication-quality figures for apple variety CTA paper.

This script creates all figures needed for the arxiv paper submission,
using results from the apple variety experiment.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Color palette for varieties
VARIETY_COLORS = {
    'Honeycrisp': '#D73502',      # Premium - Red
    'Ambrosia': '#FF6D00',        # Premium - Orange  
    'Zestar!': '#FFA726',         # Premium - Light Orange
    'Buckeye Gala': '#0066CC',   # Standard - Blue
    'Macoun': '#4CAF50',          # Standard - Green
    'Liberty': '#7CB342',         # Standard - Light Green
    'Blondee': '#AB47BC',         # Other - Purple
    'Lindamac McIntosh': '#5E35B1',  # Other - Dark Purple
    'Lindamac': '#7E57C2',        # Other - Medium Purple
    'Akane': '#78909C'            # Other - Gray
}


class FigureGenerator:
    """Generate figures for apple variety CTA paper."""
    
    def __init__(self, results_dir: str, output_dir: str):
        """Initialize figure generator.
        
        Args:
            results_dir: Directory containing experiment results
            output_dir: Directory to save figures
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        
    def _load_results(self) -> Dict[str, Any]:
        """Load experiment results from files."""
        results = {}
        
        # Load main results JSON
        results_file = self.results_dir / "apple_variety_classification_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        # Load model checkpoint for metadata
        model_file = self.results_dir / "model.pth"
        if model_file.exists():
            import torch
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            results['variety_names'] = checkpoint['variety_names']
            results['model_config'] = checkpoint['config']
        
        return results
    
    def generate_all_figures(self):
        """Generate all figures for the paper."""
        print("Generating figures for apple variety CTA paper...")
        
        # Figure 1: Trajectory Flow (Sankey)
        self.create_trajectory_flow_figure()
        
        # Figure 2: Fragmentation Analysis
        self.create_fragmentation_figure()
        
        # Figure 3: Convergence Patterns
        self.create_convergence_figure()
        
        # Figure 4: Economic Impact
        self.create_economic_impact_figure()
        
        # Figure 5: Performance Summary
        self.create_performance_summary_figure()
        
        print(f"All figures saved to: {self.output_dir}")
    
    def create_trajectory_flow_figure(self):
        """Create figure showing trajectory flow through network layers."""
        # This would typically load and enhance the Sankey diagrams
        # For now, we'll create a placeholder showing the concept
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        windows = ['Early (L0→L1)', 'Middle (L1→L2)', 'Late (L2→L3)']
        
        for ax, window in zip(axes, windows):
            # Placeholder visualization
            ax.text(0.5, 0.5, f'{window} Window\nTrajectory Flow\n[SANKEY DIAGRAM]',
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title(window)
        
        plt.suptitle('Concept Trajectory Flow Through Network Layers', fontsize=16)
        plt.tight_layout()
        
        # Save
        self._save_figure(fig, 'figure1_trajectory_flow')
        
    def create_fragmentation_figure(self):
        """Create figure showing fragmentation analysis by variety."""
        # Extract fragmentation data
        variety_metrics = self.results['results']['analysis']['variety_metrics']['by_variety']
        
        varieties = []
        train_frag = []
        test_frag = []
        train_paths = []
        test_paths = []
        
        # Sort varieties by fragmentation rate
        sorted_varieties = sorted(variety_metrics.items(), 
                                key=lambda x: x[1].get('train_fragmentation', 0), 
                                reverse=True)
        
        for variety_name, data in sorted_varieties:
            if 'train_fragmentation' in data:
                varieties.append(variety_name)
                train_frag.append(data['train_fragmentation'])
                test_frag.append(data.get('test_fragmentation', 0))
                train_paths.append(data.get('train_unique_paths', 0))
                test_paths.append(data.get('test_unique_paths', 0))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        x = np.arange(len(varieties))
        width = 0.35
        
        # Fragmentation rates
        bars1 = ax1.bar(x - width/2, train_frag, width, label='Train', alpha=0.8)
        bars2 = ax1.bar(x + width/2, test_frag, width, label='Test', alpha=0.8)
        
        # Color bars by variety type
        for i, (bar1, bar2, variety) in enumerate(zip(bars1, bars2, varieties)):
            color = VARIETY_COLORS.get(variety, '#666666')
            bar1.set_color(color)
            bar2.set_color(color)
            bar2.set_alpha(0.6)
        
        ax1.set_ylabel('Fragmentation Rate')
        ax1.set_title('Trajectory Fragmentation by Apple Variety')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Add significance markers for high fragmentation
        for i, (train, test) in enumerate(zip(train_frag, test_frag)):
            if train > 0.5:  # High fragmentation threshold
                ax1.text(i, train + 0.02, '*', ha='center', fontsize=14)
        
        # Unique paths
        bars3 = ax2.bar(x - width/2, train_paths, width, label='Train', alpha=0.8)
        bars4 = ax2.bar(x + width/2, test_paths, width, label='Test', alpha=0.8)
        
        # Color bars
        for i, (bar3, bar4, variety) in enumerate(zip(bars3, bars4, varieties)):
            color = VARIETY_COLORS.get(variety, '#666666')
            bar3.set_color(color)
            bar4.set_color(color)
            bar4.set_alpha(0.6)
        
        ax2.set_ylabel('Number of Unique Paths')
        ax2.set_xlabel('Apple Variety')
        ax2.set_xticks(x)
        ax2.set_xticklabels(varieties, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        self._save_figure(fig, 'figure2_fragmentation_analysis')
    
    def create_convergence_figure(self):
        """Create figure showing convergence patterns between varieties."""
        convergence_data = self.results['results']['analysis']['variety_metrics']['convergence']
        
        # Create convergence matrix
        varieties = self.results['variety_names']
        n_varieties = len(varieties)
        convergence_matrix = np.zeros((n_varieties, n_varieties))
        
        # Fill matrix from convergence points
        for conv_point in convergence_data['convergence_points']:
            layer = conv_point['layer']
            for pair_key, pair_data in conv_point['convergences'].items():
                v1, v2 = pair_key.split('-')
                try:
                    idx1 = varieties.index(v1)
                    idx2 = varieties.index(v2)
                    overlap = pair_data['overlap_ratio']
                    convergence_matrix[idx1, idx2] += overlap
                    convergence_matrix[idx2, idx1] += overlap
                except ValueError:
                    continue
        
        # Normalize by number of layers
        convergence_matrix /= 3  # 3 layer transitions
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mask for diagonal
        mask = np.eye(n_varieties, dtype=bool)
        
        # Plot heatmap
        sns.heatmap(convergence_matrix, 
                   xticklabels=varieties,
                   yticklabels=varieties,
                   cmap='YlOrRd',
                   vmin=0, vmax=1,
                   mask=mask,
                   square=True,
                   cbar_kws={'label': 'Convergence Score'},
                   annot=True,
                   fmt='.2f',
                   ax=ax)
        
        ax.set_title('Variety Convergence Patterns in Processing Pathways')
        
        # Highlight premium varieties
        premium_varieties = ['Honeycrisp', 'Ambrosia', 'Zestar!']
        for i, variety in enumerate(varieties):
            if variety in premium_varieties:
                ax.get_xticklabels()[i].set_weight('bold')
                ax.get_xticklabels()[i].set_color('red')
                ax.get_yticklabels()[i].set_weight('bold')
                ax.get_yticklabels()[i].set_color('red')
        
        plt.tight_layout()
        
        # Save
        self._save_figure(fig, 'figure3_convergence_patterns')
    
    def create_economic_impact_figure(self):
        """Create figure showing economic impact of misclassification."""
        # Get variety prices from config
        variety_prices = self.results['model_config']['analysis']['economic']['variety_prices']
        
        # Create price difference matrix
        varieties = self.results['variety_names']
        n_varieties = len(varieties)
        price_impact_matrix = np.zeros((n_varieties, n_varieties))
        
        for i, v1 in enumerate(varieties):
            for j, v2 in enumerate(varieties):
                if i != j:
                    price1 = variety_prices.get(v1, 1.0)
                    price2 = variety_prices.get(v2, 1.0)
                    # Economic loss if v1 is misclassified as v2
                    price_impact_matrix[i, j] = max(0, price1 - price2)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Subplot 1: Price comparison
        sorted_varieties = sorted(varieties, 
                                key=lambda x: variety_prices.get(x, 1.0), 
                                reverse=True)
        prices = [variety_prices.get(v, 1.0) for v in sorted_varieties]
        
        bars = ax1.barh(sorted_varieties, prices)
        
        # Color bars
        for bar, variety in zip(bars, sorted_varieties):
            color = VARIETY_COLORS.get(variety, '#666666')
            bar.set_color(color)
        
        ax1.set_xlabel('Price ($/lb)')
        ax1.set_title('Apple Variety Prices')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add price labels
        for i, (variety, price) in enumerate(zip(sorted_varieties, prices)):
            ax1.text(price + 0.02, i, f'${price:.2f}', va='center')
        
        # Subplot 2: Economic impact heatmap
        sns.heatmap(price_impact_matrix,
                   xticklabels=varieties,
                   yticklabels=varieties,
                   cmap='Reds',
                   vmin=0,
                   square=True,
                   cbar_kws={'label': 'Economic Loss ($/lb)'},
                   ax=ax2)
        
        ax2.set_title('Economic Impact of Misclassification')
        ax2.set_xlabel('Predicted Variety')
        ax2.set_ylabel('True Variety')
        
        plt.suptitle('Economic Analysis of Apple Variety Classification')
        plt.tight_layout()
        
        # Save
        self._save_figure(fig, 'figure4_economic_impact')
    
    def create_performance_summary_figure(self):
        """Create figure summarizing model performance."""
        # Extract performance metrics
        test_accuracy = self.results['results']['test_accuracy']
        train_accuracies = self.results['results']['train_accuracies']
        train_losses = self.results['results']['train_losses']
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training curves
        epochs = range(1, len(train_accuracies) + 1)
        ax1.plot(epochs, train_accuracies, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Training Accuracy')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)
        
        ax2.plot(epochs, train_losses, 'r-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Variety-specific performance
        variety_metrics = self.results['results']['analysis']['variety_metrics']['by_variety']
        
        varieties = []
        fragmentations = []
        sample_counts = []
        
        for variety_name, data in variety_metrics.items():
            if 'train_fragmentation' in data:
                varieties.append(variety_name)
                fragmentations.append(data['train_fragmentation'])
                sample_counts.append(data['train_samples'])
        
        # Scatter plot: fragmentation vs sample count
        scatter = ax3.scatter(sample_counts, fragmentations, s=100, alpha=0.7)
        
        # Color by variety
        for i, variety in enumerate(varieties):
            color = VARIETY_COLORS.get(variety, '#666666')
            ax3.scatter(sample_counts[i], fragmentations[i], 
                       s=100, color=color, label=variety)
        
        ax3.set_xlabel('Number of Training Samples')
        ax3.set_ylabel('Fragmentation Rate')
        ax3.set_title('Sample Size vs. Fragmentation')
        ax3.grid(True, alpha=0.3)
        
        # Add variety labels for outliers
        for i, variety in enumerate(varieties):
            if fragmentations[i] > 0.6 or sample_counts[i] < 20:
                ax3.annotate(variety, (sample_counts[i], fragmentations[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        # Summary statistics
        ax4.axis('off')
        summary_text = f"""Model Performance Summary
        
Test Accuracy: {test_accuracy:.1f}%
Final Training Accuracy: {train_accuracies[-1]:.1f}%

Total Varieties: {len(varieties)}
Avg. Fragmentation: {np.mean(fragmentations):.3f}
Max Fragmentation: {max(fragmentations):.3f}

High Fragmentation Varieties (>0.5):
"""
        high_frag_varieties = [(v, f) for v, f in zip(varieties, fragmentations) if f > 0.5]
        high_frag_varieties.sort(key=lambda x: x[1], reverse=True)
        
        for variety, frag in high_frag_varieties[:5]:
            summary_text += f"\n  • {variety}: {frag:.3f}"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.suptitle('Apple Variety Classification Performance Summary', fontsize=16)
        plt.tight_layout()
        
        # Save
        self._save_figure(fig, 'figure5_performance_summary')
    
    def _save_figure(self, fig, name: str):
        """Save figure in multiple formats."""
        # Save as PNG
        png_path = self.output_dir / f"{name}.png"
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {png_path}")
        
        # Save as PDF
        pdf_path = self.output_dir / f"{name}.pdf"
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved: {pdf_path}")
        
        plt.close(fig)


def main():
    """Generate all figures for the paper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate figures for apple variety CTA paper")
    parser.add_argument('--results-dir', 
                       default='results/apple_variety',
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir',
                       default='../../arxiv_apple/figures',
                       help='Directory to save figures')
    
    args = parser.parse_args()
    
    # Create figure generator and generate all figures
    generator = FigureGenerator(args.results_dir, args.output_dir)
    generator.generate_all_figures()


if __name__ == "__main__":
    main()