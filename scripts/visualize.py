"""
Visualization Module for Carbon Footprint AI Framework
Generates publication-quality plots, charts, and tables
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Try to import plotly for interactive charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from config import (
    FIGURES_DIR, TABLES_DIR, RESULTS_DIR, 
    COLOR_PALETTE, PLOT_STYLE, CARBON_INTENSITY
)
from metrics import ModelMetrics, MetricsProcessor


# Apply plot style
plt.rcParams.update(PLOT_STYLE)
sns.set_style("whitegrid")


class Visualizer:
    """
    Generate all visualizations for the Carbon Footprint AI Framework.
    """
    
    def __init__(
        self,
        metrics_processor: MetricsProcessor = None,
        output_dir: Path = FIGURES_DIR,
        table_dir: Path = TABLES_DIR
    ):
        self.processor = metrics_processor or MetricsProcessor()
        self.output_dir = output_dir
        self.table_dir = table_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.colors = COLOR_PALETTE["models"]
    
    def load_results(self, results_dir: Path = RESULTS_DIR):
        """Load results from JSON files"""
        self.processor.load_from_results(results_dir)
    
    # ========================================================================
    # BAR CHARTS
    # ========================================================================
    def plot_co2_by_model(self, save: bool = True) -> plt.Figure:
        """
        Bar chart: CO2 emissions by model.
        """
        models = self.processor.models
        if not models:
            return None
        
        # Sort by CO2
        models_sorted = sorted(models, key=lambda x: x.training_co2_kg)
        names = [m.model_name for m in models_sorted]
        co2_values = [m.training_co2_kg * 1000 for m in models_sorted]  # Convert to grams
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(names, co2_values, color=self.colors[:len(names)])
        
        # Add value labels
        for bar, val in zip(bars, co2_values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}g', va='center', fontsize=10)
        
        ax.set_xlabel('CO₂ Emissions (g)', fontsize=12)
        ax.set_title('Carbon Emissions by Model (Training)', fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'co2_by_model.png', dpi=150, bbox_inches='tight')
            fig.savefig(self.output_dir / 'co2_by_model.svg', bbox_inches='tight')
        
        return fig
    
    def plot_accuracy_comparison(self, save: bool = True) -> plt.Figure:
        """
        Bar chart: Accuracy comparison across models.
        """
        models = self.processor.models
        if not models:
            return None
        
        models_sorted = sorted(models, key=lambda x: x.accuracy, reverse=True)
        names = [m.model_name for m in models_sorted]
        accuracies = [m.accuracy for m in models_sorted]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(names, accuracies, color=self.colors[:len(names)])
        
        # Add value labels
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', fontsize=10)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        plt.xticks(rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'accuracy_comparison.png', dpi=150, bbox_inches='tight')
            fig.savefig(self.output_dir / 'accuracy_comparison.svg', bbox_inches='tight')
        
        return fig
    
    def plot_green_ai_scores(self, save: bool = True) -> plt.Figure:
        """
        Bar chart: Green AI Score ranking.
        """
        models = self.processor.models
        if not models:
            return None
        
        models_sorted = sorted(models, key=lambda x: x.green_ai_score, reverse=True)
        names = [m.model_name for m in models_sorted]
        scores = [m.green_ai_score for m in models_sorted]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(names, scores, color=self.colors[:len(names)])
        
        for bar, val in zip(bars, scores):
            ax.text(bar.get_width() + max(scores)*0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}', va='center', fontsize=10)
        
        ax.set_xlabel('Green AI Score (Accuracy / CO₂)', fontsize=12)
        ax.set_title('Green AI Score Leaderboard', fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'green_ai_scores.png', dpi=150, bbox_inches='tight')
            fig.savefig(self.output_dir / 'green_ai_scores.svg', bbox_inches='tight')
        
        return fig
    
    # ========================================================================
    # SCATTER PLOTS
    # ========================================================================
    def plot_accuracy_vs_co2(self, save: bool = True) -> plt.Figure:
        """
        Scatter plot: Accuracy vs CO2 emissions with Pareto frontier.
        """
        models = self.processor.models
        if not models:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot points
        for i, m in enumerate(models):
            ax.scatter(m.training_co2_kg * 1000, m.accuracy, 
                      s=150, c=[self.colors[i % len(self.colors)]], 
                      label=m.model_name, zorder=5)
            ax.annotate(m.model_name, (m.training_co2_kg * 1000, m.accuracy),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Draw Pareto frontier
        pareto_models = [m for m in models if m.model_name in 
                        self.processor.get_aggregate_summary().get('pareto_efficient', [])]
        if len(pareto_models) > 1:
            pareto_sorted = sorted(pareto_models, key=lambda x: x.training_co2_kg)
            pareto_x = [m.training_co2_kg * 1000 for m in pareto_sorted]
            pareto_y = [m.accuracy for m in pareto_sorted]
            ax.plot(pareto_x, pareto_y, 'g--', linewidth=2, label='Pareto Frontier', alpha=0.7)
        
        ax.set_xlabel('CO₂ Emissions (g)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy vs Carbon Emissions Trade-off', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add annotation for ideal corner
        ax.annotate('Ideal\n(High Acc, Low CO₂)', 
                   xy=(ax.get_xlim()[0], ax.get_ylim()[1]),
                   xytext=(10, -30), textcoords='offset points',
                   fontsize=10, color='green', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'accuracy_vs_co2.png', dpi=150, bbox_inches='tight')
            fig.savefig(self.output_dir / 'accuracy_vs_co2.svg', bbox_inches='tight')
        
        return fig
    
    def plot_parameters_vs_accuracy(self, save: bool = True) -> plt.Figure:
        """
        Scatter plot: Model parameters vs accuracy with CO2 as bubble size.
        """
        models = self.processor.models
        if not models:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        params = [m.num_parameters / 1e6 for m in models]
        accuracies = [m.accuracy for m in models]
        co2_sizes = [m.training_co2_kg * 5000 + 50 for m in models]  # Scale for visibility
        
        scatter = ax.scatter(params, accuracies, s=co2_sizes, 
                            c=[self.colors[i % len(self.colors)] for i in range(len(models))],
                            alpha=0.7, edgecolors='black', linewidth=1)
        
        for i, m in enumerate(models):
            ax.annotate(m.model_name, (params[i], accuracies[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Parameters (Millions)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Model Size vs Accuracy\n(Bubble size = CO₂ emissions)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'params_vs_accuracy.png', dpi=150, bbox_inches='tight')
            fig.savefig(self.output_dir / 'params_vs_accuracy.svg', bbox_inches='tight')
        
        return fig
    
    # ========================================================================
    # LINE CHARTS
    # ========================================================================
    def plot_training_progress(
        self,
        epoch_data: Dict[str, List[Dict]],
        metric: str = "accuracy",
        save: bool = True
    ) -> plt.Figure:
        """
        Line chart: Training progress over epochs for multiple models.
        
        Args:
            epoch_data: Dict mapping model_name -> list of epoch metrics
            metric: Metric to plot ('accuracy', 'loss', 'energy', 'co2')
        """
        if not epoch_data:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (model_name, epochs) in enumerate(epoch_data.items()):
            x = [e.get('epoch', i+1) for i, e in enumerate(epochs)]
            y = [e.get(metric, 0) for e in epochs]
            ax.plot(x, y, marker='o', linewidth=2, markersize=6,
                   label=model_name, color=self.colors[i % len(self.colors)])
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Training {metric.title()} Over Epochs', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'training_{metric}_progress.png', dpi=150, bbox_inches='tight')
            fig.savefig(self.output_dir / f'training_{metric}_progress.svg', bbox_inches='tight')
        
        return fig
    
    def plot_energy_over_time(
        self,
        energy_data: Dict[str, List[float]],
        save: bool = True
    ) -> plt.Figure:
        """
        Line chart: Cumulative energy consumption over training.
        """
        if not energy_data:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (model_name, energies) in enumerate(energy_data.items()):
            cumulative = np.cumsum(energies)
            x = range(1, len(cumulative) + 1)
            ax.plot(x, cumulative * 1000, marker='o', linewidth=2,
                   label=model_name, color=self.colors[i % len(self.colors)])
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Cumulative Energy (Wh)', fontsize=12)
        ax.set_title('Energy Consumption During Training', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'energy_over_time.png', dpi=150, bbox_inches='tight')
            fig.savefig(self.output_dir / 'energy_over_time.svg', bbox_inches='tight')
        
        return fig
    
    # ========================================================================
    # HEATMAPS
    # ========================================================================
    def plot_optimization_heatmap(self, save: bool = True) -> plt.Figure:
        """
        Heatmap: Optimization technique effectiveness.
        """
        models = self.processor.models
        
        # Filter to get baseline and optimized models
        baseline = [m for m in models if m.optimization_type == "none"]
        optimized = [m for m in models if m.optimization_type != "none"]
        
        if not optimized:
            return None
        
        # Create effectiveness matrix
        opt_types = list(set(m.optimization_type for m in optimized))
        metrics = ['Accuracy Change (%)', 'CO2 Reduction (%)', 'Size Reduction (%)', 'Speed Improvement (%)']
        
        data = np.zeros((len(opt_types), len(metrics)))
        
        for i, opt_type in enumerate(opt_types):
            opt_models = [m for m in optimized if m.optimization_type == opt_type]
            if opt_models and baseline:
                base = baseline[0]  # Use first baseline
                opt = opt_models[0]
                
                data[i, 0] = opt.accuracy - base.accuracy
                data[i, 1] = ((base.training_co2_kg - opt.training_co2_kg) / base.training_co2_kg * 100) if base.training_co2_kg > 0 else 0
                data[i, 2] = ((base.model_size_mb - opt.model_size_mb) / base.model_size_mb * 100) if base.model_size_mb > 0 else 0
                data[i, 3] = 0  # Placeholder for speed improvement
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   xticklabels=metrics, yticklabels=opt_types, ax=ax,
                   cbar_kws={'label': 'Improvement (%)'})
        
        ax.set_title('Optimization Technique Effectiveness', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'optimization_heatmap.png', dpi=150, bbox_inches='tight')
            fig.savefig(self.output_dir / 'optimization_heatmap.svg', bbox_inches='tight')
        
        return fig
    
    # ========================================================================
    # REGIONAL ANALYSIS
    # ========================================================================
    def plot_regional_comparison(self, save: bool = True) -> plt.Figure:
        """
        Grouped bar chart: CO2 emissions by region.
        """
        models = self.processor.models
        if not models:
            return None
        
        regions = list(CARBON_INTENSITY.keys())
        region_colors = [COLOR_PALETTE["regions"].get(r, "#666666") for r in regions]
        
        x = np.arange(len(models))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for i, region in enumerate(regions):
            values = [m.regional_co2.get(region, 0) * 1000 for m in models]
            bars = ax.bar(x + i * width, values, width, label=region.title(),
                         color=region_colors[i])
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('CO₂ Emissions (g)', fontsize=12)
        ax.set_title('Regional Carbon Emissions Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(regions) - 1) / 2)
        ax.set_xticklabels([m.model_name for m in models], rotation=45, ha='right')
        ax.legend(title='Region', loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'regional_comparison.png', dpi=150, bbox_inches='tight')
            fig.savefig(self.output_dir / 'regional_comparison.svg', bbox_inches='tight')
        
        return fig
    
    # ========================================================================
    # RADAR CHARTS
    # ========================================================================
    def plot_model_radar(self, model_names: List[str] = None, save: bool = True) -> plt.Figure:
        """
        Radar chart: Multi-metric model comparison.
        """
        models = self.processor.models
        if not models:
            return None
        
        if model_names:
            models = [m for m in models if m.model_name in model_names]
        
        if len(models) < 2:
            return None
        
        # Normalize metrics to 0-1 scale
        categories = ['Accuracy', 'Green AI Score', 'Size Efficiency', 'Energy Efficiency', 'Speed']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        # Calculate normalized values
        max_accuracy = max(m.accuracy for m in models)
        max_green_score = max(m.green_ai_score for m in models)
        max_params = max(m.num_parameters for m in models)
        max_energy = max(m.training_energy_kwh for m in models) or 1
        
        for i, m in enumerate(models):
            values = [
                m.accuracy / max_accuracy if max_accuracy > 0 else 0,
                m.green_ai_score / max_green_score if max_green_score > 0 else 0,
                1 - (m.num_parameters / max_params) if max_params > 0 else 0,
                1 - (m.training_energy_kwh / max_energy) if max_energy > 0 else 0,
                0.5  # Placeholder for speed
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=m.model_name,
                   color=self.colors[i % len(self.colors)])
            ax.fill(angles, values, alpha=0.25, color=self.colors[i % len(self.colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_title('Multi-Metric Model Comparison', fontsize=14, fontweight='bold', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'model_radar.png', dpi=150, bbox_inches='tight')
            fig.savefig(self.output_dir / 'model_radar.svg', bbox_inches='tight')
        
        return fig
    
    # ========================================================================
    # PIE / DONUT CHARTS
    # ========================================================================
    def plot_energy_breakdown(self, save: bool = True) -> plt.Figure:
        """
        Stacked bar chart: Energy breakdown (GPU vs CPU).
        """
        models = self.processor.models
        if not models:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = [m.model_name for m in models]
        # Estimate GPU as 80% of total energy (typical for ML workloads)
        gpu_energy = [m.training_energy_kwh * 0.8 * 1000 for m in models]
        cpu_energy = [m.training_energy_kwh * 0.2 * 1000 for m in models]
        
        x = np.arange(len(names))
        
        ax.bar(x, gpu_energy, label='GPU', color='#E63946')
        ax.bar(x, cpu_energy, bottom=gpu_energy, label='CPU', color='#457B9D')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Energy (Wh)', fontsize=12)
        ax.set_title('Energy Consumption Breakdown', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'energy_breakdown.png', dpi=150, bbox_inches='tight')
            fig.savefig(self.output_dir / 'energy_breakdown.svg', bbox_inches='tight')
        
        return fig
    
    # ========================================================================
    # BOX PLOTS
    # ========================================================================
    def plot_gpu_utilization_box(
        self,
        utilization_data: Dict[str, List[float]],
        save: bool = True
    ) -> plt.Figure:
        """
        Box plot: GPU utilization distribution.
        """
        if not utilization_data:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = [utilization_data[name] for name in utilization_data]
        labels = list(utilization_data.keys())
        
        bp = ax.boxplot(data, patch_artist=True, labels=labels)
        
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(self.colors[i % len(self.colors)])
            patch.set_alpha(0.7)
        
        ax.set_ylabel('GPU Utilization (%)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title('GPU Utilization Distribution During Training', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'gpu_utilization_box.png', dpi=150, bbox_inches='tight')
            fig.savefig(self.output_dir / 'gpu_utilization_box.svg', bbox_inches='tight')
        
        return fig
    
    # ========================================================================
    # TABLE GENERATION
    # ========================================================================
    def save_comparison_table(self, styled: bool = True) -> Path:
        """
        Save the main comparison table as CSV and styled HTML.
        """
        df = self.processor.get_comparison_table()
        
        # Save CSV
        csv_path = self.table_dir / 'comparison_table.csv'
        df.to_csv(csv_path, index=False)
        
        if styled:
            # Create styled HTML
            styled_df = df.style.background_gradient(
                subset=['Accuracy (%)', 'Green AI Score'],
                cmap='Greens'
            ).background_gradient(
                subset=['CO2 (g)', 'Energy (Wh)'],
                cmap='Reds_r'
            ).format({
                'Accuracy (%)': '{:.2f}',
                'Parameters (M)': '{:.2f}',
                'Size (MB)': '{:.2f}',
                'Training Time (min)': '{:.2f}',
                'Energy (Wh)': '{:.4f}',
                'CO2 (g)': '{:.4f}',
                'Green AI Score': '{:.2f}',
                'CEI': '{:.2f}'
            })
            
            html_path = self.table_dir / 'comparison_table.html'
            styled_df.to_html(html_path)
        
        return csv_path
    
    def save_regional_table(self) -> Path:
        """Save regional comparison table"""
        df = self.processor.get_regional_comparison()
        csv_path = self.table_dir / 'regional_comparison.csv'
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def save_leaderboard_table(self) -> Path:
        """Save Green AI Score leaderboard"""
        df = self.processor.get_leaderboard()
        csv_path = self.table_dir / 'green_ai_leaderboard.csv'
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def save_optimization_table(self) -> Path:
        """Save optimization comparison table"""
        df = self.processor.get_optimization_comparison()
        if df.empty:
            return None
        csv_path = self.table_dir / 'optimization_comparison.csv'
        df.to_csv(csv_path, index=False)
        return csv_path
    
    # ========================================================================
    # GENERATE ALL
    # ========================================================================
    def generate_all_visualizations(
        self,
        epoch_data: Dict[str, List[Dict]] = None,
        energy_data: Dict[str, List[float]] = None,
        utilization_data: Dict[str, List[float]] = None
    ) -> Dict[str, Path]:
        """
        Generate all visualizations and tables.
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        generated = {}
        
        print("Generating visualizations...")
        
        # Bar charts
        print("  - CO2 by model...")
        if self.plot_co2_by_model():
            generated['co2_by_model'] = self.output_dir / 'co2_by_model.png'
        
        print("  - Accuracy comparison...")
        if self.plot_accuracy_comparison():
            generated['accuracy_comparison'] = self.output_dir / 'accuracy_comparison.png'
        
        print("  - Green AI scores...")
        if self.plot_green_ai_scores():
            generated['green_ai_scores'] = self.output_dir / 'green_ai_scores.png'
        
        # Scatter plots
        print("  - Accuracy vs CO2...")
        if self.plot_accuracy_vs_co2():
            generated['accuracy_vs_co2'] = self.output_dir / 'accuracy_vs_co2.png'
        
        print("  - Parameters vs Accuracy...")
        if self.plot_parameters_vs_accuracy():
            generated['params_vs_accuracy'] = self.output_dir / 'params_vs_accuracy.png'
        
        # Line charts
        if epoch_data:
            print("  - Training progress...")
            if self.plot_training_progress(epoch_data, 'accuracy'):
                generated['training_accuracy'] = self.output_dir / 'training_accuracy_progress.png'
        
        if energy_data:
            print("  - Energy over time...")
            if self.plot_energy_over_time(energy_data):
                generated['energy_over_time'] = self.output_dir / 'energy_over_time.png'
        
        # Heatmap
        print("  - Optimization heatmap...")
        if self.plot_optimization_heatmap():
            generated['optimization_heatmap'] = self.output_dir / 'optimization_heatmap.png'
        
        # Regional comparison
        print("  - Regional comparison...")
        if self.plot_regional_comparison():
            generated['regional_comparison'] = self.output_dir / 'regional_comparison.png'
        
        # Radar chart
        print("  - Model radar...")
        if self.plot_model_radar():
            generated['model_radar'] = self.output_dir / 'model_radar.png'
        
        # Energy breakdown
        print("  - Energy breakdown...")
        if self.plot_energy_breakdown():
            generated['energy_breakdown'] = self.output_dir / 'energy_breakdown.png'
        
        # Box plots
        if utilization_data:
            print("  - GPU utilization box plot...")
            if self.plot_gpu_utilization_box(utilization_data):
                generated['gpu_utilization_box'] = self.output_dir / 'gpu_utilization_box.png'
        
        # Tables
        print("\nGenerating tables...")
        print("  - Comparison table...")
        generated['comparison_table'] = self.save_comparison_table()
        
        print("  - Regional table...")
        generated['regional_table'] = self.save_regional_table()
        
        print("  - Leaderboard...")
        generated['leaderboard'] = self.save_leaderboard_table()
        
        print("  - Optimization table...")
        opt_table = self.save_optimization_table()
        if opt_table:
            generated['optimization_table'] = opt_table
        
        print(f"\nGenerated {len(generated)} visualizations and tables.")
        return generated


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("Carbon Footprint AI Framework - Visualization Module")
    print("=" * 60)
    
    # Create sample data for testing
    processor = MetricsProcessor()
    
    # Add sample models
    processor.add_model_result(
        model_name="SimpleCNN",
        architecture="simple_cnn",
        dataset="cifar10",
        accuracy=75.5,
        loss=0.85,
        num_parameters=62000,
        model_size_mb=0.24,
        training_time_seconds=300,
        training_energy_kwh=0.0125,
        training_co2_kg=0.0103,
        optimization_type="none"
    )
    
    processor.add_model_result(
        model_name="ResNet-18",
        architecture="resnet18",
        dataset="cifar10",
        accuracy=92.3,
        loss=0.35,
        num_parameters=11200000,
        model_size_mb=42.7,
        training_time_seconds=1800,
        training_energy_kwh=0.075,
        training_co2_kg=0.0615,
        optimization_type="none"
    )
    
    processor.add_model_result(
        model_name="MobileNetV2",
        architecture="mobilenetv2",
        dataset="cifar10",
        accuracy=88.7,
        loss=0.45,
        num_parameters=2200000,
        model_size_mb=8.5,
        training_time_seconds=1200,
        training_energy_kwh=0.05,
        training_co2_kg=0.041,
        optimization_type="none"
    )
    
    processor.add_model_result(
        model_name="ResNet-18-Pruned",
        architecture="resnet18_pruned",
        dataset="cifar10",
        accuracy=90.1,
        loss=0.42,
        num_parameters=5600000,
        model_size_mb=21.4,
        training_time_seconds=600,
        training_energy_kwh=0.025,
        training_co2_kg=0.0205,
        optimization_type="pruned"
    )
    
    # Create visualizer and generate all plots
    viz = Visualizer(metrics_processor=processor)
    
    # Generate sample epoch data
    epoch_data = {
        "SimpleCNN": [{"epoch": i+1, "accuracy": 60 + i*2} for i in range(10)],
        "ResNet-18": [{"epoch": i+1, "accuracy": 70 + i*1.5} for i in range(15)],
    }
    
    energy_data = {
        "SimpleCNN": [0.001 for _ in range(10)],
        "ResNet-18": [0.004 for _ in range(15)],
    }
    
    generated = viz.generate_all_visualizations(
        epoch_data=epoch_data,
        energy_data=energy_data
    )
    
    print("\nGenerated files:")
    for name, path in generated.items():
        print(f"  {name}: {path}")
