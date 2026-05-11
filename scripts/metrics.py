"""
Metrics Module for Carbon Footprint AI Framework
Implements Green AI Score and other evaluation metrics
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime

import numpy as np
import pandas as pd

from config import CARBON_INTENSITY, RESULTS_DIR, METRICS_FILE


@dataclass
class ModelMetrics:
    """Complete metrics for a single model"""
    model_name: str
    architecture: str
    dataset: str
    
    # Performance
    accuracy: float
    loss: float
    
    # Model characteristics
    num_parameters: int
    model_size_mb: float
    flops: int = 0
    
    # Training costs
    training_time_seconds: float = 0.0
    training_energy_kwh: float = 0.0
    training_co2_kg: float = 0.0
    
    # Inference costs (per batch)
    inference_latency_ms: float = 0.0
    inference_energy_wh: float = 0.0
    
    # Computed metrics
    green_ai_score: float = 0.0
    carbon_efficiency_index: float = 0.0
    accuracy_per_parameter: float = 0.0
    accuracy_per_flop: float = 0.0
    
    # Regional emissions
    regional_co2: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    optimization_type: str = "none"  # none, pruned, quantized, distilled
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# CORE METRICS
# ============================================================================
def calculate_green_ai_score(
    accuracy: float,
    co2_kg: float,
    epsilon: float = 1e-6
) -> float:
    """
    Calculate the Green AI Score.
    
    Green AI Score = Accuracy / CO2 Emission
    
    Higher is better - models that achieve high accuracy with low emissions score well.
    
    Args:
        accuracy: Model accuracy (0-100)
        co2_kg: CO2 emissions in kg
        epsilon: Small value to prevent division by zero
    
    Returns:
        Green AI Score
    """
    return accuracy / (co2_kg + epsilon)


def calculate_carbon_efficiency_index(
    accuracy: float,
    co2_kg: float,
    training_time_seconds: float,
    epsilon: float = 1e-6
) -> float:
    """
    Calculate the Carbon Efficiency Index.
    
    CEI = (Accuracy × 100) / (CO2 × Training_Time)
    
    This metric penalizes both high emissions and long training times.
    
    Args:
        accuracy: Model accuracy (0-100)
        co2_kg: CO2 emissions in kg
        training_time_seconds: Training time in seconds
        epsilon: Small value to prevent division by zero
    
    Returns:
        Carbon Efficiency Index
    """
    return (accuracy * 100) / ((co2_kg + epsilon) * (training_time_seconds / 3600 + epsilon))


def calculate_accuracy_per_parameter(
    accuracy: float,
    num_parameters: int
) -> float:
    """
    Calculate accuracy gained per million parameters.
    
    Args:
        accuracy: Model accuracy (0-100)
        num_parameters: Number of model parameters
    
    Returns:
        Accuracy per million parameters
    """
    return accuracy / (num_parameters / 1e6) if num_parameters > 0 else 0


def calculate_accuracy_per_flop(
    accuracy: float,
    flops: int
) -> float:
    """
    Calculate accuracy per gigaFLOP.
    
    Args:
        accuracy: Model accuracy (0-100)
        flops: Floating point operations
    
    Returns:
        Accuracy per GFLOP
    """
    return accuracy / (flops / 1e9) if flops > 0 else 0


def calculate_energy_efficiency(
    accuracy: float,
    energy_kwh: float
) -> float:
    """
    Calculate energy efficiency (accuracy per Wh).
    
    Args:
        accuracy: Model accuracy (0-100)
        energy_kwh: Energy consumption in kWh
    
    Returns:
        Accuracy per Wh
    """
    energy_wh = energy_kwh * 1000
    return accuracy / energy_wh if energy_wh > 0 else 0


def calculate_regional_emissions(
    energy_kwh: float
) -> Dict[str, float]:
    """
    Calculate CO2 emissions for different regions.
    
    Args:
        energy_kwh: Energy consumption in kWh
    
    Returns:
        Dictionary of region -> CO2 emissions (kg)
    """
    return {
        region: energy_kwh * intensity
        for region, intensity in CARBON_INTENSITY.items()
    }


def calculate_emission_savings(
    baseline_co2: float,
    optimized_co2: float
) -> Dict[str, float]:
    """
    Calculate emission savings from optimization.
    
    Args:
        baseline_co2: Baseline model CO2 emissions
        optimized_co2: Optimized model CO2 emissions
    
    Returns:
        Dictionary with savings metrics
    """
    absolute_savings = baseline_co2 - optimized_co2
    relative_savings = (absolute_savings / baseline_co2 * 100) if baseline_co2 > 0 else 0
    
    return {
        "absolute_savings_kg": absolute_savings,
        "absolute_savings_g": absolute_savings * 1000,
        "relative_savings_percent": relative_savings,
        "reduction_factor": baseline_co2 / optimized_co2 if optimized_co2 > 0 else float('inf')
    }


# ============================================================================
# COMPARATIVE METRICS
# ============================================================================
def calculate_pareto_efficiency(
    models: List[ModelMetrics]
) -> List[str]:
    """
    Find Pareto-efficient models (optimal accuracy-emission trade-off).
    
    A model is Pareto-efficient if no other model has both higher accuracy
    and lower emissions.
    
    Args:
        models: List of ModelMetrics
    
    Returns:
        List of model names that are Pareto-efficient
    """
    pareto_models = []
    
    for model in models:
        is_pareto = True
        for other in models:
            if other.model_name != model.model_name:
                # Check if other dominates model
                if other.accuracy >= model.accuracy and other.training_co2_kg <= model.training_co2_kg:
                    if other.accuracy > model.accuracy or other.training_co2_kg < model.training_co2_kg:
                        is_pareto = False
                        break
        
        if is_pareto:
            pareto_models.append(model.model_name)
    
    return pareto_models


def rank_models_by_metric(
    models: List[ModelMetrics],
    metric: str = "green_ai_score",
    ascending: bool = False
) -> List[Dict[str, Any]]:
    """
    Rank models by a specific metric.
    
    Args:
        models: List of ModelMetrics
        metric: Metric to rank by
        ascending: Sort in ascending order (False = highest first)
    
    Returns:
        List of dictionaries with rank and model info
    """
    # Extract metric values
    model_values = []
    for m in models:
        value = getattr(m, metric, 0)
        model_values.append({
            "model_name": m.model_name,
            "value": value,
            "accuracy": m.accuracy,
            "co2_kg": m.training_co2_kg
        })
    
    # Sort
    model_values.sort(key=lambda x: x["value"], reverse=not ascending)
    
    # Add ranks
    for i, mv in enumerate(model_values):
        mv["rank"] = i + 1
    
    return model_values


def calculate_efficiency_frontier(
    models: List[ModelMetrics],
    num_points: int = 20
) -> List[Dict[str, float]]:
    """
    Calculate the efficiency frontier curve.
    
    Args:
        models: List of ModelMetrics
        num_points: Number of points on the frontier
    
    Returns:
        List of (accuracy, co2) points on the frontier
    """
    if not models:
        return []
    
    # Get all (co2, accuracy) pairs
    points = [(m.training_co2_kg, m.accuracy) for m in models]
    
    # Sort by CO2
    points.sort(key=lambda x: x[0])
    
    # Build frontier (monotonically increasing accuracy)
    frontier = []
    max_acc = 0
    
    for co2, acc in points:
        if acc > max_acc:
            frontier.append({"co2_kg": co2, "accuracy": acc})
            max_acc = acc
    
    return frontier


# ============================================================================
# AGGREGATE METRICS
# ============================================================================
def calculate_aggregate_metrics(
    models: List[ModelMetrics]
) -> Dict[str, Any]:
    """
    Calculate aggregate metrics across all models.
    
    Args:
        models: List of ModelMetrics
    
    Returns:
        Dictionary of aggregate metrics
    """
    if not models:
        return {}
    
    accuracies = [m.accuracy for m in models]
    co2_values = [m.training_co2_kg for m in models]
    green_scores = [m.green_ai_score for m in models]
    energies = [m.training_energy_kwh for m in models]
    params = [m.num_parameters for m in models]
    
    return {
        "num_models": len(models),
        "accuracy": {
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
            "min": np.min(accuracies),
            "max": np.max(accuracies),
            "best_model": models[np.argmax(accuracies)].model_name
        },
        "co2_emissions": {
            "total_kg": np.sum(co2_values),
            "mean_kg": np.mean(co2_values),
            "min_kg": np.min(co2_values),
            "max_kg": np.max(co2_values),
            "cleanest_model": models[np.argmin(co2_values)].model_name
        },
        "energy": {
            "total_kwh": np.sum(energies),
            "mean_kwh": np.mean(energies),
            "min_kwh": np.min(energies),
            "max_kwh": np.max(energies)
        },
        "green_ai_score": {
            "mean": np.mean(green_scores),
            "max": np.max(green_scores),
            "best_model": models[np.argmax(green_scores)].model_name
        },
        "parameters": {
            "smallest": min(params),
            "largest": max(params),
            "total": sum(params)
        },
        "pareto_efficient": calculate_pareto_efficiency(models)
    }


# ============================================================================
# METRICS PROCESSOR
# ============================================================================
class MetricsProcessor:
    """
    Process and analyze metrics from experiment results.
    """
    
    def __init__(self):
        self.models: List[ModelMetrics] = []
        self.comparison_tables: Dict[str, pd.DataFrame] = {}
    
    def add_model_result(
        self,
        model_name: str,
        architecture: str,
        dataset: str,
        accuracy: float,
        loss: float,
        num_parameters: int,
        model_size_mb: float,
        training_time_seconds: float,
        training_energy_kwh: float,
        training_co2_kg: float,
        flops: int = 0,
        inference_latency_ms: float = 0,
        optimization_type: str = "none"
    ) -> ModelMetrics:
        """
        Add a model's results and calculate all metrics.
        """
        # Calculate derived metrics
        green_ai_score = calculate_green_ai_score(accuracy, training_co2_kg)
        carbon_efficiency_index = calculate_carbon_efficiency_index(
            accuracy, training_co2_kg, training_time_seconds
        )
        accuracy_per_param = calculate_accuracy_per_parameter(accuracy, num_parameters)
        accuracy_per_flop = calculate_accuracy_per_flop(accuracy, flops)
        regional_co2 = calculate_regional_emissions(training_energy_kwh)
        
        metrics = ModelMetrics(
            model_name=model_name,
            architecture=architecture,
            dataset=dataset,
            accuracy=accuracy,
            loss=loss,
            num_parameters=num_parameters,
            model_size_mb=model_size_mb,
            flops=flops,
            training_time_seconds=training_time_seconds,
            training_energy_kwh=training_energy_kwh,
            training_co2_kg=training_co2_kg,
            inference_latency_ms=inference_latency_ms,
            green_ai_score=green_ai_score,
            carbon_efficiency_index=carbon_efficiency_index,
            accuracy_per_parameter=accuracy_per_param,
            accuracy_per_flop=accuracy_per_flop,
            regional_co2=regional_co2,
            optimization_type=optimization_type,
            timestamp=datetime.now().isoformat()
        )
        
        self.models.append(metrics)
        return metrics
    
    def load_from_results(self, results_dir: Path = RESULTS_DIR):
        """Load metrics from saved result files"""
        for result_file in results_dir.glob("*_result.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)
                
                self.add_model_result(
                    model_name=data.get("model_name", result_file.stem),
                    architecture=data.get("architecture", "unknown"),
                    dataset=data.get("dataset", "unknown"),
                    accuracy=data.get("final_accuracy", data.get("best_accuracy", 0)),
                    loss=data.get("final_loss", 0),
                    num_parameters=data.get("num_parameters", 0),
                    model_size_mb=data.get("model_size_mb", 0),
                    training_time_seconds=data.get("training_time_seconds", 0),
                    training_energy_kwh=data.get("total_energy_kwh", 0),
                    training_co2_kg=data.get("total_co2_kg", 0),
                    optimization_type=data.get("optimization_type", "none")
                )
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Generate a comparison table of all models.
        """
        data = []
        for m in self.models:
            data.append({
                "Model": m.model_name,
                "Architecture": m.architecture,
                "Dataset": m.dataset,
                "Accuracy (%)": round(m.accuracy, 2),
                "Parameters (M)": round(m.num_parameters / 1e6, 2),
                "Size (MB)": round(m.model_size_mb, 2),
                "Training Time (min)": round(m.training_time_seconds / 60, 2),
                "Energy (Wh)": round(m.training_energy_kwh * 1000, 4),
                "CO2 (g)": round(m.training_co2_kg * 1000, 4),
                "Green AI Score": round(m.green_ai_score, 2),
                "CEI": round(m.carbon_efficiency_index, 2),
                "Optimization": m.optimization_type
            })
        
        df = pd.DataFrame(data)
        self.comparison_tables["main"] = df
        return df
    
    def get_regional_comparison(self) -> pd.DataFrame:
        """
        Generate regional CO2 emissions comparison.
        """
        data = []
        for m in self.models:
            row = {"Model": m.model_name}
            for region, co2 in m.regional_co2.items():
                row[f"{region.title()} (g CO2)"] = round(co2 * 1000, 4)
            data.append(row)
        
        df = pd.DataFrame(data)
        self.comparison_tables["regional"] = df
        return df
    
    def get_optimization_comparison(self) -> pd.DataFrame:
        """
        Compare baseline vs optimized models.
        """
        baseline = [m for m in self.models if m.optimization_type == "none"]
        optimized = [m for m in self.models if m.optimization_type != "none"]
        
        data = []
        for opt_model in optimized:
            # Find corresponding baseline
            base_model = None
            for b in baseline:
                if b.architecture.split("_")[0] in opt_model.architecture:
                    base_model = b
                    break
            
            if base_model:
                savings = calculate_emission_savings(
                    base_model.training_co2_kg,
                    opt_model.training_co2_kg
                )
                
                data.append({
                    "Optimized Model": opt_model.model_name,
                    "Baseline Model": base_model.model_name,
                    "Optimization": opt_model.optimization_type,
                    "Baseline Acc (%)": round(base_model.accuracy, 2),
                    "Optimized Acc (%)": round(opt_model.accuracy, 2),
                    "Acc Change (%)": round(opt_model.accuracy - base_model.accuracy, 2),
                    "Baseline CO2 (g)": round(base_model.training_co2_kg * 1000, 4),
                    "Optimized CO2 (g)": round(opt_model.training_co2_kg * 1000, 4),
                    "CO2 Savings (%)": round(savings["relative_savings_percent"], 2)
                })
        
        df = pd.DataFrame(data)
        self.comparison_tables["optimization"] = df
        return df
    
    def get_leaderboard(self, metric: str = "green_ai_score") -> pd.DataFrame:
        """
        Generate leaderboard ranked by specified metric.
        """
        rankings = rank_models_by_metric(self.models, metric)
        df = pd.DataFrame(rankings)
        df.columns = ["Model", metric.replace("_", " ").title(), "Accuracy (%)", "CO2 (kg)", "Rank"]
        return df
    
    def get_aggregate_summary(self) -> Dict[str, Any]:
        """Get aggregate metrics summary"""
        return calculate_aggregate_metrics(self.models)
    
    def save_all_metrics(self, output_dir: Path = RESULTS_DIR):
        """Save all metrics to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual model metrics
        all_metrics = [m.to_dict() for m in self.models]
        with open(output_dir / "all_model_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        # Save aggregate summary
        summary = self.get_aggregate_summary()
        with open(output_dir / "aggregate_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save comparison tables as CSV
        self.get_comparison_table().to_csv(output_dir / "comparison_table.csv", index=False)
        self.get_regional_comparison().to_csv(output_dir / "regional_comparison.csv", index=False)
        
        if any(m.optimization_type != "none" for m in self.models):
            self.get_optimization_comparison().to_csv(output_dir / "optimization_comparison.csv", index=False)
        
        print(f"Metrics saved to {output_dir}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def format_metrics_summary(metrics: ModelMetrics) -> str:
    """Format metrics as a readable summary string"""
    return f"""
================================================================================
Model: {metrics.model_name}
================================================================================
Performance:
  - Accuracy: {metrics.accuracy:.2f}%
  - Loss: {metrics.loss:.4f}

Model Characteristics:
  - Parameters: {metrics.num_parameters:,} ({metrics.num_parameters/1e6:.2f}M)
  - Size: {metrics.model_size_mb:.2f} MB

Training Costs:
  - Time: {metrics.training_time_seconds:.1f}s ({metrics.training_time_seconds/60:.2f} min)
  - Energy: {metrics.training_energy_kwh*1000:.4f} Wh
  - CO2: {metrics.training_co2_kg*1000:.4f} g

Efficiency Metrics:
  - Green AI Score: {metrics.green_ai_score:.2f}
  - Carbon Efficiency Index: {metrics.carbon_efficiency_index:.2f}
  - Accuracy per Million Params: {metrics.accuracy_per_parameter:.2f}

Regional Emissions (g CO2):
  - India: {metrics.regional_co2.get('india', 0)*1000:.4f}
  - US: {metrics.regional_co2.get('us', 0)*1000:.4f}
  - EU: {metrics.regional_co2.get('eu', 0)*1000:.4f}
================================================================================
"""


def compare_two_models(model_a: ModelMetrics, model_b: ModelMetrics) -> Dict[str, Any]:
    """Compare two models and return differences"""
    return {
        "accuracy_diff": model_a.accuracy - model_b.accuracy,
        "co2_diff": model_a.training_co2_kg - model_b.training_co2_kg,
        "energy_diff": model_a.training_energy_kwh - model_b.training_energy_kwh,
        "time_diff": model_a.training_time_seconds - model_b.training_time_seconds,
        "green_score_diff": model_a.green_ai_score - model_b.green_ai_score,
        "param_diff": model_a.num_parameters - model_b.num_parameters,
        "more_accurate": model_a.model_name if model_a.accuracy > model_b.accuracy else model_b.model_name,
        "more_efficient": model_a.model_name if model_a.green_ai_score > model_b.green_ai_score else model_b.model_name,
        "greener": model_a.model_name if model_a.training_co2_kg < model_b.training_co2_kg else model_b.model_name
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("Carbon Footprint AI Framework - Metrics Module")
    print("=" * 60)
    
    # Create sample metrics for testing
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
    
    # Display comparison table
    print("\nModel Comparison Table:")
    print(processor.get_comparison_table().to_string(index=False))
    
    # Display leaderboard
    print("\n\nGreen AI Score Leaderboard:")
    print(processor.get_leaderboard().to_string(index=False))
    
    # Display aggregate summary
    print("\n\nAggregate Summary:")
    summary = processor.get_aggregate_summary()
    print(f"Total models: {summary['num_models']}")
    print(f"Best accuracy: {summary['accuracy']['max']:.2f}% ({summary['accuracy']['best_model']})")
    print(f"Best Green AI Score: {summary['green_ai_score']['max']:.2f} ({summary['green_ai_score']['best_model']})")
    print(f"Pareto efficient models: {summary['pareto_efficient']}")
