"""
Global Configuration for Carbon Footprint AI Framework
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = BASE_DIR / "scripts"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
FIGURES_DIR = PLOTS_DIR / "figures"
TABLES_DIR = PLOTS_DIR / "tables"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, FIGURES_DIR, TABLES_DIR, DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CARBON INTENSITY VALUES (kgCO2/kWh)
# ============================================================================
CARBON_INTENSITY = {
    "india": 0.82,      # Coal-heavy grid
    "us": 0.42,         # Mixed energy sources
    "eu": 0.30,         # Cleaner grid (more renewables)
    "china": 0.58,      # Coal-heavy but improving
    "global_avg": 0.47, # Global average
}

DEFAULT_REGION = "india"  # Default for experiments

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
@dataclass
class ModelConfig:
    """Configuration for a single model"""
    name: str
    architecture: str
    dataset: str
    num_classes: int
    input_size: tuple
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    pretrained: bool = False
    description: str = ""

# Image Classification Models (CIFAR-10)
CIFAR10_MODELS = {
    "simple_cnn": ModelConfig(
        name="SimpleCNN",
        architecture="simple_cnn",
        dataset="cifar10",
        num_classes=10,
        input_size=(3, 32, 32),
        batch_size=128,
        epochs=15,
        learning_rate=0.001,
        description="Baseline lightweight CNN"
    ),
    "resnet18": ModelConfig(
        name="ResNet-18",
        architecture="resnet18",
        dataset="cifar10",
        num_classes=10,
        input_size=(3, 32, 32),
        batch_size=128,
        epochs=20,
        learning_rate=0.001,
        pretrained=False,
        description="Standard ResNet-18 architecture"
    ),
    "mobilenetv2": ModelConfig(
        name="MobileNetV2",
        architecture="mobilenetv2",
        dataset="cifar10",
        num_classes=10,
        input_size=(3, 32, 32),
        batch_size=128,
        epochs=20,
        learning_rate=0.001,
        pretrained=False,
        description="Efficient mobile architecture"
    ),
}

# Text Classification Models (IMDB)
TEXT_MODELS = {
    "distilbert": ModelConfig(
        name="DistilBERT",
        architecture="distilbert",
        dataset="imdb",
        num_classes=2,
        input_size=(512,),  # max sequence length
        batch_size=16,
        epochs=3,
        learning_rate=2e-5,
        pretrained=True,
        description="Distilled BERT for text classification"
    ),
    "tinybert": ModelConfig(
        name="TinyBERT",
        architecture="tinybert",
        dataset="imdb",
        num_classes=2,
        input_size=(512,),
        batch_size=16,
        epochs=3,
        learning_rate=2e-5,
        pretrained=True,
        description="Smaller efficient BERT variant"
    ),
}

# Optimized Model Variants
OPTIMIZED_MODELS = {
    "resnet18_pruned": ModelConfig(
        name="ResNet-18-Pruned",
        architecture="resnet18_pruned",
        dataset="cifar10",
        num_classes=10,
        input_size=(3, 32, 32),
        batch_size=128,
        epochs=10,
        learning_rate=0.0001,
        description="Pruned ResNet-18 (50% sparsity)"
    ),
    "resnet18_quantized": ModelConfig(
        name="ResNet-18-Quantized",
        architecture="resnet18_quantized",
        dataset="cifar10",
        num_classes=10,
        input_size=(3, 32, 32),
        batch_size=128,
        epochs=0,  # No retraining needed
        description="INT8 quantized ResNet-18"
    ),
    "mobilenetv2_distilled": ModelConfig(
        name="MobileNetV2-Distilled",
        architecture="mobilenetv2_distilled",
        dataset="cifar10",
        num_classes=10,
        input_size=(3, 32, 32),
        batch_size=128,
        epochs=15,
        learning_rate=0.001,
        description="Knowledge distilled from ResNet-18"
    ),
}

# All models combined
ALL_MODELS = {**CIFAR10_MODELS, **TEXT_MODELS, **OPTIMIZED_MODELS}

# ============================================================================
# OPTIMIZATION CONFIGURATIONS
# ============================================================================
@dataclass
class PruningConfig:
    """Configuration for model pruning"""
    sparsity_levels: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    pruning_method: str = "l1_unstructured"
    fine_tune_epochs: int = 5

@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    quantization_type: str = "dynamic"  # dynamic, static, qat
    dtype: str = "qint8"

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation"""
    temperature: float = 4.0
    alpha: float = 0.5  # Weight for distillation loss
    student_epochs: int = 15

PRUNING_CONFIG = PruningConfig()
QUANTIZATION_CONFIG = QuantizationConfig()
DISTILLATION_CONFIG = DistillationConfig()

# ============================================================================
# EXPERIMENT SETTINGS
# ============================================================================
@dataclass
class ExperimentConfig:
    """Configuration for running experiments"""
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = False
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5
    track_per_epoch: bool = True
    track_per_batch: bool = False
    gpu_monitoring_interval: float = 0.5  # seconds

EXPERIMENT_CONFIG = ExperimentConfig()

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
}

COLOR_PALETTE = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "success": "#F18F01",
    "warning": "#C73E1D",
    "info": "#3B1F2B",
    "models": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#5C946E", "#7B2D26", "#F4D35E"],
    "regions": {"india": "#FF6B35", "us": "#004E89", "eu": "#2A9D8F", "china": "#E63946", "global_avg": "#6C757D"},
}

# ============================================================================
# RESULTS FILE PATHS
# ============================================================================
RESULTS_FILE = RESULTS_DIR / "experiment_results.json"
ENERGY_LOG_FILE = RESULTS_DIR / "energy_logs.json"
METRICS_FILE = RESULTS_DIR / "metrics_summary.json"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model"""
    return ALL_MODELS.get(model_name.lower().replace("-", "_").replace(" ", "_"))

def get_carbon_intensity(region: str = DEFAULT_REGION) -> float:
    """Get carbon intensity for a region"""
    return CARBON_INTENSITY.get(region.lower(), CARBON_INTENSITY["global_avg"])

def get_model_save_path(model_name: str, suffix: str = "") -> Path:
    """Get path for saving a model"""
    filename = f"{model_name.lower().replace(' ', '_').replace('-', '_')}{suffix}.pth"
    return MODELS_DIR / filename

def get_results_path(experiment_name: str) -> Path:
    """Get path for saving experiment results"""
    return RESULTS_DIR / f"{experiment_name}_results.json"


if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("Carbon Footprint AI Framework - Configuration")
    print("=" * 60)
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Plots Directory: {PLOTS_DIR}")
    print(f"\nCarbon Intensity Values (kgCO2/kWh):")
    for region, intensity in CARBON_INTENSITY.items():
        print(f"  {region}: {intensity}")
    print(f"\nRegistered Models: {len(ALL_MODELS)}")
    for name, config in ALL_MODELS.items():
        print(f"  - {config.name}: {config.description}")
