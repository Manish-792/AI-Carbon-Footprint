"""
Experiment Orchestrator for Carbon Footprint AI Framework
Runs all experiments, trains models, applies optimizations, and generates results
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import numpy as np

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MODELS_DIR, RESULTS_DIR, PLOTS_DIR, DATA_DIR,
    CIFAR10_MODELS, TEXT_MODELS, OPTIMIZED_MODELS, ALL_MODELS,
    EXPERIMENT_CONFIG, PRUNING_CONFIG, DISTILLATION_CONFIG,
    get_model_config, get_model_save_path
)
from models import get_model, get_model_info, ResNet18, MobileNetV2
from train import (
    Trainer, train_model, load_model,
    get_cifar10_dataloaders, get_imdb_dataloaders
)
from optimize import (
    prune_model, quantize_model, distill_model,
    fine_tune_pruned_model, benchmark_inference
)
from metrics import MetricsProcessor, ModelMetrics
from energy_tracker import EnergyTracker, track_energy
from visualize import Visualizer


class ExperimentOrchestrator:
    """
    Orchestrates all experiments for the Carbon Footprint AI Framework.
    """
    
    def __init__(
        self,
        region: str = "india",
        device: str = None,
        skip_transformers: bool = False
    ):
        self.region = region
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.skip_transformers = skip_transformers
        
        # Results storage
        self.results: Dict[str, Any] = {}
        self.metrics_processor = MetricsProcessor()
        self.epoch_data: Dict[str, List[Dict]] = {}
        self.energy_data: Dict[str, List[float]] = {}
        self.utilization_data: Dict[str, List[float]] = {}
        
        # Setup
        self._setup_directories()
        self._print_system_info()
    
    def _setup_directories(self):
        """Create necessary directories"""
        for dir_path in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, DATA_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _print_system_info(self):
        """Print system configuration"""
        print("=" * 70)
        print("CARBON FOOTPRINT AI FRAMEWORK - EXPERIMENT ORCHESTRATOR")
        print("=" * 70)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Device: {self.device}")
        print(f"Region: {self.region}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {gpu_mem:.1f} GB")
        
        print(f"PyTorch Version: {torch.__version__}")
        print("=" * 70)
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """
        Run all experiments in sequence.
        """
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print("PHASE 1: TRAINING BASE MODELS")
        print("=" * 70)
        
        # 1. Train CIFAR-10 models
        cifar_models = ["simple_cnn", "resnet18", "mobilenetv2"]
        for model_name in cifar_models:
            self._train_and_record(model_name)
        
        # 2. Train transformer models (optional)
        if not self.skip_transformers:
            print("\n" + "=" * 70)
            print("PHASE 2: TRAINING TRANSFORMER MODELS")
            print("=" * 70)
            
            try:
                for model_name in ["distilbert", "tinybert"]:
                    self._train_and_record(model_name)
            except Exception as e:
                print(f"Skipping transformer models: {e}")
        
        # 3. Apply optimizations
        print("\n" + "=" * 70)
        print("PHASE 3: APPLYING OPTIMIZATIONS")
        print("=" * 70)
        
        self._run_optimization_experiments()
        
        # 4. Generate visualizations
        print("\n" + "=" * 70)
        print("PHASE 4: GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        self._generate_visualizations()
        
        # 5. Save final results
        print("\n" + "=" * 70)
        print("PHASE 5: SAVING RESULTS")
        print("=" * 70)
        
        self._save_final_results()
        
        total_time = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"ALL EXPERIMENTS COMPLETED IN {total_time/60:.1f} MINUTES")
        print(f"{'=' * 70}")
        
        return self.results
    
    def _train_and_record(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Train a model and record results"""
        print(f"\n--- Training {model_name} ---")
        
        try:
            config = get_model_config(model_name)
            if config is None:
                print(f"  Config not found for {model_name}")
                return None
            
            # Get dataloaders
            if config.dataset == "cifar10":
                train_loader, test_loader = get_cifar10_dataloaders(
                    batch_size=config.batch_size
                )
            else:
                train_loader, test_loader, _ = get_imdb_dataloaders(
                    batch_size=config.batch_size
                )
            
            # Create model
            model = get_model(
                config.architecture,
                num_classes=config.num_classes,
                pretrained=config.pretrained
            )
            
            # Train with energy tracking
            trainer = Trainer(
                model=model,
                config=config,
                device=self.device,
                track_energy=True,
                region=self.region
            )
            
            result = trainer.train(train_loader, test_loader, save_best=True)
            
            # Record results
            self.results[model_name] = result.to_dict()
            
            # Store epoch data for visualization
            self.epoch_data[config.name] = [
                {"epoch": i+1, "accuracy": acc, "loss": loss}
                for i, (acc, loss) in enumerate(zip(result.epoch_accuracies, result.epoch_losses))
            ]
            
            if result.epoch_energies:
                self.energy_data[config.name] = result.epoch_energies
            
            # Add to metrics processor
            model_info = get_model_info(model)
            self.metrics_processor.add_model_result(
                model_name=config.name,
                architecture=config.architecture,
                dataset=config.dataset,
                accuracy=result.best_accuracy,
                loss=result.final_loss,
                num_parameters=model_info.num_params,
                model_size_mb=model_info.model_size_mb,
                training_time_seconds=result.training_time_seconds,
                training_energy_kwh=result.total_energy_kwh,
                training_co2_kg=result.total_co2_kg,
                flops=model_info.flops,
                optimization_type="none"
            )
            
            print(f"  Completed: Accuracy={result.best_accuracy:.2f}%, "
                  f"CO2={result.total_co2_kg*1000:.4f}g")
            
            return result.to_dict()
            
        except Exception as e:
            print(f"  Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _run_optimization_experiments(self):
        """Run all optimization experiments"""
        
        # Check if ResNet-18 was trained
        resnet_path = get_model_save_path("ResNet-18")
        if not resnet_path.exists():
            print("ResNet-18 not found, skipping optimization experiments")
            return
        
        # Load base model
        train_loader, test_loader = get_cifar10_dataloaders(batch_size=128)
        
        # 1. Pruning experiments
        print("\n--- Pruning Experiments ---")
        self._run_pruning_experiment(train_loader, test_loader)
        
        # 2. Quantization experiments
        print("\n--- Quantization Experiments ---")
        self._run_quantization_experiment(test_loader)
        
        # 3. Knowledge distillation
        print("\n--- Knowledge Distillation ---")
        self._run_distillation_experiment(train_loader, test_loader)
    
    def _run_pruning_experiment(self, train_loader, test_loader):
        """Run pruning experiment on ResNet-18"""
        try:
            # Load trained ResNet-18
            base_model = load_model("resnet18", device=self.device)
            base_model = base_model.to(self.device)
            
            # Energy tracking for pruning
            with track_energy("resnet18_pruning", region=self.region) as tracker:
                # Prune with 50% sparsity
                pruned_model, sparsity_info = prune_model(
                    base_model, sparsity=0.5, method="l1_unstructured"
                )
                
                # Fine-tune
                pruned_model, ft_info = fine_tune_pruned_model(
                    pruned_model,
                    train_loader,
                    test_loader,
                    epochs=PRUNING_CONFIG.fine_tune_epochs,
                    device=self.device,
                    track_energy=False
                )
            
            # Evaluate
            pruned_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = pruned_model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            accuracy = 100. * correct / total
            
            # Save model
            save_path = get_model_save_path("ResNet-18-Pruned")
            torch.save({
                'model_state_dict': pruned_model.state_dict(),
                'accuracy': accuracy,
                'sparsity': sparsity_info['global_sparsity']
            }, save_path)
            
            # Record metrics
            model_info = get_model_info(pruned_model)
            self.metrics_processor.add_model_result(
                model_name="ResNet-18-Pruned",
                architecture="resnet18_pruned",
                dataset="cifar10",
                accuracy=accuracy,
                loss=0,
                num_parameters=model_info.num_params,
                model_size_mb=model_info.model_size_mb,
                training_time_seconds=tracker.metrics.duration_seconds,
                training_energy_kwh=tracker.metrics.energy_kwh,
                training_co2_kg=tracker.metrics.co2_kg,
                optimization_type="pruned"
            )
            
            self.results["resnet18_pruned"] = {
                "accuracy": accuracy,
                "sparsity": sparsity_info['global_sparsity'],
                "energy_kwh": tracker.metrics.energy_kwh,
                "co2_kg": tracker.metrics.co2_kg
            }
            
            print(f"  Pruned model: Accuracy={accuracy:.2f}%, "
                  f"Sparsity={sparsity_info['global_sparsity']*100:.1f}%")
            
        except Exception as e:
            print(f"  Pruning error: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_quantization_experiment(self, test_loader):
        """Run quantization experiment on ResNet-18"""
        try:
            # Load trained ResNet-18
            base_model = load_model("resnet18", device="cpu")  # Quantization on CPU
            
            with track_energy("resnet18_quantization", region=self.region) as tracker:
                # Quantize
                quantized_model, size_info = quantize_model(base_model, method="dynamic")
            
            # Evaluate (on CPU)
            quantized_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = quantized_model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            accuracy = 100. * correct / total
            
            # Benchmark inference
            orig_benchmark = benchmark_inference(base_model, device="cpu", num_iterations=50)
            quant_benchmark = benchmark_inference(quantized_model, device="cpu", num_iterations=50)
            
            speedup = orig_benchmark['mean_latency_ms'] / quant_benchmark['mean_latency_ms']
            
            # Record metrics
            self.metrics_processor.add_model_result(
                model_name="ResNet-18-Quantized",
                architecture="resnet18_quantized",
                dataset="cifar10",
                accuracy=accuracy,
                loss=0,
                num_parameters=sum(p.numel() for p in base_model.parameters()),
                model_size_mb=size_info['quantized_size_mb'],
                training_time_seconds=tracker.metrics.duration_seconds,
                training_energy_kwh=tracker.metrics.energy_kwh,
                training_co2_kg=tracker.metrics.co2_kg,
                optimization_type="quantized"
            )
            
            self.results["resnet18_quantized"] = {
                "accuracy": accuracy,
                "original_size_mb": size_info['original_size_mb'],
                "quantized_size_mb": size_info['quantized_size_mb'],
                "compression_factor": size_info['compression_factor'],
                "speedup": speedup
            }
            
            print(f"  Quantized model: Accuracy={accuracy:.2f}%, "
                  f"Compression={size_info['compression_factor']:.2f}x, "
                  f"Speedup={speedup:.2f}x")
            
        except Exception as e:
            print(f"  Quantization error: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_distillation_experiment(self, train_loader, test_loader):
        """Run knowledge distillation from ResNet-18 to MobileNetV2"""
        try:
            # Load teacher (ResNet-18)
            teacher = load_model("resnet18", device=self.device)
            teacher = teacher.to(self.device)
            
            # Create student (MobileNetV2)
            student = MobileNetV2(num_classes=10)
            
            with track_energy("knowledge_distillation", region=self.region) as tracker:
                # Distill
                trained_student, distill_info = distill_model(
                    teacher=teacher,
                    student=student,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    temperature=DISTILLATION_CONFIG.temperature,
                    alpha=DISTILLATION_CONFIG.alpha,
                    epochs=DISTILLATION_CONFIG.student_epochs,
                    device=self.device
                )
            
            # Save model
            save_path = get_model_save_path("MobileNetV2-Distilled")
            torch.save({
                'model_state_dict': trained_student.state_dict(),
                'accuracy': distill_info['best_accuracy']
            }, save_path)
            
            # Record metrics
            model_info = get_model_info(trained_student)
            total_energy = tracker.metrics.energy_kwh + distill_info.get('energy_kwh', 0)
            total_co2 = tracker.metrics.co2_kg + distill_info.get('co2_kg', 0)
            
            self.metrics_processor.add_model_result(
                model_name="MobileNetV2-Distilled",
                architecture="mobilenetv2_distilled",
                dataset="cifar10",
                accuracy=distill_info['best_accuracy'],
                loss=0,
                num_parameters=model_info.num_params,
                model_size_mb=model_info.model_size_mb,
                training_time_seconds=tracker.metrics.duration_seconds,
                training_energy_kwh=total_energy,
                training_co2_kg=total_co2,
                optimization_type="distilled"
            )
            
            self.results["mobilenetv2_distilled"] = {
                "accuracy": distill_info['best_accuracy'],
                "energy_kwh": total_energy,
                "co2_kg": total_co2,
                "epochs": distill_info['epochs']
            }
            
            print(f"  Distilled model: Accuracy={distill_info['best_accuracy']:.2f}%")
            
        except Exception as e:
            print(f"  Distillation error: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_visualizations(self):
        """Generate all visualizations"""
        try:
            viz = Visualizer(metrics_processor=self.metrics_processor)
            
            generated = viz.generate_all_visualizations(
                epoch_data=self.epoch_data,
                energy_data=self.energy_data,
                utilization_data=self.utilization_data
            )
            
            self.results["visualizations"] = {
                name: str(path) for name, path in generated.items()
            }
            
            print(f"  Generated {len(generated)} visualizations")
            
        except Exception as e:
            print(f"  Visualization error: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_final_results(self):
        """Save all results to files"""
        # Save main results
        results_path = RESULTS_DIR / "experiment_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"  Saved results to {results_path}")
        
        # Save metrics
        self.metrics_processor.save_all_metrics(RESULTS_DIR)
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print experiment summary"""
        print("\n" + "=" * 70)
        print("EXPERIMENT SUMMARY")
        print("=" * 70)
        
        summary = self.metrics_processor.get_aggregate_summary()
        
        if summary:
            print(f"\nTotal models trained: {summary['num_models']}")
            print(f"\nBest accuracy: {summary['accuracy']['max']:.2f}% "
                  f"({summary['accuracy']['best_model']})")
            print(f"Best Green AI Score: {summary['green_ai_score']['max']:.2f} "
                  f"({summary['green_ai_score']['best_model']})")
            print(f"Total energy consumed: {summary['energy']['total_kwh']*1000:.2f} Wh")
            print(f"Total CO2 emitted: {summary['co2_emissions']['total_kg']*1000:.2f} g")
            print(f"\nPareto-efficient models: {', '.join(summary['pareto_efficient'])}")
        
        print("\n" + "=" * 70)
        
        # Print comparison table
        print("\nMODEL COMPARISON TABLE:")
        print("-" * 70)
        df = self.metrics_processor.get_comparison_table()
        if not df.empty:
            print(df.to_string(index=False))


def run_quick_experiment(region: str = "india", device: str = None):
    """
    Run a quick experiment with just SimpleCNN for testing.
    """
    print("Running quick experiment (SimpleCNN only)...")
    
    orchestrator = ExperimentOrchestrator(
        region=region,
        device=device,
        skip_transformers=True
    )
    
    # Train only SimpleCNN
    orchestrator._train_and_record("simple_cnn")
    orchestrator._generate_visualizations()
    orchestrator._save_final_results()
    
    return orchestrator.results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Carbon Footprint AI Framework - Experiment Orchestrator"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick experiment with SimpleCNN only"
    )
    parser.add_argument(
        "--region", type=str, default="india",
        choices=["india", "us", "eu", "china", "global_avg"],
        help="Region for carbon intensity calculation"
    )
    parser.add_argument(
        "--skip-transformers", action="store_true",
        help="Skip transformer model training"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_experiment(region=args.region, device=args.device)
    else:
        orchestrator = ExperimentOrchestrator(
            region=args.region,
            device=args.device,
            skip_transformers=args.skip_transformers
        )
        orchestrator.run_all_experiments()


if __name__ == "__main__":
    main()
