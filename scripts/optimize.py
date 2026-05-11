"""
Model Optimization Module for Carbon Footprint AI Framework
Implements pruning, quantization, and knowledge distillation
"""
import os
import copy
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    MODELS_DIR, RESULTS_DIR, 
    PRUNING_CONFIG, QUANTIZATION_CONFIG, DISTILLATION_CONFIG,
    get_model_config, get_model_save_path
)
from models import get_model, get_model_info, ResNet18, MobileNetV2
from energy_tracker import EnergyTracker


@dataclass
class OptimizationResult:
    """Results from model optimization"""
    original_model: str
    optimization_type: str
    
    # Size metrics
    original_params: int
    optimized_params: int
    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    
    # Performance metrics
    original_accuracy: float
    optimized_accuracy: float
    accuracy_drop: float
    
    # Energy metrics
    optimization_energy_kwh: float
    optimization_co2_kg: float
    inference_speedup: float
    
    # Additional info
    config: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# PRUNING
# ============================================================================
class ModelPruner:
    """
    Implements various pruning strategies for neural networks.
    Supports L1 unstructured, structured, and global pruning.
    """
    
    def __init__(self, model: nn.Module, device: str = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.original_params = sum(p.numel() for p in model.parameters())
    
    def l1_unstructured_prune(self, sparsity: float = 0.5) -> nn.Module:
        """
        Apply L1 unstructured pruning to all Conv2d and Linear layers.
        
        Args:
            sparsity: Fraction of weights to prune (0.5 = 50% pruned)
        
        Returns:
            Pruned model
        """
        model = copy.deepcopy(self.model)
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
        
        return model
    
    def global_unstructured_prune(self, sparsity: float = 0.5) -> nn.Module:
        """
        Apply global L1 unstructured pruning across all layers.
        Prunes based on global weight magnitudes.
        
        Args:
            sparsity: Fraction of total weights to prune
        
        Returns:
            Pruned model
        """
        model = copy.deepcopy(self.model)
        
        # Collect all prunable parameters
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity
        )
        
        return model
    
    def structured_prune(self, sparsity: float = 0.5, dim: int = 0) -> nn.Module:
        """
        Apply structured pruning (prune entire filters/neurons).
        
        Args:
            sparsity: Fraction of structures to prune
            dim: Dimension to prune along (0 for output channels)
        
        Returns:
            Pruned model
        """
        model = copy.deepcopy(self.model)
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=sparsity, n=1, dim=dim)
            elif isinstance(module, nn.Linear):
                prune.ln_structured(module, name='weight', amount=sparsity, n=1, dim=dim)
        
        return model
    
    def remove_pruning_reparametrization(self, model: nn.Module) -> nn.Module:
        """
        Make pruning permanent by removing the pruning reparametrization.
        This converts pruning masks to actual zero weights.
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass  # No pruning applied to this layer
        return model
    
    def get_sparsity(self, model: nn.Module) -> Dict[str, float]:
        """Calculate actual sparsity of the model"""
        total_zeros = 0
        total_params = 0
        layer_sparsity = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                zeros = (weight == 0).sum().item()
                total = weight.numel()
                
                total_zeros += zeros
                total_params += total
                layer_sparsity[name] = zeros / total if total > 0 else 0
        
        return {
            "global_sparsity": total_zeros / total_params if total_params > 0 else 0,
            "layer_sparsity": layer_sparsity,
            "total_zeros": total_zeros,
            "total_params": total_params
        }


def prune_model(
    model: nn.Module,
    sparsity: float = 0.5,
    method: str = "l1_unstructured",
    device: str = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function to prune a model.
    
    Args:
        model: Model to prune
        sparsity: Target sparsity level
        method: Pruning method ('l1_unstructured', 'global', 'structured')
        device: Device to use
    
    Returns:
        Tuple of (pruned_model, sparsity_info)
    """
    pruner = ModelPruner(model, device)
    
    if method == "l1_unstructured":
        pruned_model = pruner.l1_unstructured_prune(sparsity)
    elif method == "global":
        pruned_model = pruner.global_unstructured_prune(sparsity)
    elif method == "structured":
        pruned_model = pruner.structured_prune(sparsity)
    else:
        raise ValueError(f"Unknown pruning method: {method}")
    
    # Get sparsity info
    sparsity_info = pruner.get_sparsity(pruned_model)
    
    # Make pruning permanent
    pruned_model = pruner.remove_pruning_reparametrization(pruned_model)
    
    return pruned_model, sparsity_info


# ============================================================================
# QUANTIZATION
# ============================================================================
class ModelQuantizer:
    """
    Implements quantization strategies for neural networks.
    Supports dynamic quantization and static quantization preparation.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_size = self._get_model_size(model)
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def dynamic_quantize(self) -> nn.Module:
        """
        Apply dynamic quantization (INT8) to Linear layers.
        This is the simplest form of quantization, requiring no calibration.
        
        Returns:
            Quantized model (CPU only)
        """
        model_cpu = copy.deepcopy(self.model).cpu()
        
        # Dynamic quantization only works on CPU
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear},  # Quantize Linear layers
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def prepare_static_quantization(self, calibration_loader: DataLoader = None) -> nn.Module:
        """
        Prepare model for static quantization.
        Note: Full static quantization requires calibration data.
        
        Returns:
            Prepared model (requires calibration and conversion)
        """
        model_cpu = copy.deepcopy(self.model).cpu()
        model_cpu.eval()
        
        # Set quantization config
        model_cpu.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare for static quantization
        prepared_model = torch.quantization.prepare(model_cpu, inplace=False)
        
        return prepared_model
    
    def get_size_reduction(self, quantized_model: nn.Module) -> Dict[str, float]:
        """Calculate size reduction from quantization"""
        quantized_size = self._get_model_size(quantized_model)
        
        return {
            "original_size_mb": self.original_size,
            "quantized_size_mb": quantized_size,
            "reduction_ratio": 1 - (quantized_size / self.original_size) if self.original_size > 0 else 0,
            "compression_factor": self.original_size / quantized_size if quantized_size > 0 else 1
        }


def quantize_model(
    model: nn.Module,
    method: str = "dynamic"
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Convenience function to quantize a model.
    
    Args:
        model: Model to quantize
        method: Quantization method ('dynamic', 'static_prep')
    
    Returns:
        Tuple of (quantized_model, size_info)
    """
    quantizer = ModelQuantizer(model)
    
    if method == "dynamic":
        quantized_model = quantizer.dynamic_quantize()
    elif method == "static_prep":
        quantized_model = quantizer.prepare_static_quantization()
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    size_info = quantizer.get_size_reduction(quantized_model)
    
    return quantized_model, size_info


# ============================================================================
# KNOWLEDGE DISTILLATION
# ============================================================================
class KnowledgeDistiller:
    """
    Implements knowledge distillation to train a smaller student model
    using a larger teacher model's knowledge.
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = teacher.to(self.device)
        self.student = student.to(self.device)
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss vs hard label loss
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined distillation loss.
        
        Loss = alpha * KL_div(soft_student, soft_teacher) + (1-alpha) * CE(student, labels)
        """
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Distillation loss (KL divergence)
        distill_loss = F.kl_div(
            soft_student,
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard label loss (cross entropy)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
        
        return total_loss
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 15,
        learning_rate: float = 0.001,
        track_energy: bool = True
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Train student model using knowledge distillation.
        
        Returns:
            Tuple of (trained_student, training_info)
        """
        optimizer = torch.optim.Adam(self.student.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Energy tracking
        energy_tracker = None
        if track_energy:
            energy_tracker = EnergyTracker(
                experiment_name="knowledge_distillation",
                track_gpu=True
            )
            energy_tracker.start()
        
        # Training loop
        epoch_info = []
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            # Train
            self.student.train()
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Distillation Epoch {epoch+1}/{epochs}")
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
                
                # Get student predictions
                student_logits = self.student(inputs)
                
                # Compute loss
                loss = self.distillation_loss(student_logits, teacher_logits, labels)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})
            
            scheduler.step()
            
            # Evaluate
            accuracy = self._evaluate(test_loader)
            epoch_info.append({
                'epoch': epoch + 1,
                'loss': running_loss / len(train_loader),
                'accuracy': accuracy
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            
            print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, "
                  f"Acc={accuracy:.2f}%, Best={best_accuracy:.2f}%")
        
        # Stop energy tracking
        energy_metrics = None
        if energy_tracker:
            energy_metrics = energy_tracker.stop()
        
        training_info = {
            'epochs': epochs,
            'temperature': self.temperature,
            'alpha': self.alpha,
            'best_accuracy': best_accuracy,
            'epoch_info': epoch_info,
            'energy_kwh': energy_metrics.energy_kwh if energy_metrics else 0,
            'co2_kg': energy_metrics.co2_kg if energy_metrics else 0
        }
        
        return self.student, training_info
    
    def _evaluate(self, test_loader: DataLoader) -> float:
        """Evaluate student model accuracy"""
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.student(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100. * correct / total


def distill_model(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    temperature: float = 4.0,
    alpha: float = 0.5,
    epochs: int = 15,
    device: str = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function for knowledge distillation.
    
    Args:
        teacher: Teacher model (larger, trained)
        student: Student model (smaller, to be trained)
        train_loader: Training data
        test_loader: Test data
        temperature: Softmax temperature
        alpha: Weight for distillation loss
        epochs: Training epochs
        device: Device to use
    
    Returns:
        Tuple of (trained_student, training_info)
    """
    distiller = KnowledgeDistiller(
        teacher=teacher,
        student=student,
        temperature=temperature,
        alpha=alpha,
        device=device
    )
    
    return distiller.train(train_loader, test_loader, epochs=epochs)


# ============================================================================
# FINE-TUNING PRUNED MODELS
# ============================================================================
def fine_tune_pruned_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 5,
    learning_rate: float = 0.0001,
    device: str = None,
    track_energy: bool = True
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Fine-tune a pruned model to recover accuracy.
    
    Args:
        model: Pruned model to fine-tune
        train_loader: Training data
        test_loader: Test data
        epochs: Fine-tuning epochs
        learning_rate: Learning rate (typically lower than original training)
        device: Device to use
        track_energy: Whether to track energy consumption
    
    Returns:
        Tuple of (fine_tuned_model, training_info)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Energy tracking
    energy_tracker = None
    if track_energy:
        energy_tracker = EnergyTracker(
            experiment_name="pruned_fine_tuning",
            track_gpu=True
        )
        energy_tracker.start()
    
    epoch_info = []
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        epoch_info.append({
            'epoch': epoch + 1,
            'loss': running_loss / len(train_loader),
            'accuracy': accuracy
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
        print(f"Fine-tune Epoch {epoch+1}: Acc={accuracy:.2f}%")
    
    # Stop energy tracking
    energy_metrics = None
    if energy_tracker:
        energy_metrics = energy_tracker.stop()
    
    return model, {
        'epochs': epochs,
        'best_accuracy': best_accuracy,
        'epoch_info': epoch_info,
        'energy_kwh': energy_metrics.energy_kwh if energy_metrics else 0,
        'co2_kg': energy_metrics.co2_kg if energy_metrics else 0
    }


# ============================================================================
# INFERENCE BENCHMARKING
# ============================================================================
def benchmark_inference(
    model: nn.Module,
    input_size: Tuple = (1, 3, 32, 32),
    num_iterations: int = 100,
    device: str = None,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: Model to benchmark
        input_size: Input tensor size
        num_iterations: Number of inference iterations
        device: Device to benchmark on
        warmup: Warmup iterations
    
    Returns:
        Dictionary with timing statistics
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle quantized models (CPU only)
    try:
        model = model.to(device)
    except RuntimeError:
        # Quantized models must stay on CPU
        device = "cpu"
        model = model.to(device)
    
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Synchronize CUDA if needed
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    return {
        "mean_latency_ms": np.mean(times) * 1000,
        "std_latency_ms": np.std(times) * 1000,
        "min_latency_ms": np.min(times) * 1000,
        "max_latency_ms": np.max(times) * 1000,
        "throughput_fps": 1.0 / np.mean(times),
        "device": device
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import numpy as np
    
    print("Carbon Footprint AI Framework - Optimization Module")
    print("=" * 60)
    
    # Test with a simple model
    print("\n1. Testing Pruning...")
    model = ResNet18(num_classes=10)
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {original_params:,}")
    
    # Prune model
    pruned_model, sparsity_info = prune_model(model, sparsity=0.5, method="l1_unstructured")
    print(f"Global sparsity: {sparsity_info['global_sparsity']*100:.1f}%")
    
    print("\n2. Testing Quantization...")
    # Quantize model
    quantized_model, size_info = quantize_model(model, method="dynamic")
    print(f"Original size: {size_info['original_size_mb']:.2f} MB")
    print(f"Quantized size: {size_info['quantized_size_mb']:.2f} MB")
    print(f"Compression: {size_info['compression_factor']:.2f}x")
    
    print("\n3. Testing Inference Benchmark...")
    # Benchmark
    orig_benchmark = benchmark_inference(model, device="cpu", num_iterations=50)
    quant_benchmark = benchmark_inference(quantized_model, device="cpu", num_iterations=50)
    
    print(f"Original latency: {orig_benchmark['mean_latency_ms']:.2f} ms")
    print(f"Quantized latency: {quant_benchmark['mean_latency_ms']:.2f} ms")
    print(f"Speedup: {orig_benchmark['mean_latency_ms']/quant_benchmark['mean_latency_ms']:.2f}x")
    
    print("\n" + "=" * 60)
    print("Optimization module ready!")
