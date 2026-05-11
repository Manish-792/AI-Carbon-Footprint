"""
Training Pipeline for Carbon Footprint AI Framework
Supports both image (CIFAR-10) and text (IMDB) classification with energy tracking
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from tqdm import tqdm

# Try to import datasets for image data
try:
    import torchvision
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# Try to import HuggingFace datasets for text data
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

from config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, EXPERIMENT_CONFIG,
    ModelConfig, get_model_config, get_model_save_path
)
from models import get_model, get_model_info, count_parameters
from energy_tracker import EnergyTracker, track_energy


@dataclass
class TrainingResult:
    """Container for training results"""
    model_name: str
    architecture: str
    dataset: str
    
    # Performance metrics
    final_accuracy: float
    best_accuracy: float
    final_loss: float
    
    # Energy metrics
    total_energy_kwh: float
    total_co2_kg: float
    training_time_seconds: float
    
    # Model info
    num_parameters: int
    model_size_mb: float
    
    # Per-epoch tracking
    epoch_accuracies: list
    epoch_losses: list
    epoch_energies: list
    epoch_co2: list
    
    # Metadata
    epochs_trained: int
    batch_size: int
    learning_rate: float
    device: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# DATA LOADING
# ============================================================================
def get_cifar10_dataloaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: Path = DATA_DIR
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 train and test dataloaders with standard augmentations.
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision required for CIFAR-10")
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=EXPERIMENT_CONFIG.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=EXPERIMENT_CONFIG.pin_memory
    )
    
    return train_loader, test_loader


class IMDBDataset(Dataset):
    """Custom dataset wrapper for IMDB sentiment classification"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def get_imdb_dataloaders(
    batch_size: int = 16,
    max_length: int = 256,
    num_samples: int = 5000,  # Use subset for faster experiments
    tokenizer_name: str = "distilbert-base-uncased"
) -> Tuple[DataLoader, DataLoader, Any]:
    """
    Get IMDB train and test dataloaders for text classification.
    Uses a subset of data for reasonable experiment times.
    """
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets and transformers required for IMDB")
    
    # Load dataset
    dataset = load_dataset("imdb")
    
    # Use subset for faster experiments
    train_data = dataset["train"].shuffle(seed=42).select(range(min(num_samples, len(dataset["train"]))))
    test_data = dataset["test"].shuffle(seed=42).select(range(min(num_samples // 5, len(dataset["test"]))))
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    train_encodings = tokenize_function(train_data)
    test_encodings = tokenize_function(test_data)
    
    # Create datasets
    train_dataset = IMDBDataset(
        {"input_ids": train_encodings["input_ids"], 
         "attention_mask": train_encodings["attention_mask"]},
        train_data["label"]
    )
    test_dataset = IMDBDataset(
        {"input_ids": test_encodings["input_ids"],
         "attention_mask": test_encodings["attention_mask"]},
        test_data["label"]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, tokenizer


def get_dataloaders(
    dataset_name: str,
    batch_size: int = 128,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Get dataloaders for a specified dataset"""
    dataset_name = dataset_name.lower()
    
    if dataset_name == "cifar10":
        return get_cifar10_dataloaders(batch_size=batch_size, **kwargs)
    elif dataset_name == "imdb":
        train_loader, test_loader, _ = get_imdb_dataloaders(batch_size=batch_size, **kwargs)
        return train_loader, test_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
class Trainer:
    """
    Main trainer class with energy tracking support.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        device: str = None,
        track_energy: bool = True,
        region: str = "india"
    ):
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.track_energy = track_energy
        self.region = region
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # Tracking
        self.energy_tracker = None
        self.epoch_metrics = []
        self.best_accuracy = 0.0
    
    def _get_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config"""
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            return optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
    
    def _get_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.scheduler.lower() == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        elif self.config.scheduler.lower() == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.1)
        return None
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            else:
                # Text classification batch
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        return {
            'loss': running_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    targets = batch['labels'].to(self.device)
                    outputs = self.model(input_ids, attention_mask)
                
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return {
            'loss': running_loss / len(test_loader),
            'accuracy': 100. * correct / total
        }
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        save_best: bool = True
    ) -> TrainingResult:
        """
        Full training loop with energy tracking.
        """
        print(f"\n{'='*60}")
        print(f"Training {self.config.name}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch Size: {self.config.batch_size}")
        print(f"Learning Rate: {self.config.learning_rate}")
        
        # Get model info
        model_info = get_model_info(self.model)
        print(f"Parameters: {model_info.num_params:,}")
        print(f"Model Size: {model_info.model_size_mb:.2f} MB")
        print(f"{'='*60}\n")
        
        # Initialize energy tracker
        if self.track_energy:
            self.energy_tracker = EnergyTracker(
                experiment_name=self.config.name.lower().replace(" ", "_"),
                region=self.region,
                track_gpu=True,
                save_to_file=True
            )
            self.energy_tracker.start()
        
        # Training metrics
        epoch_accuracies = []
        epoch_losses = []
        epoch_energies = []
        epoch_co2 = []
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config.epochs):
                # Start epoch tracking
                if self.energy_tracker:
                    self.energy_tracker.start_epoch(epoch)
                
                # Train
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Evaluate
                test_metrics = self.evaluate(test_loader)
                
                # End epoch tracking
                if self.energy_tracker:
                    epoch_energy_info = self.energy_tracker.end_epoch()
                    epoch_energies.append(epoch_energy_info.get('energy_kwh', 0))
                    epoch_co2.append(epoch_energy_info.get('co2_kg', 0))
                
                # Record metrics
                epoch_accuracies.append(test_metrics['accuracy'])
                epoch_losses.append(test_metrics['loss'])
                
                # Update best accuracy
                if test_metrics['accuracy'] > self.best_accuracy:
                    self.best_accuracy = test_metrics['accuracy']
                    if save_best:
                        self._save_checkpoint(epoch, test_metrics['accuracy'])
                
                # Update scheduler
                if self.scheduler:
                    self.scheduler.step()
                
                # Print epoch summary
                print(f"Epoch {epoch+1}: Train Loss={train_metrics['loss']:.4f}, "
                      f"Test Acc={test_metrics['accuracy']:.2f}%, "
                      f"Best={self.best_accuracy:.2f}%")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted!")
        
        # Stop energy tracking
        energy_metrics = None
        if self.energy_tracker:
            energy_metrics = self.energy_tracker.stop()
            print(self.energy_tracker.get_summary())
        
        training_time = time.time() - start_time
        
        # Create result
        result = TrainingResult(
            model_name=self.config.name,
            architecture=self.config.architecture,
            dataset=self.config.dataset,
            final_accuracy=epoch_accuracies[-1] if epoch_accuracies else 0,
            best_accuracy=self.best_accuracy,
            final_loss=epoch_losses[-1] if epoch_losses else 0,
            total_energy_kwh=energy_metrics.energy_kwh if energy_metrics else 0,
            total_co2_kg=energy_metrics.co2_kg if energy_metrics else 0,
            training_time_seconds=training_time,
            num_parameters=model_info.num_params,
            model_size_mb=model_info.model_size_mb,
            epoch_accuracies=epoch_accuracies,
            epoch_losses=epoch_losses,
            epoch_energies=epoch_energies,
            epoch_co2=epoch_co2,
            epochs_trained=len(epoch_accuracies),
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            device=self.device,
            timestamp=datetime.now().isoformat()
        )
        
        # Save result
        result_path = RESULTS_DIR / f"{self.config.name.lower().replace(' ', '_')}_result.json"
        result.save(result_path)
        
        return result
    
    def _save_checkpoint(self, epoch: int, accuracy: float):
        """Save model checkpoint"""
        save_path = get_model_save_path(self.config.name)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else vars(self.config)
        }
        torch.save(checkpoint, save_path)
        print(f"  Saved checkpoint: {save_path}")


def train_model(
    model_name: str,
    track_energy: bool = True,
    region: str = "india",
    device: str = None
) -> TrainingResult:
    """
    Train a model by name with energy tracking.
    
    Args:
        model_name: Name of the model to train (from config)
        track_energy: Whether to track energy consumption
        region: Region for carbon intensity calculation
        device: Device to train on (auto-detect if None)
    
    Returns:
        TrainingResult with all metrics
    """
    # Get config
    config = get_model_config(model_name)
    if config is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create model
    model = get_model(
        config.architecture,
        num_classes=config.num_classes,
        pretrained=config.pretrained
    )
    
    # Get dataloaders
    train_loader, test_loader = get_dataloaders(
        config.dataset,
        batch_size=config.batch_size
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        track_energy=track_energy,
        region=region
    )
    
    # Train
    result = trainer.train(train_loader, test_loader)
    
    return result


def load_model(model_name: str, device: str = None) -> nn.Module:
    """Load a trained model from checkpoint"""
    config = get_model_config(model_name)
    if config is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create model
    model = get_model(
        config.architecture,
        num_classes=config.num_classes
    )
    
    # Load checkpoint
    checkpoint_path = get_model_save_path(config.name)
    if checkpoint_path.exists():
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
        print(f"  Accuracy: {checkpoint.get('accuracy', 'N/A')}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
    
    return model


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Test training pipeline
    print("Carbon Footprint AI Framework - Training Pipeline")
    print("=" * 60)
    
    # Check available datasets
    print(f"TorchVision available: {TORCHVISION_AVAILABLE}")
    print(f"HuggingFace datasets available: {HF_DATASETS_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Quick test with SimpleCNN if run directly
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("\nRunning quick test with SimpleCNN...")
        result = train_model("simple_cnn", track_energy=True, region="india")
        print(f"\nFinal Results:")
        print(f"  Accuracy: {result.final_accuracy:.2f}%")
        print(f"  Energy: {result.total_energy_kwh*1000:.4f} Wh")
        print(f"  CO2: {result.total_co2_kg*1000:.4f} g")
