"""
Energy Tracking Module for Carbon Footprint AI Framework
Combines CodeCarbon with custom GPU monitoring via pynvml
"""
import warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import json
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

import torch
import numpy as np

# Try to import energy tracking libraries
try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("Warning: codecarbon not installed. Using fallback energy estimation.")

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU power monitoring disabled.")

from config import (
    CARBON_INTENSITY, DEFAULT_REGION, RESULTS_DIR, 
    ENERGY_LOG_FILE, EXPERIMENT_CONFIG
)


@dataclass
class EnergyMetrics:
    """Container for energy measurement results"""
    energy_kwh: float = 0.0
    co2_kg: float = 0.0
    duration_seconds: float = 0.0
    gpu_energy_kwh: float = 0.0
    cpu_energy_kwh: float = 0.0
    avg_gpu_power_w: float = 0.0
    max_gpu_power_w: float = 0.0
    avg_gpu_utilization: float = 0.0
    avg_gpu_memory_used_mb: float = 0.0
    region: str = DEFAULT_REGION
    carbon_intensity: float = CARBON_INTENSITY[DEFAULT_REGION]
    
    # Per-epoch tracking
    epoch_energies: List[float] = field(default_factory=list)
    epoch_co2: List[float] = field(default_factory=list)
    epoch_durations: List[float] = field(default_factory=list)
    
    # GPU samples
    gpu_power_samples: List[float] = field(default_factory=list)
    gpu_utilization_samples: List[float] = field(default_factory=list)
    gpu_memory_samples: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def calculate_regional_emissions(self) -> Dict[str, float]:
        """Calculate CO2 emissions for different regions"""
        return {
            region: self.energy_kwh * intensity
            for region, intensity in CARBON_INTENSITY.items()
        }


class GPUMonitor:
    """Real-time GPU power and utilization monitoring using pynvml"""
    
    def __init__(self, device_index: int = 0, interval: float = 0.5):
        self.device_index = device_index
        self.interval = interval
        self.running = False
        self.thread = None
        
        self.power_samples = []
        self.utilization_samples = []
        self.memory_samples = []
        self.timestamps = []
        
        if PYNVML_AVAILABLE:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.device_name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(self.device_name, bytes):
                self.device_name = self.device_name.decode()
        else:
            self.handle = None
            self.device_name = "Unknown"
    
    def _monitor_loop(self):
        """Background thread for GPU monitoring"""
        while self.running:
            try:
                if self.handle:
                    # Get power usage (in milliwatts)
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                    power_w = power_mw / 1000.0
                    
                    # Get utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    gpu_util = util.gpu
                    
                    # Get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    mem_used_mb = mem_info.used / (1024 ** 2)
                    
                    self.power_samples.append(power_w)
                    self.utilization_samples.append(gpu_util)
                    self.memory_samples.append(mem_used_mb)
                    self.timestamps.append(time.time())
                    
            except Exception as e:
                pass  # Silently continue on errors
            
            time.sleep(self.interval)
    
    def start(self):
        """Start GPU monitoring in background thread"""
        self.running = True
        self.power_samples = []
        self.utilization_samples = []
        self.memory_samples = []
        self.timestamps = []
        
        if PYNVML_AVAILABLE:
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return statistics"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        stats = {
            "device_name": self.device_name,
            "num_samples": len(self.power_samples),
            "avg_power_w": np.mean(self.power_samples) if self.power_samples else 0,
            "max_power_w": np.max(self.power_samples) if self.power_samples else 0,
            "min_power_w": np.min(self.power_samples) if self.power_samples else 0,
            "avg_utilization": np.mean(self.utilization_samples) if self.utilization_samples else 0,
            "avg_memory_mb": np.mean(self.memory_samples) if self.memory_samples else 0,
            "max_memory_mb": np.max(self.memory_samples) if self.memory_samples else 0,
            "power_samples": self.power_samples,
            "utilization_samples": self.utilization_samples,
            "memory_samples": self.memory_samples,
        }
        
        return stats
    
    def get_current_power(self) -> float:
        """Get current GPU power draw in watts"""
        if self.handle:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                return power_mw / 1000.0
            except Exception:
                pass
        return 0.0


class EnergyTracker:
    """
    Main energy tracking class combining CodeCarbon and custom GPU monitoring.
    Supports per-epoch tracking, regional carbon intensity analysis, and detailed logging.
    """
    
    def __init__(
        self,
        experiment_name: str = "experiment",
        region: str = DEFAULT_REGION,
        track_gpu: bool = True,
        save_to_file: bool = True,
        log_dir: Optional[Path] = None
    ):
        self.experiment_name = experiment_name
        self.region = region
        self.carbon_intensity = CARBON_INTENSITY.get(region, CARBON_INTENSITY["global_avg"])
        self.track_gpu = track_gpu
        self.save_to_file = save_to_file
        self.log_dir = log_dir or RESULTS_DIR
        
        # Initialize trackers
        self.emissions_tracker = None
        self.gpu_monitor = None
        
        # Timing
        self.start_time = None
        self.end_time = None
        
        # Results
        self.metrics = EnergyMetrics(region=region, carbon_intensity=self.carbon_intensity)
        
        # Epoch tracking
        self.current_epoch = 0
        self.epoch_start_time = None
        self.epoch_start_energy = 0.0
        
        # Initialize CodeCarbon if available
        if CODECARBON_AVAILABLE:
            try:
                self.emissions_tracker = OfflineEmissionsTracker(
                    country_iso_code="IND" if region == "india" else "USA",
                    log_level="error",
                    save_to_file=False,
                    tracking_mode="process"
                )
            except Exception:
                # Fallback to basic tracker
                self.emissions_tracker = None
        
        # Initialize GPU monitor
        if track_gpu and PYNVML_AVAILABLE and torch.cuda.is_available():
            self.gpu_monitor = GPUMonitor(
                device_index=0,
                interval=EXPERIMENT_CONFIG.gpu_monitoring_interval
            )
    
    def start(self):
        """Start energy tracking"""
        self.start_time = time.time()
        self.metrics = EnergyMetrics(region=self.region, carbon_intensity=self.carbon_intensity)
        
        # Start CodeCarbon
        if self.emissions_tracker:
            try:
                self.emissions_tracker.start()
            except Exception:
                pass
        
        # Start GPU monitoring
        if self.gpu_monitor:
            self.gpu_monitor.start()
        
        return self
    
    def stop(self) -> EnergyMetrics:
        """Stop tracking and calculate final metrics"""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        # Stop CodeCarbon and get emissions
        codecarbon_energy = 0.0
        codecarbon_co2 = 0.0
        
        if self.emissions_tracker:
            try:
                emissions = self.emissions_tracker.stop()
                if emissions is not None:
                    codecarbon_co2 = emissions  # kg CO2
                    # Estimate energy from CO2 and carbon intensity
                    codecarbon_energy = codecarbon_co2 / self.carbon_intensity
            except Exception:
                pass
        
        # Stop GPU monitoring and get stats
        gpu_stats = {}
        gpu_energy = 0.0
        
        if self.gpu_monitor:
            gpu_stats = self.gpu_monitor.stop()
            # Calculate GPU energy from power samples
            if gpu_stats["power_samples"]:
                # Energy = Power * Time (convert W*s to kWh)
                avg_power = gpu_stats["avg_power_w"]
                gpu_energy = (avg_power * duration) / (1000 * 3600)  # kWh
        
        # Use best available energy estimate
        if codecarbon_energy > 0:
            total_energy = codecarbon_energy
        elif gpu_energy > 0:
            # Add estimated CPU overhead (typically 20-30% of GPU for ML workloads)
            total_energy = gpu_energy * 1.25
        else:
            # Fallback: estimate based on typical GPU power (150W average)
            estimated_power = 150 if torch.cuda.is_available() else 65
            total_energy = (estimated_power * duration) / (1000 * 3600)
        
        # Calculate CO2 emissions
        total_co2 = total_energy * self.carbon_intensity
        
        # Update metrics
        self.metrics.energy_kwh = total_energy
        self.metrics.co2_kg = total_co2
        self.metrics.duration_seconds = duration
        self.metrics.gpu_energy_kwh = gpu_energy
        self.metrics.cpu_energy_kwh = max(0, total_energy - gpu_energy)
        
        if gpu_stats:
            self.metrics.avg_gpu_power_w = gpu_stats.get("avg_power_w", 0)
            self.metrics.max_gpu_power_w = gpu_stats.get("max_power_w", 0)
            self.metrics.avg_gpu_utilization = gpu_stats.get("avg_utilization", 0)
            self.metrics.avg_gpu_memory_used_mb = gpu_stats.get("avg_memory_mb", 0)
            self.metrics.gpu_power_samples = gpu_stats.get("power_samples", [])
            self.metrics.gpu_utilization_samples = gpu_stats.get("utilization_samples", [])
            self.metrics.gpu_memory_samples = gpu_stats.get("memory_samples", [])
        
        # Save results if requested
        if self.save_to_file:
            self._save_results()
        
        return self.metrics
    
    def start_epoch(self, epoch: int):
        """Start tracking for a new epoch"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.epoch_start_energy = self._get_current_energy_estimate()
    
    def end_epoch(self) -> Dict[str, float]:
        """End epoch tracking and record metrics"""
        if self.epoch_start_time is None:
            return {}
        
        epoch_duration = time.time() - self.epoch_start_time
        epoch_energy = self._get_current_energy_estimate() - self.epoch_start_energy
        epoch_co2 = epoch_energy * self.carbon_intensity
        
        self.metrics.epoch_durations.append(epoch_duration)
        self.metrics.epoch_energies.append(epoch_energy)
        self.metrics.epoch_co2.append(epoch_co2)
        
        return {
            "epoch": self.current_epoch,
            "duration": epoch_duration,
            "energy_kwh": epoch_energy,
            "co2_kg": epoch_co2
        }
    
    def _get_current_energy_estimate(self) -> float:
        """Get current cumulative energy estimate"""
        if self.start_time is None:
            return 0.0
        
        duration = time.time() - self.start_time
        
        if self.gpu_monitor and self.gpu_monitor.power_samples:
            avg_power = np.mean(self.gpu_monitor.power_samples)
            return (avg_power * 1.25 * duration) / (1000 * 3600)
        else:
            estimated_power = 150 if torch.cuda.is_available() else 65
            return (estimated_power * duration) / (1000 * 3600)
    
    def _save_results(self):
        """Save results to JSON file"""
        results = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics.to_dict(),
            "regional_emissions": self.metrics.calculate_regional_emissions()
        }
        
        log_file = self.log_dir / f"{self.experiment_name}_energy.json"
        
        with open(log_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
    
    def get_summary(self) -> str:
        """Get human-readable summary of energy metrics"""
        m = self.metrics
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║              Energy Tracking Summary                          ║
╠══════════════════════════════════════════════════════════════╣
║  Experiment: {self.experiment_name:<45} ║
║  Duration: {m.duration_seconds:.2f} seconds ({m.duration_seconds/60:.2f} minutes)          ║
╠══════════════════════════════════════════════════════════════╣
║  ENERGY CONSUMPTION                                           ║
║  ├─ Total Energy: {m.energy_kwh*1000:.4f} Wh ({m.energy_kwh:.6f} kWh)       ║
║  ├─ GPU Energy: {m.gpu_energy_kwh*1000:.4f} Wh                           ║
║  └─ CPU Energy: {m.cpu_energy_kwh*1000:.4f} Wh                           ║
╠══════════════════════════════════════════════════════════════╣
║  CARBON EMISSIONS                                             ║
║  ├─ CO2 Emitted: {m.co2_kg*1000:.4f} g ({m.co2_kg:.6f} kg)          ║
║  └─ Carbon Intensity: {m.carbon_intensity:.2f} kgCO2/kWh ({m.region})     ║
╠══════════════════════════════════════════════════════════════╣
║  GPU STATISTICS                                               ║
║  ├─ Avg Power: {m.avg_gpu_power_w:.2f} W                              ║
║  ├─ Max Power: {m.max_gpu_power_w:.2f} W                              ║
║  ├─ Avg Utilization: {m.avg_gpu_utilization:.1f}%                         ║
║  └─ Avg Memory: {m.avg_gpu_memory_used_mb:.0f} MB                          ║
╚══════════════════════════════════════════════════════════════╝
"""
        return summary
    
    @contextmanager
    def track(self):
        """Context manager for energy tracking"""
        self.start()
        try:
            yield self
        finally:
            self.stop()


@contextmanager
def track_energy(
    experiment_name: str = "experiment",
    region: str = DEFAULT_REGION,
    track_gpu: bool = True
):
    """
    Convenient context manager for energy tracking.
    
    Usage:
        with track_energy("my_experiment") as tracker:
            # Your training code here
            pass
        metrics = tracker.metrics
    """
    tracker = EnergyTracker(
        experiment_name=experiment_name,
        region=region,
        track_gpu=track_gpu
    )
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()
        print(tracker.get_summary())


def estimate_training_carbon(
    model_params: int,
    epochs: int,
    batch_size: int,
    dataset_size: int,
    gpu_power_w: float = 150,
    region: str = DEFAULT_REGION
) -> Dict[str, float]:
    """
    Estimate carbon emissions for a training run before executing.
    
    Args:
        model_params: Number of model parameters
        epochs: Number of training epochs
        batch_size: Training batch size
        dataset_size: Number of training samples
        gpu_power_w: Expected GPU power draw in watts
        region: Region for carbon intensity
    
    Returns:
        Dictionary with estimated energy and CO2 emissions
    """
    # Estimate training time based on params and data
    # Rough heuristic: ~1e-9 seconds per param per sample
    batches_per_epoch = dataset_size // batch_size
    time_per_batch = model_params * 1e-9 * batch_size * 3  # forward + backward + update
    estimated_time = epochs * batches_per_epoch * time_per_batch
    
    # Add overhead for data loading, logging, etc. (20%)
    estimated_time *= 1.2
    
    # Calculate energy
    energy_kwh = (gpu_power_w * estimated_time) / (1000 * 3600)
    
    # Calculate CO2
    carbon_intensity = CARBON_INTENSITY.get(region, CARBON_INTENSITY["global_avg"])
    co2_kg = energy_kwh * carbon_intensity
    
    return {
        "estimated_time_minutes": estimated_time / 60,
        "estimated_energy_kwh": energy_kwh,
        "estimated_co2_kg": co2_kg,
        "carbon_intensity": carbon_intensity,
        "region": region
    }


if __name__ == "__main__":
    # Test energy tracking
    print("Testing Energy Tracker...")
    print(f"CodeCarbon available: {CODECARBON_AVAILABLE}")
    print(f"pynvml available: {PYNVML_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Quick test
    with track_energy("test_experiment", region="india") as tracker:
        # Simulate some computation
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000, device="cuda")
            for _ in range(100):
                x = torch.matmul(x, x)
                x = x / x.max()
        else:
            x = torch.randn(1000, 1000)
            for _ in range(100):
                x = torch.matmul(x, x)
                x = x / x.max()
        
        time.sleep(2)  # Wait for monitoring samples
    
    # Print regional comparison
    print("\nRegional Emissions Comparison:")
    regional = tracker.metrics.calculate_regional_emissions()
    for region, co2 in regional.items():
        print(f"  {region}: {co2*1000:.4f} g CO2")
