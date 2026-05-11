"""
Carbon Footprint AI Framework
A research framework for measuring and reducing AI model carbon emissions.
"""

from .config import *
from .models import get_model, get_model_info
from .train import train_model, Trainer
from .optimize import prune_model, quantize_model, distill_model
from .metrics import MetricsProcessor, calculate_green_ai_score
from .energy_tracker import EnergyTracker, track_energy
from .visualize import Visualizer

__version__ = "1.0.0"
__author__ = "Carbon Footprint AI Research"
