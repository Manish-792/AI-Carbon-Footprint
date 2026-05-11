"""
Model Definitions for Carbon Footprint AI Framework
Contains CNN, ResNet, MobileNet, and Transformer architectures
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Try to import transformers for BERT models
try:
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertConfig,
        AutoModelForSequenceClassification,
        AutoConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Text models unavailable.")

from config import CIFAR10_MODELS, TEXT_MODELS, ModelConfig


@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    num_params: int
    flops: int  # Approximate FLOPs for one forward pass
    model_size_mb: float
    architecture_type: str


# ============================================================================
# SIMPLE CNN (Baseline)
# ============================================================================
class SimpleCNN(nn.Module):
    """
    Lightweight baseline CNN for CIFAR-10.
    ~62K parameters - serves as the low-carbon baseline.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# RESNET-18 (Modified for CIFAR-10)
# ============================================================================
class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32 images).
    ~11M parameters - standard architecture.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolution (smaller kernel for CIFAR)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================
# MOBILENET V2 (Efficient Architecture)
# ============================================================================
class InvertedResidual(nn.Module):
    """Inverted residual block for MobileNetV2"""
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        layers.extend([
            # Depthwise conv
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise conv
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 adapted for CIFAR-10.
    ~2.2M parameters - efficient mobile architecture.
    """
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        
        # Setting for inverted residual blocks
        # t: expansion factor, c: output channels, n: repeat, s: stride
        settings = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # Changed stride from 2 to 1 for CIFAR
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult)
        
        # First layer
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]
        
        # Inverted residual blocks
        for t, c, n, s in settings:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
        
        # Last layers
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, 1, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        ))
        
        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.last_channel, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# TRANSFORMER MODELS (BERT variants for Text Classification)
# ============================================================================
class TextClassifier(nn.Module):
    """
    Wrapper for transformer-based text classifiers.
    Supports DistilBERT and TinyBERT.
    """
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 2,
        pretrained: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for text models")
        
        if pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes
            )
        else:
            config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
            self.model = AutoModelForSequenceClassification.from_config(config)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


def get_distilbert(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Get DistilBERT model for text classification"""
    return TextClassifier(
        model_name="distilbert-base-uncased",
        num_classes=num_classes,
        pretrained=pretrained
    )


def get_tinybert(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Get TinyBERT model for text classification"""
    # Using a smaller BERT variant
    return TextClassifier(
        model_name="huawei-noah/TinyBERT_General_4L_312D",
        num_classes=num_classes,
        pretrained=pretrained
    )


# ============================================================================
# MODEL FACTORY
# ============================================================================
def get_model(
    architecture: str,
    num_classes: int = 10,
    pretrained: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models by architecture name.
    
    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for transformers)
    
    Returns:
        PyTorch model instance
    """
    architecture = architecture.lower().replace("-", "_").replace(" ", "_")
    
    if architecture == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    
    elif architecture == "resnet18":
        return ResNet18(num_classes=num_classes)
    
    elif architecture in ["resnet18_pruned", "resnet18_quantized"]:
        # Return base ResNet18, optimization applied separately
        return ResNet18(num_classes=num_classes)
    
    elif architecture == "mobilenetv2":
        return MobileNetV2(num_classes=num_classes)
    
    elif architecture == "mobilenetv2_distilled":
        return MobileNetV2(num_classes=num_classes)
    
    elif architecture == "distilbert":
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required for DistilBERT")
        return get_distilbert(num_classes=num_classes, pretrained=pretrained)
    
    elif architecture == "tinybert":
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required for TinyBERT")
        return get_tinybert(num_classes=num_classes, pretrained=pretrained)
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def get_model_info(model: nn.Module, input_size: Tuple = (1, 3, 32, 32)) -> ModelInfo:
    """
    Get information about a model including parameter count and estimated FLOPs.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size for FLOP calculation
    
    Returns:
        ModelInfo dataclass with model statistics
    """
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    # Estimate FLOPs (simplified calculation)
    # For more accurate FLOPs, use tools like thop or fvcore
    flops = estimate_flops(model, input_size)
    
    # Determine architecture type
    class_name = model.__class__.__name__
    if "CNN" in class_name:
        arch_type = "cnn"
    elif "ResNet" in class_name:
        arch_type = "resnet"
    elif "MobileNet" in class_name:
        arch_type = "mobilenet"
    elif "TextClassifier" in class_name or "Bert" in class_name.lower():
        arch_type = "transformer"
    else:
        arch_type = "unknown"
    
    return ModelInfo(
        name=class_name,
        num_params=num_params,
        flops=flops,
        model_size_mb=model_size_mb,
        architecture_type=arch_type
    )


def estimate_flops(model: nn.Module, input_size: Tuple) -> int:
    """
    Estimate FLOPs for a model (simplified calculation).
    
    For convolutional layers: 2 * K^2 * C_in * C_out * H * W
    For linear layers: 2 * input_features * output_features
    """
    total_flops = 0
    
    def hook_fn(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, nn.Conv2d):
            batch, out_c, out_h, out_w = output.shape
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            total_flops += 2 * kernel_ops * out_c * out_h * out_w
        
        elif isinstance(module, nn.Linear):
            total_flops += 2 * module.in_features * module.out_features
    
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Run forward pass
    device = next(model.parameters()).device if len(list(model.parameters())) > 0 else "cpu"
    dummy_input = torch.randn(input_size).to(device)
    
    model.eval()
    with torch.no_grad():
        try:
            model(dummy_input)
        except Exception:
            # For transformer models with different input
            pass
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_flops


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters by layer type"""
    param_counts = {
        "conv": 0,
        "bn": 0,
        "linear": 0,
        "embedding": 0,
        "other": 0,
        "total": 0,
        "trainable": 0
    }
    
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            num_params = param.numel()
            param_counts["total"] += num_params
            
            if param.requires_grad:
                param_counts["trainable"] += num_params
            
            if isinstance(module, nn.Conv2d):
                param_counts["conv"] += num_params
            elif isinstance(module, nn.BatchNorm2d):
                param_counts["bn"] += num_params
            elif isinstance(module, nn.Linear):
                param_counts["linear"] += num_params
            elif isinstance(module, nn.Embedding):
                param_counts["embedding"] += num_params
            else:
                param_counts["other"] += num_params
    
    return param_counts


def print_model_summary(model: nn.Module, input_size: Tuple = (1, 3, 32, 32)):
    """Print detailed model summary"""
    info = get_model_info(model, input_size)
    param_counts = count_parameters(model)
    
    print("=" * 60)
    print(f"Model: {info.name}")
    print("=" * 60)
    print(f"Architecture Type: {info.architecture_type}")
    print(f"Total Parameters: {info.num_params:,}")
    print(f"Trainable Parameters: {param_counts['trainable']:,}")
    print(f"Model Size: {info.model_size_mb:.2f} MB")
    print(f"Estimated FLOPs: {info.flops:,}")
    print("-" * 60)
    print("Parameters by Layer Type:")
    print(f"  Convolutional: {param_counts['conv']:,}")
    print(f"  BatchNorm: {param_counts['bn']:,}")
    print(f"  Linear: {param_counts['linear']:,}")
    print(f"  Embedding: {param_counts['embedding']:,}")
    print(f"  Other: {param_counts['other']:,}")
    print("=" * 60)


# ============================================================================
# MAIN - Test model creation
# ============================================================================
if __name__ == "__main__":
    print("Testing Model Definitions...")
    print()
    
    # Test SimpleCNN
    print("1. SimpleCNN")
    model = SimpleCNN(num_classes=10)
    print_model_summary(model, (1, 3, 32, 32))
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print()
    
    # Test ResNet18
    print("2. ResNet-18")
    model = ResNet18(num_classes=10)
    print_model_summary(model, (1, 3, 32, 32))
    out = model(x)
    print(f"Output shape: {out.shape}")
    print()
    
    # Test MobileNetV2
    print("3. MobileNetV2")
    model = MobileNetV2(num_classes=10)
    print_model_summary(model, (1, 3, 32, 32))
    out = model(x)
    print(f"Output shape: {out.shape}")
    print()
    
    # Test factory function
    print("4. Testing get_model factory...")
    for arch in ["simple_cnn", "resnet18", "mobilenetv2"]:
        model = get_model(arch, num_classes=10)
        info = get_model_info(model)
        print(f"  {arch}: {info.num_params:,} params, {info.model_size_mb:.2f} MB")
    
    # Test transformer models if available
    if TRANSFORMERS_AVAILABLE:
        print("\n5. Transformer Models")
        print("  DistilBERT and TinyBERT available for text classification")
