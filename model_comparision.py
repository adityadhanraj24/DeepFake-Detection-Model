import os
import time
import torch
import torch.nn as nn
from torchvision import models
import numpy as np

# Force CPU to evaluate deployment metrics fairly
device = torch.device("cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 1. Initialize Architectures with 2-class heads
def get_mobilenet_v2():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def get_efficientnet_b0():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def get_shufflenet_v2():
    model = models.shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

architectures = {
    "MobileNetV2": get_mobilenet_v2(),
    "EfficientNet-B0": get_efficientnet_b0(),
    "ShuffleNetV2": get_shufflenet_v2()
}

# 2. Benchmark Loop
dummy_input = torch.randn(1, 3, 224, 224)

print("=== Running Model Comparison Study (Forced CPU) ===\n")
print(f"{'Architecture':<18} | {'Params (M)':<12} | {'Avg Latency':<15} | {'Throughput':<10}")
print("-" * 65)

for name, model in architectures.items():
    model.eval()
    params = count_parameters(model) / 1e6  # Millions of params
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
            
    # Measure Latency
    timings = []
    with torch.no_grad():
        for _ in range(50):
            start = time.time()
            _ = model(dummy_input)
            timings.append(time.time() - start)
            
    avg_latency_ms = np.mean(timings) * 1000
    fps = 1000 / avg_latency_ms
    
    print(f"{name:<18} | {params:<12.2f} | {avg_latency_ms:<12.2f} ms | {fps:.2f} FPS")

print("\n[✓] Benchmark matrix complete! You can copy this table straight into your report.")