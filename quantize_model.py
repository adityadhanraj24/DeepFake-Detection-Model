import os
import time
import torch
import torch.nn as nn
from torchvision import models
import numpy as np

# Quantization benchmarking must be conducted on CPU
device = torch.device("cpu")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "mobilenetv2_deepfake.pth")
quant_model_path = os.path.join(BASE_DIR, "models", "quantized_mobilenetv2.pth")

# 1. Rebuild standard floating-point model
model_fp32 = models.mobilenet_v2(weights=None)
num_features = model_fp32.classifier[1].in_features
model_fp32.classifier[1] = nn.Linear(num_features, 2)
model_fp32.load_state_dict(torch.load(model_path, map_location=device))
model_fp32.eval()

# 2. Apply Dynamic INT8 Quantization to Linear Layer components
print("Applying PyTorch Dynamic Quantization (FP32 -> INT8)...")
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, 
    {nn.Linear}, 
    dtype=torch.qint8
)

# Save quantized weights
torch.save(model_int8.state_dict(), quant_model_path)

# 3. Size Comparison Analysis
size_fp32 = os.path.getsize(model_path) / (1024 * 1024)
size_int8 = os.path.getsize(quant_model_path) / (1024 * 1024)

print(f"\n================ SIZE COMPARISON ================")
print(f"Original FP32 Model Size: {size_fp32:.2f} MB")
print(f"Quantized INT8 Model Size: {size_int8:.2f} MB")
print(f"Size Reduction: {((size_fp32 - size_int8) / size_fp32) * 100:.2f}%")
print(f"=================================================")

# 4. Speed Latency Benchmarking
dummy_input = torch.randn(1, 3, 224, 224)

def benchmark_latency(model, name):
    # Warmup runs
    for _ in range(10):
        _ = model(dummy_input)
        
    timings = []
    for _ in range(100):
        start = time.time()
        _ = model(dummy_input)
        timings.append(time.time() - start)
        
    avg_latency = np.mean(timings) * 1000
    fps = 1000 / avg_latency
    print(f"{name} Performance -> Avg Latency: {avg_latency:.2f} ms | Throughput: {fps:.2f} FPS")

print("\n--- Running Latency Comparison Benchmark ---")
benchmark_latency(model_fp32, "Original Model (FP32)")
benchmark_latency(model_int8, "Quantized Model (INT8)")