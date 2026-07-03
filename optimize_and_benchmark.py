import os
import time
import torch
import torch.nn as nn
from torchvision import models
import numpy as np

# 1. Force CPU for benchmarking lightweight deployment
device = torch.device("cpu")
print("Running optimization and benchmarking on:", device)

# 2. Paths Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "mobilenetv2_deepfake.pth")
onnx_path = os.path.join(BASE_DIR, "models", "mobilenetv2_deepfake.onnx")

# 3. Load Trained Model Architecture
model = models.mobilenet_v2(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("[✓] Loaded PyTorch model weights successfully.")
else:
    raise FileNotFoundError(f"Weights not found at {model_path}")

model.eval()

# 4. STEP 1: Convert/Export Model to ONNX Format
print("\n--- Step 1: Exporting Model to ONNX format ---")
# Create a dummy input tensor matching MobileNetV2 shape: (Batch_Size, Channels, Height, Width)
dummy_input = torch.randn(1, 3, 224, 224, device=device)

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"[✓] Model successfully exported and saved to: {onnx_path}")

# 5. STEP 2: CPU Latency Benchmarking
print("\n--- Step 2: Running CPU Latency Benchmark ---")
num_runs = 100
warmup_runs = 10

# Warm up the CPU (ignores initial caching latencies)
with torch.no_grad():
    for _ in range(warmup_runs):
        _ = model(dummy_input)

# Measure execution timings over 100 consecutive loops
timings = []
with torch.no_grad():
    for _ in range(num_runs):
        start_time = time.time()
        _ = model(dummy_input)
        end_time = time.time()
        timings.append(end_time - start_time)

# Calculate statistics
avg_latency_ms = np.mean(timings) * 1000
fps = 1000 / avg_latency_ms

print(f"==================================================")
print(f"Average CPU Inference Latency: {avg_latency_ms:.2f} ms per image")
print(f"Throughput (Frames Per Second): {fps:.2f} FPS")
print(f"==================================================\n")
print("Your project pipeline is now technically complete! You can mention these speed metrics in your project summary.")