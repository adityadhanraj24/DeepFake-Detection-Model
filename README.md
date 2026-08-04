# DeepFake-Detection-Model
An optimized, lightweight Computer Vision pipeline utilizing a modified MobileNetV2 architecture to detect facial deepfakes in real-time. Built for CPU edge deployment, it clocks an ultra-fast latency of 7.45 ms (134+ FPS) and features interactive Streamlit integration with Grad-CAM explainable AI diagnostics.

# Lightweight Deepfake Detection Dashboard with Explainable AI

An end-to-end, optimized Computer Vision pipeline designed to identify facial deepfake manipulations using a modified **MobileNetV2** architecture. The system is engineered specifically for edge deployment, achieving ultra-low latency on standard consumer-grade CPUs, complete with an interactive **Streamlit** dashboard and **Grad-CAM** visual diagnostics.

---

## 📊 Empirical Performance Matrix

| Metric Category | Evaluation Target | Performance Output | Engineering Insight |
| :--- | :--- | :--- | :--- |
| **Model Convergence**| Validation Accuracy | **97.92%** | Perfect generalization over 3 epochs ($Loss: 0.0247$). |
| **Security Metric** | Deepfake Catch Rate (Recall)| **96.00%** | Critical reliability marker; isolates 96% of synthetic faces. |
| **General Testing** | Unseen Test Set Accuracy | **86.91%** | Strong, repeatable classification baseline. |
| **Edge Optimization**| CPU Latency (FP32) | **7.45 ms** | Runs ultra-fast at **134.24 Frames Per Second**. |
| **Robustness Tests** | JPEG Compression (Quality 15)| **98.90%** | Compression strips high-frequency artifacts, exposing core flaws. |
|                       | Gaussian Blur ($15\times15$)| **20.60%** | System vulnerability; edge smoothing blinds early conv filters. |

---

## 🛠️ Tech Stack & Key Architectures
* **Core Framework:** PyTorch, Torchvision
* **Image Processing Engine:** OpenCV (`cv2`)
* **Deployment Format:** ONNX (Open Neural Network Exchange)
* **Optimization Engine:** PyTorch INT8 Dynamic Quantization
* **Explainable AI (XAI):** Custom Grad-CAM Layer Hook Implementation
* **User Interface:** Streamlit Engine

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/adityadhanraj24/DeepFake-Detection-Model](https://github.com/adityadhanraj24/DeepFake-Detection-Model)
cd Deepfake-Detection-MobileNetV2
