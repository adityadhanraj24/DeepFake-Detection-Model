import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Lightweight Deepfake Detector",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Lightweight Deepfake Detection Dashboard")
st.markdown("Powered by an optimized MobileNetV2 architecture with Explainable AI (Grad-CAM).")
st.sidebar.header("System Settings")

# 2. Setup Device & Robust Pathing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenetv2_deepfake.pth")

# 3. Load Model with Caching to prevent re-loading on every click
@st.cache_resource
def load_deepfake_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure training is complete.")
        return None
    
    # Reconstruct MobileNetV2 Binary Classifier
    model = models.mobilenet_v2(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    
    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_deepfake_model()

# 4. Grad-CAM Class Engine
class StreamlitGradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = model.features[-1]
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam

# 5. UI File Upload Component
uploaded_file = st.sidebar.file_uploader("Upload a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and prepare image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Define MobileNet evaluation transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Layout Columns for Results Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Original Uploaded Image")
        st.image(image, use_container_width=True)
        
    # Run Inference
    if model is not None:
        with st.spinner("Analyzing image patterns..."):
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                pred_idx = torch.argmax(probabilities).item()
                
            classes = ["Fake", "Real"]  # Order mapping matches PyTorch ImageFolder alphabetics
            prediction = classes[pred_idx]
            confidence = probabilities[pred_idx].item() * 100
            
            # Generate Heatmap data using gradients
            # Turn on gradients explicitly just for Grad-CAM execution pass
            model.zero_grad()
            g_cam = StreamlitGradCAM(model)
            heatmap = g_cam.generate_heatmap(input_tensor, pred_idx)
            
            # Superimpose Heatmap over original raw image
            open_cv_img = np.array(image)
            open_cv_img = cv2.resize(open_cv_img, (224, 224))
            open_cv_img = cv2.cvtColor(open_cv_img, cv2.COLOR_RGB2BGR)
            
            heatmap_color = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
            overlayed = cv2.addWeighted(open_cv_img, 0.6, heatmap_color, 0.4, 0)
            overlayed_rgb = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
            
        with col2:
            st.subheader("🧠 Explainable AI Diagnostic")
            st.image(overlayed_rgb, use_container_width=True, caption="Grad-CAM Visualization (Red areas show anomalies)")
            
        # Display Final Prediction Alert banner below metrics layout
        st.write("---")
        if prediction == "Real":
            st.success(f"### Result: Verified **REAL** Face (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"### Result: Warning! **FAKE/MANIPULATED** Face Detected (Confidence: {confidence:.2f}%)")
            
        # Optional Metrics Breakdown details drop-down panel
        with st.expander("See Raw Model Probabilities"):
            st.write(f"Class 0 (Fake): {probabilities[0].item()*100:.2f}%")
            st.write(f"Class 1 (Real): {probabilities[1].item()*100:.2f}%")
else:
    st.info("💡 Please upload an image from the sidebar panel to check for deepfake manipulation markers.")