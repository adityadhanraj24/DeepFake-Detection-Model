import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Grad-CAM Class to handle hooks and heatmap generation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        # Pool the gradients across channels
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        
        # Calculate weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # Apply ReLU to keep features that positively contribute to the class
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        
        # Normalize between 0 and 1
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam, class_idx

# 3. Visualization Helper Function
def overlay_heatmap(img_path, heatmap, pred_class, output_path="gradcam_result.png"):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    # Convert heatmap to RGB colorspace
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on original image
    overlayed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # Save and display plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Grad-CAM (Predicted: {pred_class})")
    plt.imshow(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.savefig(output_path, bbox_inches='tight')
    print(f"[✓] Grad-CAM visualization saved successfully to: {output_path}")
    plt.show()

# 4. Main Execution Block
if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(BASE_DIR, "models", "mobilenetv2_deepfake.pth")
    
    # CHANGE THIS: Path to any single image from your test set you want to analyze
    sample_image_path = os.path.join(BASE_DIR, "DataSet", "Dataset", "Test", "Fake", "fake_0.jpg")

    if not os.path.exists(model_path):
        print("[-] Error: Model file not found. Make sure train_model.py finishes execution first!")
        exit()

    # Initialize and load model architecture
    model = models.mobilenet_v2(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Target the last conv layer of MobileNetV2 features block
    target_layer = model.features[-1]
    
    # Preprocess the individual image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(sample_image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate Grad-CAM Map
        cam_extractor = GradCAM(model, target_layer)
        heatmap, predicted_idx = cam_extractor.generate_heatmap(input_tensor)
        
        # Map class index back to text label (Assumes 0: Fake, 1: Real based on alphabetical order)
        classes = ["Fake", "Real"]
        predicted_label = classes[predicted_idx]
        
        # Save output visual presentation
        output_img_path = os.path.join(BASE_DIR, "gradcam_output.png")
        overlay_heatmap(sample_image_path, heatmap, predicted_label, output_path=output_img_path)
        
    except FileNotFoundError:
        print(f"[-] Could not find sample image at: {sample_image_path}")
        print("[!] Please update 'sample_image_path' with a valid filename from your Test folder to try it out.")
