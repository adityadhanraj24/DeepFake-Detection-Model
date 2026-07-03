import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# 1. Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
test_dir = os.path.join(BASE_DIR, "DataSet", "Dataset", "Test")
model_path = os.path.join(BASE_DIR, "models", "mobilenetv2_deepfake.pth")

# 2. Custom Dataset to apply OpenCV distortions on-the-fly
class DistortedDataset(Dataset):
    def __init__(self, base_dataset, distortion_type=None, severity=None):
        self.base_dataset = base_dataset
        self.distortion_type = distortion_type
        self.severity = severity
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        # Using a subset of 1000 images for faster benchmarking
        return min(1000, len(self.base_dataset))

    def __getitem__(self, idx):
        img_path, label = self.base_dataset.samples[idx]
        # Read image via OpenCV
        cv_img = cv2.imread(img_path)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # Apply Distortions
        if self.distortion_type == "jpeg":
            # Lower quality means higher compression (severity 10 = very low quality)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.severity]
            _, encimg = cv2.imencode('.jpg', cv_img, encode_param)
            cv_img = cv2.imdecode(encimg, 1)
        
        elif self.distortion_type == "blur":
            # Kernel size must be odd
            k_size = self.severity
            cv_img = cv2.GaussianBlur(cv_img, (k_size, k_size), 0)

        # Convert back to PIL Image for standard PyTorch Transforms
        pil_img = Image.fromarray(cv_img)
        return self.transform(pil_img), label

# 3. Load Base Test Data and Model
from torchvision.datasets import ImageFolder
raw_test_dataset = ImageFolder(test_dir)

model = models.mobilenet_v2(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 4. Evaluation Function
def evaluate_robustness(dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100

# 5. Run Experiments
print("--- Baseline Test Accuracy (Clean Images): 86.91% ---")

# Experiment A: Heavy JPEG Compression (Quality = 15)
jpeg_dataset = DistortedDataset(raw_test_dataset, distortion_type="jpeg", severity=15)
jpeg_loader = DataLoader(jpeg_dataset, batch_size=32, shuffle=False)
jpeg_acc = evaluate_robustness(jpeg_loader)
print(f"[Result] Accuracy under severe JPEG compression (Quality 15): {jpeg_acc:.2f}%")

# Experiment B: Severe Motion/Gaussian Blur (Kernel = 15x15)
blur_dataset = DistortedDataset(raw_test_dataset, distortion_type="blur", severity=15)
blur_loader = DataLoader(blur_dataset, batch_size=32, shuffle=False)
blur_acc = evaluate_robustness(blur_loader)
print(f"[Result] Accuracy under severe Gaussian Blur (15x15 Kernel): {blur_acc:.2f}%")