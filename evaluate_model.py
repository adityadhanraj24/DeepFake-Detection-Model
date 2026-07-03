import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device for evaluation:", device)

# 2. Paths Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
test_dir = os.path.join(BASE_DIR, "DataSet", "Dataset", "Test")
model_path = os.path.join(BASE_DIR, "models", "mobilenetv2_deepfake.pth")

# 3. Test Transformations (No augmentation, just resize and normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Load Test Dataset
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
print(f"Loaded test dataset from: {test_dir}")
print(f"Classes: {test_dataset.classes}")

# 5. Recreate MobileNetV2 Architecture
model = models.mobilenet_v2(weights=None)  # No need for default weights since we load our own
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

# Load the trained weights
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded trained model weights from {model_path}")
else:
    raise FileNotFoundError(f"Trained model file not found at {model_path}. Wait for training to complete first!")

model = model.to(device)
model.eval()

# 6. Evaluation Loop
all_preds = []
all_labels = []

print("\nEvaluating on test dataset... Please wait.")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# 7. Compute Metrics
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = 100 * np.sum(all_preds == all_labels) / len(all_labels)
print(f"\n=========================================")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"=========================================\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
