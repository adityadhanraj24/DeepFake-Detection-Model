import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 1. Hardware Detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Dynamic Path Configuration pointing to your actual root-level folders
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_dir = os.path.join(BASE_DIR, "DataSet", "Dataset", "Train")
val_dir = os.path.join(BASE_DIR, "DataSet", "Dataset", "Validation")

print(f"Targeting training directory: {train_dir}")
print(f"Targeting validation directory: {val_dir}")

# 3. Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

# 4. Load Datasets using ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

# 5. Data Loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

print("Detected Classes:", train_dataset.classes)

# 6. Initialize MobileNetV2 Architecture
model = models.mobilenet_v2(weights="DEFAULT")

# Replace Final Classifier Layer for Binary Classification
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)
model = model.to(device)

# 7. Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 8. Training Loop
epochs = 3
num_train_batches = len(train_loader)

for epoch in range(epochs):
    # --- TRAINING PHASE ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    average_train_loss = running_loss / num_train_batches

    # --- VALIDATION PHASE ---
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {average_train_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Acc: {val_acc:.2f}%"
    )

# 9. Save Model Weights
output_model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(output_model_dir, exist_ok=True)
model_path = os.path.join(output_model_dir, "mobilenetv2_deepfake.pth")

torch.save(model.state_dict(), model_path)
print(f"Model saved successfully at: {model_path}")
