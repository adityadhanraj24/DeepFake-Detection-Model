import kagglehub
import os

# Specify the DataSet folder as download destination
dataset_folder = os.path.join(os.path.dirname(__file__), "DataSet")
os.makedirs(dataset_folder, exist_ok=True)

# Download latest version to DataSet folder
path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images", path=dataset_folder)

print("Path to dataset files:", path)