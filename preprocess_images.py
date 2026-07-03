import cv2
import os
from pathlib import Path

# Define base paths
base_input = "DataSet/Dataset"
base_output = "processed"

# Categories to process
categories = ["Train", "Test", "Validation"]
types = ["Real", "Fake"]

processed_count = 0
error_count = 0

for category in categories:
    for img_type in types:
        input_folder = os.path.join(base_input, category, img_type)
        output_folder = os.path.join(base_output, category, img_type)
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        if not os.path.exists(input_folder):
            print(f"Warning: {input_folder} does not exist")
            continue
        
        files = os.listdir(input_folder)
        print(f"Processing {category}/{img_type}: {len(files)} files")
        
        for file in files:
            try:
                input_path = os.path.join(input_folder, file)
                output_path = os.path.join(output_folder, file)
                
                img = cv2.imread(input_path)
                if img is None:
                    error_count += 1
                    continue
                    
                img = cv2.resize(img, (224, 224))
                cv2.imwrite(output_path, img)
                processed_count += 1
                
                if processed_count % 10000 == 0:
                    print(f"  Processed {processed_count} images...")
            except Exception as e:
                print(f"Error processing {file}: {e}")
                error_count += 1

print(f"\nPreprocessing complete!")
print(f"Successfully processed: {processed_count} images")
print(f"Errors: {error_count} images")