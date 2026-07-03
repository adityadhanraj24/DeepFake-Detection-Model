import os
import shutil
import random

random.seed(42)

source_dir = "processed"
output_dir = "data"

# Map image types to class names
type_to_class = {"Real": "real", "Fake": "fake"}
categories = ["Train", "Test", "Validation"]

for img_type, cls in type_to_class.items():
    all_images = []
    
    # Collect all images from all categories (Train, Test, Validation)
    for category in categories:
        img_dir = os.path.join(source_dir, category, img_type)
        if os.path.exists(img_dir):
            images = os.listdir(img_dir)
            all_images.extend([(category, img) for img in images])
    
    print(f"Found {len(all_images)} {cls} images")
    random.shuffle(all_images)

    total = len(all_images)
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)

    train_imgs = all_images[:train_end]
    val_imgs = all_images[train_end:val_end]
    test_imgs = all_images[val_end:]

    for split, split_imgs in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):
        dest_folder = os.path.join(output_dir, split, cls)
        os.makedirs(dest_folder, exist_ok=True)

        for category, img in split_imgs:
            src = os.path.join(source_dir, category, img_type, img)
            dst = os.path.join(dest_folder, img)
            shutil.copy(src, dst)
        
        print(f"  {split}: {len(split_imgs)} images")

print("Dataset split completed!")