import os
import shutil

# Base directory where the folders "dresses", "lower_body", "upper_body" exist
base_dir = "Dataset/dresses/organized_images/image"

new_base_dir = "Dataset/dresses/organized_images/image/agnostic_mask"

# Folders to process
big_folders = ["dresses", "lower_body", "upper_body"]

for folder in big_folders:
    images_path = os.path.join(base_dir, folder, "images")
    
    # Create "organized_images/image" and "organized_images/cloth" directories
    organized_path = os.path.join(base_dir, folder, "organized_images")
    image_dir = os.path.join(organized_path, "image")
    cloth_dir = os.path.join(organized_path, "cloth")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(cloth_dir, exist_ok=True)

    # Loop through files in the images folder
    for file_name in os.listdir(images_path):
        source_file = os.path.join(images_path, file_name)
        if file_name.endswith("_0.jpg"):
            shutil.move(source_file, os.path.join(image_dir, file_name))
        elif file_name.endswith("_1.jpg"):
            shutil.move(source_file, os.path.join(cloth_dir, file_name))

print("Images have been organized into new folders successfully!")
