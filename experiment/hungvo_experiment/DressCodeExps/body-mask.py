from SegBody import segment_body
import os
from PIL import Image

# Define the paths to the folders
image_folder = "Dresscode_Dresses/image"
mask_folder = "Dresscode_Dresses/agnostic-mask"

# Ensure the output folder exists
os.makedirs(mask_folder, exist_ok=True)

# Function to process all images
def process_images(image_folder, mask_folder):
    # List all image files in the image folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        # Construct full path to the image file
        image_path = os.path.join(image_folder, image_file)

        # Open the image
        image = Image.open(image_path)

        # Perform segmentation (replace segment_body with your actual function)
        seg_image, mask_image = segment_body(image, face=False)

        # Save the mask image to the output folder with the same name
        mask_path = os.path.join(mask_folder, image_file)
        mask_image.save(mask_path)

        print(f"Processed and saved mask for: {image_file}")

# Call the function
process_images(image_folder, mask_folder)