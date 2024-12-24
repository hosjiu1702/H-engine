import os

# Define the path to the base folder
base_dir = "Dresscode_toy"  # Replace with the actual path

# Subfolders in the base directory
subfolders = ["agnostic_mask", "cloth", "image"]

# Loop through the 'cloth' subfolder to rename files from '_1.jpg' to '_0.jpg'
cloth_folder_path = os.path.join(base_dir, "cloth")
if os.path.exists(cloth_folder_path):
    for file_name in os.listdir(cloth_folder_path):
        if file_name.endswith("_1.jpg"):
            new_file_name = file_name.replace("_1.jpg", "_0.jpg")
            old_file_path = os.path.join(cloth_folder_path, file_name)
            new_file_path = os.path.join(cloth_folder_path, new_file_name)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {file_name} -> {new_file_name}")

# Create the 'prompt' folder if it doesn't exist
prompt_folder_path = os.path.join(base_dir, "prompt")
os.makedirs(prompt_folder_path, exist_ok=True)

# Create empty text files with the same names as files in the subfolders
for subfolder in subfolders:
    subfolder_path = os.path.join(base_dir, subfolder)
    
    # Loop through each file in the subfolder
    for file_name in os.listdir(subfolder_path):
        if file_name.endswith(".jpg"):
            # Create an empty text file with the same name as the image file
            text_file_path = os.path.join(prompt_folder_path, f"{os.path.splitext(file_name)[0]}.txt")
            with open(text_file_path, "w") as f:
                pass  # Create an empty text file

            print(f"Created empty file: {text_file_path}")

print("Task completed successfully!")
