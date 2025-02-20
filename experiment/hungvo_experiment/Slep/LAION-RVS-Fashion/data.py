import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()


DATA_DIR_BODY = os.getenv("DATA_DIR_BODY")
DATA_DIR_GARMENT = os.getenv("DATA_DIR_GARMENT")
BASE_IMAGE_DIR = os.getenv("BASE_IMAGE_DIR")




complex_metadata = pd.read_parquet(DATA_DIR_BODY)
simple_metadata = pd.read_parquet(DATA_DIR_GARMENT)

merged_data = pd.merge(complex_metadata, simple_metadata, on="PRODUCT_ID", how="inner")
merged_data.to_parquet("merged_products_metadata.parquet", index=False)

# Load the complex metadata file
merged_data = pd.read_parquet('merged_products_metadata.parquet')

# Create a base directory to save the images
base_image_dir = BASE_IMAGE_DIR
os.makedirs(base_image_dir, exist_ok=True)

# Function to download and save an image
def download_image(url, save_path, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()  # Raise an error for bad status codes

            # Open the image using PIL and save it
            image = Image.open(BytesIO(response.content))
            image.save(save_path)
            print(f"Downloaded and saved: {save_path}")
            return True  # Return True if download is successful
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to download {url}: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait before retrying
            else:
                print(f"Failed to download after {retries} attempts: {url}")
    return False  # Return False if all attempts fail


# Function to filter and download images based on width and height thresholds
def download_images_with_threshold(df, width_threshold_body, height_threshold_body, width_threshold_garment, height_threshold_garment):
    for index, row in df.iterrows():
        url_body = row['URL_x']
        url_garment = row['URL_y']
        product_id = row['PRODUCT_ID']
        height_body = int(row['HEIGHT_x'])  # Get the HEIGHT_body
        width_body = int(row['WIDTH_x'])    # Get the WIDTH_body
        height_garment = int(row['HEIGHT_y'])  # Get the HEIGHT_garment
        width_garment = int(row['WIDTH_y'])    # Get the WIDTH_garment
        garment_type = row['CATEGORY']     # Get the GARMENT_TYPE 

        # Check if both width and height are greater than the thresholds
        if width_body > width_threshold_body and height_body > height_threshold_body and height_garment > height_threshold_garment and width_garment > width_threshold_garment:
            # Create a subdirectory for the TYPE if it doesn't exist


            image_type = 'human'
            type_dir_human = os.path.join(base_image_dir, garment_type, image_type)
            os.makedirs(type_dir_human, exist_ok=True)

            image_type = 'garment'
            type_dir_garment = os.path.join(base_image_dir, garment_type, image_type)
            os.makedirs(type_dir_garment, exist_ok=True)


            # Construct the filename: PRODUCT_ID_HEIGHT_WIDTH.jpg
            
            image_name = f"{product_id}.jpg"
            image_path_human = os.path.join(type_dir_human, image_name)
            success_body = download_image(url_body, image_path_human)


       
            image_name = f"{product_id}.jpg"
            image_path_garment = os.path.join(type_dir_garment, image_name)
            success_garment = download_image(url_garment, image_path_garment)

            if success_body and success_garment:
                print(f"Successfully downloaded both images for product {product_id}")
            else:
                # Clean up by deleting the incomplete folder
                #if not success_body:
                print(f"Deleting incomplete body image folder for {product_id}")
                if os.path.exists(type_dir_human):
                    for root, dirs, files in os.walk(type_dir_human, topdown=False):
                        for file in files:
                            os.remove(os.path.join(root, file))
                        for dir in dirs:
                            os.rmdir(os.path.join(root, dir))
                    os.rmdir(type_dir_human)

                #if not success_garment:
                print(f"Deleting incomplete garment image folder for {product_id}")
                if os.path.exists(type_dir_garment):
                    for root, dirs, files in os.walk(type_dir_garment, topdown=False):
                        for file in files:
                            os.remove(os.path.join(root, file))
                        for dir in dirs:
                            os.rmdir(os.path.join(root, dir))
                    os.rmdir(type_dir_garment)

                print(f"Failed to download one or both images for product {product_id}")



        else:
            print(f"Skipping {product_id}")


if __name__ == '__main__':

    WIDTH_THRESHOLD_BODY = int(os.getenv("WIDTH_THRESHOLD_BODY"))
    HEIGHT_THRESHOLD_BODY = int(os.getenv("HEIGHT_THRESHOLD_BODY"))
    WIDTH_THRESHOLD_GARMENT = int(os.getenv("WIDTH_THRESHOLD_GARMENT"))
    HEIGHT_THRESHOLD_GARMENT = int(os.getenv("HEIGHT_THRESHOLD_GARMENT"))

    download_images_with_threshold(merged_data, WIDTH_THRESHOLD_BODY, HEIGHT_THRESHOLD_BODY, WIDTH_THRESHOLD_GARMENT, HEIGHT_THRESHOLD_GARMENT)
    