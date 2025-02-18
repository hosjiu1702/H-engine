from transformers import pipeline
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw
from ultralytics import YOLO
#import mediapipe as mp

segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes", device = 'cuda')


model = YOLO('yolov11m-face.pt')

# Function to get faces using Haar Cascades
def get_faces(img_arr):
    # Run YOLO inference
    results = model.predict(source=img_arr, save=False, conf=0.2, verbose=False)

    # Extract bounding boxes
    faces = []
    for r in results[0].boxes:
        x1, y1, x2, y2 = r.xyxy[0]  # Coordinates of the bounding box
        faces.append({'bbox': [int(x1), int(y1), int(x2), int(y2)]})
    return faces



def remove_face(img, mask):
    # Convert image to numpy array
    img_arr = np.asarray(img)
    
    # Run face detection
    faces = get_faces(img_arr)

    
    # Get the first face
    print(faces)
    faces = faces[0]['bbox']

    # Width and height of face
    w = faces[2] - faces[0]
    h = faces[3] - faces[1]

    # Make face locations bigger
    faces[0] = faces[0] - int(w * 0.7)  # x left (expand further for hair)
    faces[2] = faces[2] + int(w * 0.7)  # x right (expand further for hair)
    faces[1] = faces[1] - int(h * 0.7)  # y top (expand further for hair)
    faces[3] = faces[3] + int(h * 0.5) + int(h * 0.3)  # y bottom (extend to neck)


    # Convert to [(x_left, y_top), (x_right, y_bottom)]
    face_locations = [(faces[0], faces[1]), (faces[2], faces[3])]

    # Draw black rect onto mask
    img1 = ImageDraw.Draw(mask)
    img1.rectangle(face_locations, fill=0)

    return mask

def segment_body(original_img, face=True):
    # Make a copy
    img = original_img.copy()
    
    # Segment image
    segments = segmenter(img)

    # Create list of masks
    segment_include = ["Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag","Scarf"]
    mask_list = []
    for s in segments:
        if(s['label'] in segment_include):
            mask_list.append(s['mask'])


    # Paste all masks on top of eachother 
    final_mask = np.array(mask_list[0])
    for mask in mask_list:
        current_mask = np.array(mask)
        final_mask = final_mask + current_mask
            
    # Convert final mask from np array to PIL image
    final_mask = Image.fromarray(final_mask)

    # Remove face
    if(face==False):
        final_mask = remove_face(img.convert('RGB'), final_mask)

    # Apply mask to original image
    img.putalpha(final_mask)

    return img, final_mask


def segment_torso(original_img):
    # Make a copy
    img = original_img.copy()
    
    # Segment image
    segments = segmenter(img)

    # Create list of masks
    segment_include = ["Upper-clothes", "Dress", "Belt", "Face", "Left-arm", "Right-arm"]
    mask_list = []
    for s in segments:
        if(s['label'] in segment_include):
            mask_list.append(s['mask'])


    # Paste all masks on top of eachother 
    final_mask = np.array(mask_list[0])
    for mask in mask_list:
        current_mask = np.array(mask)
        final_mask = final_mask + current_mask
            
    # Convert final mask from np array to PIL image
    final_mask = Image.fromarray(final_mask)

    # Remove face
    final_mask = remove_face(img.convert('RGB'), final_mask)

    # Apply mask to original image
    img.putalpha(final_mask)

    return img, 


