import cv2
import numpy as np

# Load the image
image_path = "023664_0.jpg"  # Replace with your image path
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image at {image_path} not found.")
    
# Copy for displaying and mask initialization
drawing_image = image.copy()
mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Grayscale mask

# Mouse callback function for drawing
drawing = False
last_point = None
pen_size = 50  # Initial pen size

def draw_mask(event, x, y, flags, param):
    global drawing, last_point, pen_size
    
    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
        drawing = True
        last_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:  # Draw on movement
        if last_point:
            cv2.line(drawing_image, last_point, (x, y), (0, 255, 0), thickness=pen_size)  # Green line for visualization
            cv2.line(mask, last_point, (x, y), 255, thickness=pen_size)  # White line on mask
            last_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:  # Stop drawing
        drawing = False
        last_point = None

# Create a window and set the mouse callback
cv2.namedWindow("Draw Mask")
cv2.setMouseCallback("Draw Mask", draw_mask)

while True:
    cv2.imshow("Draw Mask", drawing_image)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # Press 'ESC' to exit
        break
    elif key == ord('r'):  # Press 'r' to reset
        drawing_image = image.copy()
        mask.fill(0)
    elif key == ord('+') or key == ord('='):  # Press '+' to increase pen size
        pen_size += 1
        print(f"Pen size increased to: {pen_size}")
    elif key == ord('-'):  # Press '-' to decrease pen size
        pen_size = max(1, pen_size - 1)  # Minimum pen size is 1
        print(f"Pen size decreased to: {pen_size}")

# Save the mask if needed
cv2.imwrite("custom_mask.png", mask)
cv2.destroyAllWindows()
