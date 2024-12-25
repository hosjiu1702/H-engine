import cv2
import numpy as np

# Load the cloth image and mask image
cloth_image = cv2.imread("validation_cloth_image_0.jpg")
mask_image = cv2.imread("custom_mask.png", cv2.IMREAD_GRAYSCALE)

# Ensure the mask is binary (0 for background, 255 for the cloth)
_, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)

# Create a black background
black_background = np.zeros_like(cloth_image)

# Extract the cloth region from the cloth image
cloth_only = cv2.bitwise_and(cloth_image, cloth_image, mask=binary_mask)

# Combine the cloth region with the black background
result = cv2.add(cloth_only, black_background)

# Save or display the result
cv2.imwrite("result_image.jpg", result)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()