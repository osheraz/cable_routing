import cv2
import math
import numpy as np

# Load the image
image = cv2.imread("cable_routing/configs/board/board.png")
if image is None:
    raise FileNotFoundError("Image not found!")

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define yellow color range in HSV
lower_yellow = np.array([20, 60,240])
upper_yellow = np.array([35, 255, 255])

# Create a binary mask where yellow is white and the rest is black
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and bounding boxes
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 100:  # Ignore small contours
        continue
    
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    circularity = 4 * math.pi * area / (perimeter * perimeter)
    
    # Example condition: yellow + roughly circular + size filter
    if circularity > 0.4 and area > 300:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)


# Show results
cv2.imshow("Mask", mask)
cv2.imshow("Yellow Pegs", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
