from string import ascii_uppercase
import cv2
import json
import math
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
#from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber
#from cable_routing.configs.envconfig import ExperimentConfig
#zed_cam = ZedCameraSubscriber()
#^Should work but no rospy on mac :(
image = None
#image = zed_cam.rgb_image
if image is None:
    print("No frame received!")
    image = cv2.imread("cable_routing/configs/board/board.png")
    if image is None:
        raise FileNotFoundError("Image not found!")

# Convert BGR to HSV (why is it BGR?)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define yellow color range in HSV
#might need tweaking to be robust to dif lighting
lower_yellow = np.array([20, 60,240])
upper_yellow = np.array([35, 255, 255])

# Create a binary mask where yellow is white and the rest is black
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

data = {}
letters = ascii_uppercase
letter_num = 0
for i, clip in enumerate(contours):
    area = cv2.contourArea(contours[i])
    if area > 300:  # Ignore small contours
     perimeter = cv2.arcLength(contours[i], True)
     circularity = 4 * math.pi * area / (perimeter * perimeter)
     if circularity > 0.4:
        x, y, w, h = cv2.boundingRect(contours[i])
        center = ((int)(x+w/2),(int)(y+h/2))
        color = (0,255,0)
        cv2.circle(image, center, 5, color, -1)
        #add_to_output()
        letter = letters[letter_num] if letter_num < len(letters) else f"Clip_{letter_num}"
        letter_num = letter_num+1
        data[letter] = { 
            "x": (int)(x+w/2),
            "y": (int)(y+h/2),
            "type": 1,
            "orientation": 0}
  
with open('cable_routing/configs/board/output.json', 'w') as f:
    json.dump(data, f, indent=4)

# Show results
cv2.imshow("Mask", mask)
cv2.imshow("Yellow Pegs", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

