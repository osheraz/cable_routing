from cable_routing.env.dummy_env import DummyExperimentEnv
import os
import cv2
print(os.getcwd())

img_path = 'data/zed_images/img_1.png'
img = cv2.imread(img_path)
DummyExperimentEnv().trace_cable_from_clips(img)