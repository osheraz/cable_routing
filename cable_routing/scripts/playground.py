import numpy as np
from cable_routing.env.dummy_env import DummyExperimentEnv
import os
import cv2
import torch
import time
from scipy.interpolate import make_splrep

def get_points(img_path='data/zed_images/img_1.png'):
    
    img = cv2.imread(img_path)
    cv2.imshow('img', img)
    
    coords = []
    
    def clickEvent(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            print(f'Clicked at {x, y}')
            param.append([x, y])
            print(param)
            cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
    
    cv2.setMouseCallback('img', clickEvent, coords)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if not coords:
        print("No points clicked. Exiting.")
        return [], []
    
    x = [coord[0] for coord in coords]
    y = [coord[1] for coord in coords]
    
    return x, y

def fit_spline(x, y, img_path='data/zed_images/img_1.png'):
    if not x or not y:
        print("No valid points to fit.")
        return
    
    img = cv2.imread(img_path)
    
    spl = make_splrep(x, y, s=0)
    x_smooth = np.linspace(min(x), max(x) + 100, 100)
    y_smooth = spl(x_smooth)
    
    for x0, y0, x1, y1 in zip(x_smooth, y_smooth, x_smooth[1:], y_smooth[1:]):
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), color=(0, 255, 0), thickness=3)
    
    for x, y in zip(x_smooth, y_smooth):
        cv2.circle(img, (int(x), int(y)), radius=5, thickness=-1, color=(0, 255, 0))
    
    cv2.imshow('img with path', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    DummyExperimentEnv()
    x, y = get_points()
    fit_spline(x, y)