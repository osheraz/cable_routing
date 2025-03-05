# import sys
# import os
# Add the parent directory of 'cable_routing' to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch # This saves everything! For some reason...
import numpy as np
from cable_routing.env.dummy_env import DummyExperimentEnv
from scipy.interpolate import UnivariateSpline, make_splrep
import cv2

img_path = 'data/zed_images/img_1.png'
img = cv2.imread(img_path)
cv2.imshow('img', img)

coords = []

# def clickEvent(event, x, y, flags, param):
#     if event==cv2.EVENT_LBUTTONUP:
#         print (f'Clicked at {x, y}')
#         param.append([x, y])
#         print (param)
#         cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)

# cv2.setMouseCallback('img', clickEvent, coords)
# cv2.waitKey(0)
# print (coords)
# x = [coord[0] for coord in coords]
# print ('x')
# print (x)
# y = [coord[1] for coord in coords]
# print('y')
# print(y)
# spl = make_splrep(x, y, s=0)
# x_smooth = np.linspace(min(x), max(x)+100, 100)
# y_smooth = spl(x_smooth)

# for x0, y0, x1, y1 in zip(x_smooth, y_smooth, x_smooth[1:], y_smooth[1:]):
#     print (x0, y0, x1, y1)
#     cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), color=(0, 255, 0), thickness=3)

# for x, y, in zip(x_smooth, y_smooth):
#     cv2.circle(img, (int(x), int(y)), radius=5, thickness=-1, color=(0, 255, 0))

# cv2.imshow('img with path', img)
# cv2.waitKey(0)

env = DummyExperimentEnv()
print(env.board.clip_positions)
env.trace_cable(img=img)

# from cable_routing.handloom.model_training.src.model import KeypointsGauss
# from cable_routing.handloom.model_training.config import *
# import torch
# from cable_routing.handloom.handloom_pipeline.tracer import (
#     AnalyticTracer,
#     TraceEnd,
#     Tracer,
# )
# import torch

# analytic_tracer = Tracer()

# trace_config = TRCR32_CL3_12_UNet34_B64_OS_MedleyFix_MoreReal_Sharp()
# trace_model = KeypointsGauss(
#     1,
#     img_height=trace_config.img_height,
#     img_width=trace_config.img_width,
#     channels=3,
#     resnet_type=trace_config.resnet_type,
#     pretrained=trace_config.pretrained,
# )
# trace_model.load_state_dict(
#     torch.load("cable_routing/handloom/models/tracer/tracer_model.pth", map_location=torch.device('cpu')), 
# )

# DummyExperimentEnv().trace_cable(img)