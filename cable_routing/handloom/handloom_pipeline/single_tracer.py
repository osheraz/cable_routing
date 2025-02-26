import numpy as np
import cv2
from cable_routing.handloom.handloom_pipeline.tracer import AnalyticTracer, Tracer


class CableTracer:
    def __init__(self):
        self.tracer = Tracer()
        self.analytic_tracer = AnalyticTracer()

    def convert_to_handloom_input(self, img, invert=True):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if invert:
            img = cv2.bitwise_not(img)
        return np.stack([img] * 3, axis=-1).squeeze()

    def trace(self, img, start_points, end_points=None):
        img = self.convert_to_handloom_input(img, invert=False)

        start_pixels = np.array(start_points)[::-1]  # Convert to (y, x) format
        img_cp = img.copy()

        start_pixels, _ = self.analytic_tracer.trace(
            img, start_pixels, path_len=3, viz=False, idx=100
        )

        if len(start_pixels) < 3:
            print("failed analytical trace")
            return None  # Failed analytical trace

        path, status, _, _, _, _ = self.tracer.trace(
            img_cp, start_pixels, path_len=1200, viz=True, idx=100
        )

        path = np.flip(path, axis=1)

        return path, status
