import numpy as np
import cv2
from cable_routing.handloom.handloom_pipeline.tracer import (
    AnalyticTracer,
    TraceEnd,
    Tracer,
)


class CableTracer:
    def __init__(self):
        self.tracer = Tracer()
        self.analytic_tracer = AnalyticTracer()

    def convert_to_handloom_input(self, img, invert=True):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if invert:
            img = cv2.bitwise_not(img)
        return np.stack([img] * 3, axis=-1).squeeze()

    def trace(
        self,
        img,
        start_points,
        end_points=None,
        clips=None,
        save_folder="./trace_test",
        last_path=None,
        idx=1,
        viz=False,
    ):

        img = self.convert_to_handloom_input(img, invert=False)

        start_pixels = np.array(start_points)[::-1]  # TODO: To y-x

        if end_points is not None:
            end_points = [np.array(end_points)[::-1]]

        img_cp = img.copy()

        if last_path is None:
            start_pixels, _ = self.analytic_tracer.trace(
                img, start_pixels, endpoints=end_points, path_len=3, viz=viz, idx=100
            )
        else:
            start_pixels = np.flip(last_path[-4:], axis=1)

        if len(start_pixels) < 3:
            print("Failed analytical trace")
            return None

        path, status, _, _, _, _ = self.tracer.trace(
            img_cp,
            start_pixels,
            endpoints=end_points,
            path_len=200,
            clips=clips,
            viz=viz,
            idx=idx,
            save_folder=save_folder,
        )

        path = np.flip(path, axis=1)

        if viz:
            cv2.destroyAllWindows()

        return path, status
