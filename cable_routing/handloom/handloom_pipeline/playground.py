import pyzed.sl as sl
import cv2
import os
import numpy as np
from cable_routing.handloom.handloom_pipeline.tracer import AnalyticTracer, Tracer

coordinates = []


def click_event(event, x, y, flags, param):
    global coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at ({x}, {y})")
        coordinates = [x, y]


def convert_to_handloom_input(img):

    img = cv2.resize(img, (772, 1032))
    img = np.average(img, axis=2, keepdims=True).astype(np.uint8)
    img = np.stack([img] * 3, axis=-1).squeeze()

    return img


def main():

    tracer = Tracer()
    analytic_tracer = AnalyticTracer()

    for i in range(9, 10):

        # global zed
        # zed = init_cam()
        img = cv2.imread(
            f"/home/osheraz/handloom/handloom_pipeline/kavish_data/iphone_img{i}.jpg"
        )  # get_image()

        img = convert_to_handloom_input(img)

        cv2.imshow("zed image", img)
        cv2.setMouseCallback("zed image", click_event)
        cv2.waitKey(0)

        # now coordinates are updated for the handloom to use, but need to swap
        start_pixels = np.array(coordinates)[::-1]
        img_cp = img.copy()
        # img[-130:, ...] = 0

        print("Starting analytical tracer")
        start_pixels, _ = analytic_tracer.trace(
            img, start_pixels, path_len=6, viz=False, idx=100 + i
        )
        if len(start_pixels) < 5:
            print("failed analytical trace")
            exit()
        print("Starting learned tracer")

        # output: path, TraceEnd.FINISHED, heatmaps, crops, covariances, max_sums

        path, status, heatmaps, crops, covs, sums = tracer.trace(
            img_cp, start_pixels, path_len=1200, viz=True, idx=100 + i
        )
        # path [N, 2]
        a = 5
        # status TraceEnd\Trace..
        # heatmaps [N, 65, 65]
        # crops [N, 65, 65, 3]
        # covs [N, 1]
        # sums [N, 1]
        # todo?
        cv2.destroyAllWindows()
    exit()


if __name__ == "__main__":
    main()
