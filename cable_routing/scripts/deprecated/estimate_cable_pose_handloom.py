from tracemalloc import start
from flask.cli import F
import rospy
from sympy import false
from sensor_msgs.msg import CameraInfo
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import cv2
from autolab_core import RigidTransform
from cable_routing.configs.envconfig import ZedMiniConfig
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber
from cable_routing.handloom.handloom_pipeline.tracer import AnalyticTracer, Tracer
from cable_routing.env.ext_camera.utils.img_utils import (
    define_board_region,
    select_target_point,
)


# os.environ["QT_QPA_PLATFORM"] = "offscreen"
# def convert_to_handloom_input(img, invert=False):

#     img = cv2.cvtColor(cv2.resize(img, (772, 1032)), cv2.COLOR_RGB2GRAY)
#     # print(img)
#     # print(img.shape)
#     if invert:
#         img = cv2.bitwise_not(img)
#     # img = np.average(img, axis=2, keepdims=True).astype(np.uint8)
#     img = np.stack([img] * 3, axis=-1).squeeze()

#     return img


def convert_to_handloom_input(img, invert=False, pad=True):
    target_size = img.shape[:2]  # (772, 1032)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if invert:
        img = cv2.bitwise_not(img)

    h, w = img.shape
    aspect_ratio = w / h

    if pad:
        extra_pad = int(min(h, w) * 0.05)
        new_h = h + 2 * extra_pad
        new_w = int(new_h * aspect_ratio)
        padded_img = 0 * np.ones((new_h, new_w), dtype=np.uint8)
        y_offset = (new_h - h) // 2
        x_offset = (new_w - w) // 2
        padded_img[y_offset : y_offset + h, x_offset : x_offset + w] = img
        img = padded_img

    img = cv2.resize(img, target_size)

    img = np.stack([img] * 3, axis=-1).squeeze()

    return img


def crop_img(img):
    point1, point2 = define_board_region(img)[0]
    x1, x2, y1, y2 = (
        min(point1[0], point2[0]),
        max(point1[0], point2[0]),
        min(point1[1], point2[1]),
        max(point1[1], point2[1]),
    )

    return img[y1:y2, x1:x2]


def main():
    print("Starting")
    rospy.init_node("zed_cable_tracking")
    print("Initialized Camera")

    zed_cam = ZedCameraSubscriber()
    print("Set up Zed Subscriber")
    tracer = Tracer()
    analytic_tracer = AnalyticTracer()

    rospy.loginfo("Waiting for ZED camera stream...")
    while zed_cam.rgb_image is None or zed_cam.depth_image is None:
        rospy.sleep(0.1)

    camera_info = rospy.wait_for_message("/zedm/zed_node/depth/camera_info", CameraInfo)
    width = camera_info.width
    height = camera_info.height
    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    zed_config = ZedMiniConfig()
    zed_intrinsic = zed_config.get_intrinsic_matrix()

    print("ZED Intrinsic:", zed_intrinsic)
    print("O3D Intrinsic:", intrinsic.intrinsic_matrix)

    T_CAM_BASE = RigidTransform.load(
        "/home/osheraz/cable_routing/data/zed/zed_to_world.tf"
    ).as_frames(from_frame="zed", to_frame="base_link")

    ext_mat = T_CAM_BASE.matrix

    rospy.loginfo("Processing single frame...")

    rgb_frame = zed_cam.rgb_image
    cv2.imwrite("/home/osheraz/cable_routing/records/test.jpg", rgb_frame)

    depth_frame = zed_cam.depth_image

    if rgb_frame is None or depth_frame is None:
        rospy.logerr("No frame received!")
        return

    # cropping the image
    img = crop_img(rgb_frame)
    img = convert_to_handloom_input(img, invert=True)

    coordinates = select_target_point(img)
    start_pixels = np.array(coordinates)[::-1]
    img_cp = img.copy()

    print("Starting analytical tracer")

    start_pixels, _ = analytic_tracer.trace(
        img, start_pixels, path_len=6, viz=False, idx=100
    )
    print(start_pixels)
    if len(start_pixels) < 5:
        print("failed analytical trace")
        exit()

    print("Starting learned tracer")

    # output: path, TraceEnd.FINISHED, heatmaps, crops, covariances, max_sums

    path, status, heatmaps, crops, covs, sums = tracer.trace(
        img_cp, start_pixels, path_len=1200, viz=True, idx=100
    )

    print(status)


if __name__ == "__main__":
    main()
