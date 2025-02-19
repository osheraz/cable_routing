import rospy
from sensor_msgs.msg import CameraInfo
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import cv2
from autolab_core import RigidTransform
from cable_routing.configs.envconfig import ZedMiniConfig
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber
from cable_routing.handloom.handloom_pipeline.tracer import AnalyticTracer, Tracer


# os.environ["QT_QPA_PLATFORM"] = "offscreen"
def convert_to_handloom_input(img):

    img = cv2.resize(img, (772, 1032))
    img = np.average(img, axis=2, keepdims=True).astype(np.uint8)
    img = np.stack([img] * 3, axis=-1).squeeze()

    return img


def main():
    rospy.init_node("zed_cable_tracking")

    zed_cam = ZedCameraSubscriber()
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

    img = convert_to_handloom_input(rgb_frame)

    coordinates = []

    def click_event(event, x, y, flags, param):
        global coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at ({x}, {y})")
            coordinates = [x, y]

    cv2.imshow("zed image", img)
    cv2.setMouseCallback("zed image", click_event)
    cv2.waitKey(0)

    start_pixels = np.array(coordinates)[::-1]
    img_cp = img.copy()

    print("Starting analytical tracer")
    start_pixels, _ = analytic_tracer.trace(
        img, start_pixels, path_len=6, viz=False, idx=100
    )
    if len(start_pixels) < 5:
        print("failed analytical trace")
        exit()
    print("Starting learned tracer")

    # output: path, TraceEnd.FINISHED, heatmaps, crops, covariances, max_sums

    path, status, heatmaps, crops, covs, sums = tracer.trace(
        img_cp, start_pixels, path_len=1200, viz=True, idx=100
    )


if __name__ == "__main__":
    main()
