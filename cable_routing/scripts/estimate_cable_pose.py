import rospy
from sensor_msgs.msg import CameraInfo
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import cv2
from autolab_core import RigidTransform
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber
from cable_routing.env.ext_camera.utils.img_utils import (
    rescale_intrinsics,
    mask_image_outside_roi,
    green_color_segment,
)


def construct_skeletal_graph(cable_points, num_nodes=1000):

    pca = PCA(n_components=1)
    pca.fit(cable_points)
    projected = pca.transform(cable_points).flatten()
    sorted_indices = np.argsort(projected)
    cable_points_sorted = cable_points[sorted_indices]
    node_indices = np.linspace(0, len(cable_points_sorted) - 1, num_nodes, dtype=int)
    cable_nodes = cable_points_sorted[node_indices]

    return cable_nodes


def main():
    rospy.init_node("zed_cable_tracking")

    zed_cam = ZedCameraSubscriber()

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

    T_CAM_BASE = RigidTransform.load(
        "/home/osheraz/cable_routing/data/zed/zed_to_world.tf"
    ).as_frames(from_frame="zed", to_frame="base_link")

    ext_mat = T_CAM_BASE.matrix

    rospy.loginfo("Processing single frame...")

    rgb_frame = zed_cam.rgb_image
    cv2.imwrite("/home/osheraz/cable_routing/records/test.jpg", rgb_frame)

    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

    depth_frame = zed_cam.depth_image

    if rgb_frame is None or depth_frame is None:
        rospy.logerr("No frame received!")
        return

    roi = cv2.selectROI(
        "Select Region",
        cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR),
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyWindow("Select Region")
    if roi == (0, 0, 0, 0):
        roi = (0, 0, rgb_frame.shape[1], rgb_frame.shape[0])
    rgb_frame = mask_image_outside_roi(rgb_frame, roi)

    color_image = o3d.geometry.Image(rgb_frame)
    depth_image = o3d.geometry.Image(depth_frame.astype(np.float32))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    pcd.transform(ext_mat)

    cable_points = green_color_segment(pcd)
    cable_nodes = construct_skeletal_graph(cable_points)

    cable_pcd = o3d.geometry.PointCloud()
    cable_pcd.points = o3d.utility.Vector3dVector(cable_nodes)
    cable_pcd.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([pcd, cable_pcd])


if __name__ == "__main__":
    main()
