import rospy
from sensor_msgs.msg import CameraInfo
import open3d as o3d
import numpy as np
import cv2
from autolab_core import RigidTransform
from cable_routing.configs.envconfig import ZedMiniConfig
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber
from cable_routing.handloom.handloom_pipeline.single_tracer import CableTracer
from cable_routing.env.ext_camera.utils.img_utils import select_target_point, crop_img
from cable_routing.env.ext_camera.utils.pcl_utils import (
    create_pcd_from_rgbd,
    path_to_3d,
    generate_color_gradient,
)


def main():
    print("Starting")
    rospy.init_node("zed_cable_tracking")

    zed_cam = ZedCameraSubscriber()
    tracer = CableTracer()

    print("Waiting for ZED camera stream...")
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
    zed_intrinsic = ZedMiniConfig().get_intrinsic_matrix()

    print("ZED Intrinsic:", zed_intrinsic)
    print("O3D Intrinsic:", intrinsic.intrinsic_matrix)

    T_CAM_BASE = (
        RigidTransform.load("/home/osherexp/cable_routing/data/zed/zed_to_world.tf")
        .as_frames(from_frame="zed", to_frame="base_link")
        .matrix
    )

    rgb_frame = zed_cam.rgb_image
    depth_frame = zed_cam.depth_image

    if rgb_frame is None or depth_frame is None:
        print("No frame received!")
        return

    print("Processing single frame...")
    img, (crop_x, crop_y) = crop_img(rgb_frame)
    coordinates = select_target_point(img)

    path, status = tracer.trace(img=img, endpoints=coordinates)
    print(status)

    original_path = [
        (
            x + crop_x,
            y + crop_y,
        )
        for x, y in path
    ]

    cable_3d_points = path_to_3d(original_path, depth_frame, zed_intrinsic, T_CAM_BASE)

    cable_pcd = o3d.geometry.PointCloud()
    cable_pcd.points = o3d.utility.Vector3dVector(cable_3d_points)
    cable_pcd.colors = o3d.utility.Vector3dVector(
        generate_color_gradient(len(cable_3d_points))
    )

    pcd = create_pcd_from_rgbd(rgb_frame, depth_frame, zed_intrinsic, T_CAM_BASE)

    o3d.visualization.draw_geometries([pcd, cable_pcd])


if __name__ == "__main__":
    main()
