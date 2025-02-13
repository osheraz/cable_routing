import rospy
from sensor_msgs.msg import CameraInfo
import open3d as o3d
import numpy as np
import cv2
from autolab_core import RigidTransform
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber


def main():
    rospy.init_node("zed_pointcloud_visualization")

    zed_cam = ZedCameraSubscriber()

    rospy.loginfo("Waiting for images from ZED camera...")
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
    color_image = o3d.geometry.Image(cv2.cvtColor(zed_cam.rgb_image, cv2.COLOR_BGR2RGB))
    depth_image = o3d.geometry.Image(zed_cam.depth_image.astype(np.float32))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    T_CAM_BASE = RigidTransform.load(
        "/home/osheraz/cable_routing/data/zed/zed_to_world.tf"
    ).as_frames(from_frame="zed", to_frame="base_link")

    ext_mat = T_CAM_BASE.matrix

    pcd.transform(ext_mat)

    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
