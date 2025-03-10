import cv2
import numpy as np
import os
import time
import tyro
import rospy
from autolab_core import RigidTransform, Point, CameraIntrinsics
from yumi_jacobi.interface import Interface
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber
from sensor_msgs.msg import CameraInfo


def main():
    cam_name = "zed"

    script_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(script_dir, "test_calibration")
    os.makedirs(save_dir, exist_ok=True)

    rospy.init_node("zed_calibration_checker", anonymous=True)

    # Initialize ZED Camera Subscriber
    zed_cam = ZedCameraSubscriber()

    rospy.loginfo("Waiting for images from ZED camera...")
    while zed_cam.rgb_image is None or zed_cam.depth_image is None:
        rospy.sleep(0.1)

    # Get camera intrinsics from ROS topic
    camera_info = rospy.wait_for_message("/zedm/zed_node/rgb/camera_info", CameraInfo)

    T_CAM_BASE = RigidTransform.load(
        "/home/osheraz/cable_routing/cable_routing/configs/cameras/zed_to_world.tf"
    ).as_frames(from_frame="zed", to_frame="base_link")

    CAM_INTR = CameraIntrinsics(
        fx=camera_info.K[0],
        fy=camera_info.K[4],
        cx=camera_info.K[2],
        cy=camera_info.K[5],
        width=camera_info.width,
        height=camera_info.height,
        frame="zed",
    )

    frame = zed_cam.get_rgb()

    interface = Interface(
        speed=0.26,
    )

    interface.home()
    interface.calibrate_grippers()
    interface.close_grippers()

    rot_mat = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    wp1_l = RigidTransform(rotation=rot_mat, translation=[0.4, 0.1, 0.2])
    wp2_l = RigidTransform(rotation=rot_mat, translation=[0.4, 0.0, 0.1])
    wp3_l = RigidTransform(rotation=rot_mat, translation=[0.5, -0.1, 0.1])

    for i in range(9):
        if i < 3:
            wp1_l.translation[1] -= 0.05
            curr_wp = wp1_l
        elif i < 6:
            wp2_l.translation[1] -= 0.05
            curr_wp = wp2_l
        elif i < 9:
            wp3_l.translation[1] -= 0.05
            curr_wp = wp3_l

        interface.go_linear_single(r_target=curr_wp)
        time.sleep(1.0)

        gripper_in_world = interface.get_FK("right").translation
        world_to_camera = T_CAM_BASE.inverse()

        # Convert world coordinates to camera frame
        gripper_in_camera = world_to_camera * Point(gripper_in_world, frame="base_link")

        gripper_point = np.array(
            [[gripper_in_camera.x, gripper_in_camera.y, gripper_in_camera.z]]
        )  # 3D point as a 1x3 array

        gripper_pixel, _ = cv2.projectPoints(
            gripper_point,
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            CAM_INTR.K,
            np.zeros(5),  # Assuming no distortion coefficients
        )

        print(gripper_pixel)

        # Get the latest image from ZED
        frame = zed_cam.get_rgb()

        image_with_point = cv2.circle(
            frame,
            (int(gripper_pixel[0][0][0]), int(gripper_pixel[0][0][1])),
            radius=15,
            color=(0, 0, 255),
            thickness=-1,
        )

        cv2.imwrite(f"{save_dir}/frame_{i:04d}.jpg", image_with_point)

    cv2.destroyAllWindows()
    exit()


if __name__ == "__main__":
    tyro.cli(main)
