import sys
import math
import rospy
import numpy as np
import cv2
import time
from autolab_core import RigidTransform, Point
from tqdm import tqdm
from cable_routing.env.robots.yumi import YuMiRobotEnv
from cable_routing.configs.envconfig import ExperimentConfig
import tyro
from cable_routing.env.ext_camera.ros.image_utils import image_msg_to_numpy
from sensor_msgs.msg import CameraInfo, Image

# Load the transformation from the world frame to the ZED camera frame
zed_to_world_path = "/home/osheraz/cable_routing/data/zed/zed2world.tf"
world_to_zed = RigidTransform.load(zed_to_world_path).inverse()


class ZedCameraSubscriber:

    def __init__(
        self,
        topic_depth="/zedm/zed_node/depth/depth_registered",
        topic_rgb="/zedm/zed_node/rgb/image_rect_color",
    ):
        self.depth_image = None
        self.rgb_image = None

        self.depth_subscriber = rospy.Subscriber(
            topic_depth, Image, self.depth_callback, queue_size=2
        )
        self.rgb_subscriber = rospy.Subscriber(
            topic_rgb, Image, self.rgb_callback, queue_size=2
        )

    def depth_callback(self, msg):
        try:
            self.depth_image = image_msg_to_numpy(msg)
        except Exception as e:
            rospy.logerr(f"Depth callback error: {e}")

    def rgb_callback(self, msg):
        try:
            self.rgb_image = image_msg_to_numpy(msg)
        except Exception as e:
            rospy.logerr(f"RGB callback error: {e}")


def click_event(event, u, v, flags, param):
    """Handles mouse click events to get pixel coordinates."""
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = param["img"][v, u]
        print(f"Pixel coordinates: (u={u}, v={v}) - Pixel value: {pixel_value}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            param["img"],
            f"({u},{v})",
            (u, v),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            param["img"],
            str(pixel_value),
            (u, v + 20),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        time.sleep(0.5)
        param["u"] = u
        param["v"] = v


def main(args: ExperimentConfig):
    """Main function to run the robot-camera interaction loop."""
    rospy.init_node("zed_yumi_integration")

    # Initialize YuMi robot
    yumi = YuMiRobotEnv(args.robot_cfg)
    yumi.close_grippers()
    yumi.move_to_home()

    # Initialize ZED camera subscriber
    zed_cam = ZedCameraSubscriber()

    # Wait for the first set of images to be received
    rospy.loginfo("Waiting for images from ZED camera...")
    while zed_cam.rgb_image is None or zed_cam.depth_image is None:
        rospy.sleep(0.1)

    camera_info = rospy.wait_for_message("/zedm/zed_node/depth/camera_info", CameraInfo)
    cam_width = camera_info.width
    cam_height = camera_info.height
    f_x = camera_info.K[0]  # fx
    f_y = camera_info.K[4]  # fy
    c_x = camera_info.K[2]  # cx
    c_y = camera_info.K[5]  # cy

    for _ in tqdm(range(3)):
        # Get the current end-effector pose
        left_pose, _ = yumi.get_ee_pose()
        previous_pose = left_pose  # Store for later use

        # Get RGB and depth images
        left_img = zed_cam.rgb_image
        depth = zed_cam.depth_image

        # Display image and wait for user selection
        params = {"img": left_img.copy(), "u": None, "v": None}
        cv2.imshow("Image", left_img)
        cv2.setMouseCallback("Image", click_event, param=params)

        while params["u"] is None or params["v"] is None:
            if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
                break

        # Get user-selected pixel coordinates
        u, v = params["u"], params["v"]
        Z = depth[v, u]
        X = ((u - c_x) * Z) / f_x
        Y = ((v - c_y) * Z) / f_y
        print(f"Projected point: X={X}, Y={Y}, Z={Z}")

        # Transform point from camera to robot frame
        point = Point(np.array([X, Y, Z]), frame="world")
        point_in_robot = world_to_zed * point
        print("Point in robot frame:", point_in_robot.data)

        # Move the robot end-effector to the target point
        # target_pose = RigidTransform(rotation=RigidTransform.x_axis_rotation(math.pi), translation=point_in_robot.data)
        # target_pose.translation[2] += 0.2  # Move above the point
        # yumi.set_ee_pose(left_pose=target_pose)

        # target_pose.translation[2] -= 0.2  # Move down to the point
        # yumi.set_ee_pose(left_pose=target_pose)

        # input("Press Enter to continue...")
        # yumi.set_ee_pose(left_pose=previous_pose)  # Move back to original pose

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
