import rospy
import numpy as np
import cv2
from autolab_core import RigidTransform, Point
from tqdm import tqdm
from cable_routing.env.robots.yumi import YuMiRobotEnv
import tyro
from autolab_core import RigidTransform, Point, CameraIntrinsics
from cable_routing.configs.envconfig import ExperimentConfig

from cable_routing.env.ext_camera.ros.utils.image_utils import image_msg_to_numpy
from sensor_msgs.msg import CameraInfo, Image
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber

from cable_routing.env.ext_camera.utils.img_utils import (
    SCALE_FACTOR,
    define_board_region,
    select_target_point,
)

TABLE_HEIGHT = 0.05  # 0.023
BOARD_HEIGHT = 0.05  # 0.035


def get_world_coord_from_pixel_coord(
    pixel_coord, cam_intrinsics, cam_extrinsics, board_rect
):
    """
    Convert pixel coordinates to world coordinates using camera intrinsics and extrinsics.
    Adjust height if inside the board region.
    """
    pixel_coord = np.array(pixel_coord)
    point_3d_cam = np.linalg.inv(cam_intrinsics._K).dot(
        np.r_[pixel_coord, 0.81 - TABLE_HEIGHT]
    )
    point_3d_world = cam_extrinsics.matrix.dot(np.r_[point_3d_cam, 1.0])
    point_3d_world = point_3d_world[:3] / point_3d_world[3]

    if board_rect:
        (x_min, y_min), (x_max, y_max) = board_rect[0]
        if x_min < pixel_coord[0] < x_max and y_min < pixel_coord[1] < y_max:
            point_3d_world[-1] = BOARD_HEIGHT

    return point_3d_world


def main(args: ExperimentConfig):
    """Main function to run the robot-camera interaction loop."""
    rospy.init_node("zed_yumi_integration")

    yumi = YuMiRobotEnv(args.robot_cfg)
    yumi.close_grippers()
    # yumi.open_grippers()

    zed_cam = ZedCameraSubscriber()

    rospy.loginfo("Waiting for images from ZED camera...")
    while zed_cam.rgb_image is None or zed_cam.depth_image is None:
        rospy.sleep(0.1)

    camera_info = rospy.wait_for_message("/zedm/zed_node/rgb/camera_info", CameraInfo)

    T_CAM_BASE = RigidTransform.load(
        "/home/osheraz/cable_routing/data/zed/zed_to_world.tf"
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
    resized_frame = cv2.resize(
        frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA
    )

    print("Draw a rectangle for the board area. Press 's' to continue.")
    board_rect = define_board_region(resized_frame)
    if board_rect is None:
        return

    board_rect = [
        (tuple(np.array(corner) / SCALE_FACTOR) for corner in rect)
        for rect in board_rect
    ]

    print("Select a target point.")
    pixel_coord = select_target_point(resized_frame)
    if pixel_coord is None:
        return

    world_coord = get_world_coord_from_pixel_coord(
        pixel_coord, CAM_INTR, T_CAM_BASE, board_rect
    )

    print("World Coordinate: ", world_coord)

    yumi.single_hand_grasp(world_coord, slow_mode=True)

    # yumi.dual_hand_grasp(
    #     world_coord=world_coord,
    #     axis="y",
    #     slow_mode=True,
    # )
    # world_coord[2] += 0.1
    # world_coord[0] += 0.1
    # yumi.move_dual_hand_insertion(world_coord)
    # yumi.slide_hand(arm="left", axis="y", amount=0.1)

    # world_coord[2] += 0.1
    # world_coord[0] += 0.1
    # yumi.move_dual_hand_to(world_coord, slow_mode=True)

    input("Press Enter to return...")
    yumi.move_to_home()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
