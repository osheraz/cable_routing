import rospy
import numpy as np
import cv2
from cable_routing.env.robots.yumi import YuMiRobotEnv
import tyro
from autolab_core import RigidTransform, CameraIntrinsics
from cable_routing.configs.envconfig import ExperimentConfig
from sensor_msgs.msg import CameraInfo
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber

from cable_routing.env.ext_camera.utils.img_utils import (
    SCALE_FACTOR,
    define_board_region,
    select_target_point,
    get_world_coord_from_pixel_coord,
)


def main(args: ExperimentConfig):
    rospy.init_node("zed_yumi_integration")

    yumi = YuMiRobotEnv(args.robot_cfg)
    yumi.close_grippers()
    # yumi.open_grippers()

    zed_cam = ZedCameraSubscriber()
    rospy.loginfo("Waiting for images from ZED camera...")
    while zed_cam.rgb_image is None or zed_cam.depth_image is None:
        rospy.sleep(0.1)

    camera_info = rospy.wait_for_message("/zedm/zed_node/rgb/camera_info", CameraInfo)

    arm = "left"  # if world_coord[1] < 0 else "left"

    T_CAM_BASE = RigidTransform.load(
        f"/home/osherexp/cable_routing/cable_routing/configs/cameras/zed_to_world_{arm}.tf"
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

    # Uncomment for board selection
    # resized_frame = cv2.resize(
    #     frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA
    # )

    # print("Draw a rectangle for the board area. Press 's' to continue.")
    # board_rect = define_board_region(resized_frame)
    # if board_rect is None:
    #     return

    # board_rect = [
    #     (tuple(np.array(corner) / SCALE_FACTOR) for corner in rect)
    #     for rect in board_rect
    # ]

    print("Select a target point.")
    pixel_coord = select_target_point(frame)
    print("Pixel Coord:", pixel_coord)
    if pixel_coord is None:
        return

    world_coord = get_world_coord_from_pixel_coord(
        pixel_coord, CAM_INTR, T_CAM_BASE, depth_map=zed_cam.get_depth()
    )

    print("World Coordinate: ", world_coord)

    # input("Press Enter to apply...")

    yumi.single_hand_grasp(arm, world_coord, eef_rot=0, slow_mode=True)

    # yumi.dual_hand_grasp(
    #     world_coord=world_coord,
    #     axis="x",
    #     slow_mode=True,
    # )

    # world_coord[2] += 0.1
    # world_coord[0] -= 0.1
    # world_coord[1] += 0.1
    # yumi.rotate_dual_hands_around_center(angle=np.pi / 2)
    # yumi.move_dual_hand_insertion(world_coord)
    # yumi.slide_hand(arm="left", axis="x", amount=0.1)

    # world_coord[2] += 0.1
    # world_coord[0] += 0.1
    # yumi.move_dual_hand_to(world_coord, slow_mode=True)

    input("Press Enter to return...")
    yumi.open_grippers()
    yumi.move_to_home()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
