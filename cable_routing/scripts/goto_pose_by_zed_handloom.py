from math import pi
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
from cable_routing.env.ext_camera.utils.img_utils import select_target_point, crop_img

from cable_routing.env.ext_camera.utils.img_utils import (
    SCALE_FACTOR,
    define_board_region,
    select_target_point,
)

from cable_routing.handloom.handloom_pipeline.single_tracer import CableTracer


def get_world_coord_from_pixel_coord(
    pixel_coord, cam_intrinsics, cam_extrinsics, image_shape=None, table_depth=0.73
):
    pixel_coord = np.array(pixel_coord, dtype=np.float32)

    if image_shape and (
        cam_intrinsics.width != image_shape[1]
        or cam_intrinsics.height != image_shape[0]
    ):
        scale_x = cam_intrinsics.width / image_shape[1]
        scale_y = cam_intrinsics.height / image_shape[0]
        pixel_coord[0] *= scale_x
        pixel_coord[1] *= scale_y

    pixel_homogeneous = np.array([pixel_coord[0], pixel_coord[1], 1.0])
    point_3d_cam = np.linalg.inv(cam_intrinsics._K).dot(pixel_homogeneous) * table_depth

    point_3d_world = (
        cam_extrinsics.rotation.dot(point_3d_cam) + cam_extrinsics.translation
    )

    return point_3d_world


def pick_target_on_path(img, path):
    img_display = img.copy()

    for x, y in path:
        cv2.circle(img_display, (x, y), 2, (0, 255, 0), -1)

    selected_point = select_target_point(img_display)
    if selected_point is not None:
        cv2.destroyAllWindows()
        return selected_point


def main(args: ExperimentConfig):

    rospy.init_node("handloom_integration")

    yumi = YuMiRobotEnv(args.robot_cfg)
    yumi.close_grippers()

    tracer = CableTracer()

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

    img, (crop_x, crop_y) = crop_img(frame)

    print("Select a target point.")
    pixel_coord = select_target_point(img)

    path, status = tracer.trace(img=img, endpoints=pixel_coord)
    print(status)

    original_path = [(x + crop_x, y + crop_y) for x, y in path]

    cv2.destroyAllWindows()

    move_to_pixel = pick_target_on_path(frame, original_path)

    world_coord = get_world_coord_from_pixel_coord(
        move_to_pixel, CAM_INTR, T_CAM_BASE  # , board_rect
    )

    print("World Coordinate: ", world_coord)

    yumi.single_hand_grasp(world_coord, slow_mode=True)

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
