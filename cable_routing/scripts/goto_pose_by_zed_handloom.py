from math import pi
import rospy
import numpy as np
import tyro
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.env.env import ExperimentEnv
from cable_routing.env.ext_camera.utils.img_utils import (
    get_world_coord_from_pixel_coord,
    pick_target_on_path,
)


def main(args: ExperimentConfig):

    rospy.init_node("handloom_integration")
    env = ExperimentEnv(args)

    path_in_pixels, path_in_world, cable_orientations = env.update_cable_path()
    grasp_in_pixels, grasp_in_world, idx = env.grasp_cable_node(
        path_in_pixels, cable_orientations, arm="right", display=False
    )
    env.robot.move_to_home()

    # yumi = env.robot
    # yumi.close_grippers()

    # frame = env.zed_cam.get_rgb()

    # path, _ = env.trace_cable(frame)
    # move_to_pixel = pick_target_on_path(frame, path)

    # world_coord = get_world_coord_from_pixel_coord(
    #     move_to_pixel, env.zed_cam.intrinsic, env.T_CAM_BASE
    # )

    # print("World Coordinate: ", world_coord)

    # yumi.single_hand_grasp(world_coord, slow_mode=True)

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

    # input("Press Enter to return...")
    # yumi.open_grippers()
    # yumi.move_to_home()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
