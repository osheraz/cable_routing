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

    rospy.init_node("pick_nic")
    env = ExperimentEnv(args)

    path_in_pixels, path_in_world, cable_orientations = env.update_cable_path()
    grasp_in_pixels, grasp_in_world, idx = env.grasp_cable_node(
        path_in_pixels, cable_orientations
    )
    env.goto_clip_node()

    env.robot.move_to_home()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
