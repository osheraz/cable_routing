import rospy
import numpy as np
import tyro
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.env.env import ExperimentEnv
from cable_routing.env.board.cable import Cable


def main(args: ExperimentConfig):

    rospy.init_node("integration")

    env = ExperimentEnv(args)

    board = ExperimentEnv.board

    goal_sequence = [(701, 84), (829, 177), (974, 256), (890, 578), (1317, 559)]

    goal_cable = Cable(
        coordinates=goal_sequence,
        env_keypoints=board.key_locations,
        grid_size=board.grid_size,
        id=-1,
    )
    board.add_cable(goal_cable)

    path_in_pixels, path_in_world, cable_orientations = env.update_cable_path()

    cur_cable = Cable(
        coordinates=path_in_pixels,
        env_keypoints=board.key_locations,
        grid_size=board.grid_size,
        id=0,
    )
    board.add_cable(cur_cable)

    grasp_in_pixels, grasp_in_world, idx = env.grasp_cable_node(
        path_in_pixels, cable_orientations
    )
    env.robot.move_to_home()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
