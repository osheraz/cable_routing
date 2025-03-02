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

    sequence = [(701, 84), (829, 177), (974, 256), (890, 578), (1317, 559)]

    goal_cable = Cable(
        coordinates=sequence,
        env_keypoints=board.key_locations,
        grid_size=board.grid_size,
        id=-1,
    )

    path = env.update_cable_path()

    cur_cable = Cable(
        coordinates=path,
        env_keypoints=board.key_locations,
        grid_size=board.grid_size,
        id=-1,
    )
    board.add_cable(cur_cable)

    env.goto_cable_node(path)

    env.robot.move_to_home()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
