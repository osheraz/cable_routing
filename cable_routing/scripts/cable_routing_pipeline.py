import rospy
import numpy as np
import tyro
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.env.env import ExperimentEnv
from cable_routing.env.board.cable import Cable
from cable_routing.algo.levenshtein import suggest_modifications


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

    path_in_pixels, path_in_world, cable_orientations = env.update_cable_path()

    cur_cable = Cable(
        coordinates=path_in_pixels,
        env_keypoints=board.key_locations,
        grid_size=board.grid_size,
        id=-1,
    )
    board.add_cable(cur_cable)

    goal_config = {cur_cable.id: goal_cable}
    suggestion = suggest_modifications(board, goal_configuration=goal_config)

    print(suggestion[0])
    counts = {"Add": 0, "Swap": 0, "Remove": 0}
    counts["Add"] += len([idea for idea in suggestion[0] if "Add" in idea])
    counts["Swap"] += len([idea for idea in suggestion[0] if "Swap" in idea])
    counts["Remove"] += len([idea for idea in suggestion[0] if "Remove" in idea])
    grasp_in_pixels, grasp_in_world, idx = env.grasp_cable_node(
        path_in_pixels, cable_orientations
    )
    env.robot.move_to_home()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
