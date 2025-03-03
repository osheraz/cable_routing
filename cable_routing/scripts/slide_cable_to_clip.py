import rospy
import numpy as np
import tyro
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.env.env import ExperimentEnv


def main(args: ExperimentConfig):

    rospy.init_node("pick_nic")
    env = ExperimentEnv(args)

    path = env.update_cable_path()
    env.goto_cable_node(path)
    env.slideto_cable_node()

    env.robot.move_to_home()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
