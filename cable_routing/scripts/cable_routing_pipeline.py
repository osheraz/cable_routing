from tracemalloc import start
import rospy
import tyro
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.env.env import ExperimentEnv


def main(args: ExperimentConfig):

    rospy.init_node("pick_nic")

    env = ExperimentEnv(args)
    env.robot.open_grippers()

    routing = ["A", "C", "E"]

    env.route_cable(
            routing, 
            display=False, 
            dual_arm=True,
            primary_arm="left",
            save_viz=False
        )
    
    exit()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)

