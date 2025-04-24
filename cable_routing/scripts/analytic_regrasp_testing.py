from tracemalloc import start
import rospy
import tyro
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.env.env import ExperimentEnv


def main(args: ExperimentConfig):

    rospy.init_node("pick_nic")

    env = ExperimentEnv(args)
    env.robot.open_grippers()

    routing = ["A", "C", "E", "I"]

    # env.route_cable(
    #         routing, display=False, dual_arm=True, primary_arm="right", save_viz=False
    #     )

    print(env.get_nearest_analytic_grasp_point((1155, 538)))
    print(env.get_nearest_analytic_grasp_point((758, 420)))

    env.perform_nearest_analytic_grasp_dual((1155, 538), (758, 420), visualize=True)
    
    exit()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
