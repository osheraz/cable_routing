import rospy
import tyro
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.env.env import ExperimentEnv


def main(args: ExperimentConfig):

    rospy.init_node("pick_nic")
    env = ExperimentEnv(args)

    path_in_pixels, path_in_world, cable_orientations = env.update_cable_path(
        display=False
    )

    grasp_in_pixels, grasp_in_world, idx = env.grasp_cable_node(
        path_in_pixels, cable_orientations, display=True
    )

    env.slideto_cable_node(
        path_in_pixels,
        path_in_world,
        cable_orientations,
        idx,
        side="left",
        display=True,
    )

    # Safe quit.
    env.robot.go_delta(
        left_delta=[0, 0, 0.05] if env.cable_in_arm == "left" else None,
        right_delta=[0, 0, 0.05] if env.cable_in_arm == "right" else None,
    )
    env.robot.move_to_home()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
