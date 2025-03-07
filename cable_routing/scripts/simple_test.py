import rospy
import tyro
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.env.env import ExperimentEnv


def main(args: ExperimentConfig):

    rospy.init_node("pick_nic")
    env = ExperimentEnv(args)

    path_in_pixels, path_in_world, cable_orientations = env.update_cable_path()

    arm = "right"

    grasp_in_pixels, grasp_in_world, idx = env.grasp_cable_node(
        path_in_pixels, cable_orientations, "left"
    )

    env.robot.grippers_move_to("left", distance=3)
    delta = "left"
    env.robot.go_delta(
        left_delta=[0, 0, 0.05] if delta == "left" else None,
        right_delta=[0, 0, 0.05] if delta == "right" else None,
    )

    path_in_pixels, path_in_world, cable_orientations = env.update_cable_path()

    grasp_in_pixels, grasp_in_world, idx = env.grasp_cable_node(
        path_in_pixels, cable_orientations, arm
    )

    seq = ["right", "up", "up", "left"]  #  "left", "down", "right", "up", "left"]

    env.slideto_cable_node(
        path_in_pixels,
        path_in_world,
        cable_orientations,
        idx,
        arm=arm,
        side="right",
        display=False,
    )

    env.slideto_cable_node(
        path_in_pixels,
        path_in_world,
        cable_orientations,
        idx,
        arm=arm,
        side="up",
        display=False,
    )
    env.robot.go_delta(
        left_delta=[0, 0.05, 0.0] if delta == "left" else None,
        right_delta=[0, 0.0, 0.0] if delta == "right" else None,
    )

    env.slideto_cable_node(
        path_in_pixels,
        path_in_world,
        cable_orientations,
        idx,
        arm=arm,
        side="down",
        display=False,
    )

    env.slideto_cable_node(
        path_in_pixels,
        path_in_world,
        cable_orientations,
        idx,
        arm=arm,
        side="left",
        display=False,
    )

    other_arm = "left" if env.cable_in_arm == "right" else "right"

    # grasp_in_pixels, grasp_in_world, idx = env.release_cable_node(
    #     path_in_pixels, cable_orientations, other_arm
    # )

    # grasp_in_pixels, grasp_in_world, idx = env.grasp_cable_node(
    #     path_in_pixels, cable_orientations, other_arm
    # )

    # Safe quit.
    env.robot.go_delta(
        left_delta=[0, 0, 0.1] if env.cable_in_arm == "left" else None,
        right_delta=[0, 0, 0.05] if env.cable_in_arm == "right" else None,
    )
    env.robot.move_to_home()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
