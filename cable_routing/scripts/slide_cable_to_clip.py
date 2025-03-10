import rospy
import tyro
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.env.env import ExperimentEnv


def main(args: ExperimentConfig):

    rospy.init_node("pick_nic")
    env = ExperimentEnv(args)

    arm = "right"

    path_in_pixels, path_in_world, cable_orientations = env.update_cable_path(
        arm=arm, display=False
    )

    grasp_in_pixels, grasp_in_world, idx = env.dual_grasp_cable_node(
        path_in_pixels, cable_orientations, grasp_arm=arm, display=False
    )

    # grasp_in_pixels, grasp_in_world, idx = env.grasp_cable_node(
    #     path_in_pixels, cable_orientations, arm=arm, display=False
    # )

    env.slideto_cable_node(
        cable_path_in_pixel=path_in_pixels,
        # path_in_world,
        cable_orientations=cable_orientations,
        idx=idx,
        arm=arm,
        single_hand=False,
        side="left",
        display=False,
    )

    env.robot.go_delta(
        left_delta=[0, 0, 0.05] if env.cable_in_arm == "left" else None,
        right_delta=[0, 0, 0.05] if env.cable_in_arm == "right" else None,
    )
    env.robot.move_to_home()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
