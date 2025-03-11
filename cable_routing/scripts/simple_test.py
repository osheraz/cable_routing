from tracemalloc import start
import rospy
import tyro
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.env.env import ExperimentEnv
from cable_routing.env.ext_camera.utils.img_utils import distance


def main(args: ExperimentConfig):
    """
    Plan:
        Input: given the sequence of clips that is desired and the starting clip where the cable is already plugged in

        Need to route the cable through all of the desired clips in order

        First:
            Find the endpoint of the cable near the clip
            Trace using handloom a little bit down the cable and grasp (so you don't grasp right next to the clip)
                Grasp with enough closure to grab the cable but not enough to hold it in place (allow the cable to slip in the gripper)

        Next:
            Plan out the desired motion path through all the clips
                We know clip orientations so need to plan out how to route around each clip in the correct way
                Can add in the bungee style clips and have the gripper go through them

        Execute this motion with a single arm (no dual arm needed for the task)
    """
    MIN_DIST = 0.07  # minimum distance from a clip when doing the first grasp (in m)
    rospy.init_node("pick_nic")
    env = ExperimentEnv(args)

    routing = ["F", "G", "D"][::-1]

    print(env.route_cable(routing, display=False, dual_arm=True))
    exit()
    # clips = env.board.get_clips()
    # print(env.board.get_clips())
    # # env.board.visualize_board(env.workspace_img)
    # desired_routing = ["E", "K", "C"]

    # start_clip, end_clip = clips[desired_routing[0]], clips[desired_routing[-1]]

    # path_in_pixels, path_in_world, cable_orientations = env.update_cable_path(
    #     display=False,
    #     # start_points=[start_clip["x"], start_clip["y"]],
    # )

    # initial_grasp_idx = -1
    # curr_dist = 0
    # while curr_dist < MIN_DIST:
    #     initial_grasp_idx += 1
    #     curr_dist = distance(
    #         path_in_world[initial_grasp_idx], [start_clip["x"], start_clip["y"]]
    #     )

    # _, _, idx = env.grasp_cable_node(
    #     path_in_pixels,
    #     cable_orientations,
    #     arm="right",
    # )

    # env.slideto_cable_node(
    #     path_in_pixels, cable_orientations, idx, arm="right", display=True
    # )

    # arm = "right"  # do all manipulation with the right arm

    # print(env.rosute_around_clip("B", "I", "A"))
    # rospy.init_node("pick_nic")
    # env = ExperimentEnv(args)

    # path_in_pixels, path_in_world, cable_orientations = env.update_cable_path()

    # arm = "right"

    # grasp_in_pixels, grasp_in_world, idx = env.grasp_cable_node(
    #     path_in_pixels, cable_orientations, "left"
    # )

    # env.robot.grippers_move_to("left", distance=3)
    # delta = "left"
    # env.robot.go_delta(
    #     left_delta=[0, 0, 0.05] if delta == "left" else None,
    #     right_delta=[0, 0, 0.05] if delta == "right" else None,
    # )

    # path_in_pixels, path_in_world, cable_orientations = env.update_cable_path()

    # grasp_in_pixels, grasp_in_world, idx = env.grasp_cable_node(
    #     path_in_pixels, cable_orientations, arm
    # )

    # seq = ["right", "up", "up", "left"]  #  "left", "down", "right", "up", "left"]

    # env.slideto_cable_node(
    #     path_in_pixels,
    #     path_in_world,
    #     cable_orientations,
    #     idx,
    #     arm=arm,
    #     side="right",
    #     display=False,
    # )

    # env.slideto_cable_node(
    #     path_in_pixels,
    #     path_in_world,
    #     cable_orientations,
    #     idx,
    #     arm=arm,
    #     side="up",
    #     display=False,
    # )
    # env.robot.go_delta(
    #     left_delta=[0, 0.05, 0.0] if delta == "left" else None,
    #     right_delta=[0, 0.0, 0.0] if delta == "right" else None,
    # )

    # env.slideto_cable_node(
    #     path_in_pixels,
    #     path_in_world,
    #     cable_orientations,
    #     idx,
    #     arm=arm,
    #     side="down",
    #     display=False,
    # )

    # env.slideto_cable_node(
    #     path_in_pixels,
    #     path_in_world,
    #     cable_orientations,
    #     idx,
    #     arm=arm,
    #     side="left",
    #     display=False,
    # )

    # other_arm = "left" if env.cable_in_arm == "right" else "right"

    # # grasp_in_pixels, grasp_in_world, idx = env.release_cable_node(
    # #     path_in_pixels, cable_orientations, other_arm
    # # )

    # # grasp_in_pixels, grasp_in_world, idx = env.grasp_cable_node(
    # #     path_in_pixels, cable_orientations, other_arm
    # # )

    # # Safe quit.
    # env.robot.go_delta(
    #     left_delta=[0, 0, 0.1] if env.cable_in_arm == "left" else None,
    #     right_delta=[0, 0, 0.05] if env.cable_in_arm == "right" else None,
    # )
    # env.robot.move_to_home()


if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
