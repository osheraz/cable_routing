from math import e
from re import S
import rospy
import cv2
from sympy import im
from tqdm import tqdm
from cable_routing.env.robots.yumi import YuMiRobotEnv
import tyro
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import random
import heapq
from autolab_core import RigidTransform, Point, CameraIntrinsics
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber
from cable_routing.handloom.handloom_pipeline.single_tracer import CableTracer
from cable_routing.env.board.new_board import Board
from cable_routing.algo.astar_board_planner import BoardPlanner, draw_planner_overlay
from cable_routing.env.ext_camera.utils.img_utils import (
    crop_img,
    crop_board,
    select_target_point,
    get_world_coord_from_pixel_coord,
    pick_target_on_path,
    find_nearest_point,
    get_perpendicular_orientation,
    get_path_angle,
    find_nearest_white_pixel,
    normalize,
)
from cable_routing.env.ext_camera.utils.pcl_utils import (
    project_points_to_image,
    get_rotation_matrix,
)
from cable_routing.handloom.handloom_pipeline.tracer import (
    TraceEnd,
)


class ExperimentEnv:

    def __init__(
        self,
        exp_config: ExperimentConfig,
    ):

        rospy.logwarn("Setting up the environment")
        self.exp_config = exp_config
        self.robot = YuMiRobotEnv(exp_config.robot_cfg)
        rospy.sleep(2.0)

        # todo: add cfg support for this
        self.zed_cam = ZedCameraSubscriber()
        while self.zed_cam.rgb_image is None or self.zed_cam.depth_image is None:
            rospy.sleep(0.1)
            rospy.loginfo("Waiting for images from ZED camera...")

        # camera to base transform
        self.T_CAM_BASE = {
            "left": RigidTransform.load(
                exp_config.cam_to_robot_left_trans_path
            ).as_frames(from_frame="zed", to_frame="base_link"),
            "right": RigidTransform.load(
                exp_config.cam_to_robot_right_trans_path
            ).as_frames(from_frame="zed", to_frame="base_link"),
        }

        # self.adjust_extrinsic("right", yaw=-5.0)

        self.tracer = CableTracer()
        self.planner = BoardPlanner(show_animation=False)

        self.board = Board(config_path=exp_config.board_cfg_path)
        rospy.logwarn("Env is ready")

        self.cable_in_arm = None

        self.workspace_img = self.zed_cam.get_rgb().copy()

    def adjust_extrinsic(self, arm, roll=0.0, pitch=0.0, yaw=0.0):

        rotation_tilt = get_rotation_matrix(roll, pitch, yaw)
        self.T_CAM_BASE[arm].rotation = rotation_tilt @ self.T_CAM_BASE[arm].rotation

    def get_extrinsic(self, arm):

        return self.T_CAM_BASE[arm]

    def set_board_region(self, img=None):

        if img is None:
            img = self.zed_cam.rgb_image

        _, self.point1, self.point2 = crop_board(img)

    def check_calibration(self, arm):
        """we are going to poke all the clips/plugs etc"""

        self.robot.move_to_home()

        self.board.visualize_board(self.zed_cam.rgb_image)

        clips = self.board.get_clips()

        abort = False
        for id, clip in clips.items():

            if not abort:
                clip_type = clip["type"]
                clip_ori = clip["orientation"]
                pixel_coord = (clip["x"], clip["y"])
                world_coord = get_world_coord_from_pixel_coord(
                    pixel_coord,
                    self.zed_cam.intrinsic,
                    self.T_CAM_BASE[arm],
                    is_clip=True,
                )

                print(
                    f"Poking clip {clip_type}, orientation: {clip_ori} at: {world_coord}"
                )

                # TODO add support for ori
                self.robot.single_hand_grasp(
                    arm, world_coord, eef_rot=np.deg2rad(clip_ori), slow_mode=True
                )
                self.robot.move_to_home()
                #                 arm, world_coord, eef_rot=cable_ori, slow_mode=True

                # abort = input("Abort? (y/n): ") == "y"

    def update_cable_path(
        self,
        arm,
        start_points=None,
        end_points=None,
        user_pick=False,
        display=False,
    ):

        # TODO convert_path_to_world_coord is w.r.t the right arm.

        path_in_pixels, _ = self.trace_cable(
            start_points=start_points,
            end_points=end_points,
            user_pick=user_pick,
            viz=display,
        )

        self.board.set_cable_path(path_in_pixels)

        path_in_world = self.convert_path_to_world_coord(path_in_pixels, arm=arm)

        cable_orientations = [
            get_perpendicular_orientation(
                path_in_world[idx - 1],
                path_in_world[idx],
            )
            for idx in range(1, len(path_in_world))
        ]

        cable_orientations.append(cable_orientations[-1])
        cable_orientations = np.unwrap(cable_orientations)
        cable_orientations = np.mod(cable_orientations, 2 * np.pi).tolist()

        if display:
            path_arr = np.array(path_in_world)

            plt.figure(figsize=(10, 8))
            plt.plot(path_arr[:, 0], path_arr[:, 1], "b-", label="Cable Path")

            for idx, (point, orientation) in enumerate(
                zip(path_in_world, cable_orientations)
            ):
                dx = 0.02 * np.cos(orientation)
                dy = 0.02 * np.sin(orientation)

                plt.arrow(
                    point[0],
                    point[1],
                    dx,
                    dy,
                    head_width=0.005,
                    head_length=0.01,
                    fc="red",
                    ec="red",
                )
                plt.text(point[0], point[1], f"{idx}", fontsize=12, color="darkgreen")

            plt.xlabel("X (meters)")
            plt.ylabel("Y (meters)")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.title("Cable Path with Orientations")
            plt.show()

        return path_in_pixels, path_in_world, cable_orientations

    def convert_path_to_world_coord(self, path, arm):

        world_path = []
        for pixel_coord in path:
            world_coord = get_world_coord_from_pixel_coord(
                pixel_coord, self.zed_cam.intrinsic, self.T_CAM_BASE[arm]
            )
            world_path.append(world_coord)

        return world_path

    def grasp_cable_node(
        self,
        path,
        cable_orientations,
        arm,
        idx=None,
        start_clip_id="A",
        display=False,
        user_pick=False,
    ):
        offset_distance = self.exp_config.grasp_cfg.offset_distance

        if idx is None and user_pick:
            frame = self.zed_cam.get_rgb()
            move_to_pixel = pick_target_on_path(frame, path)
            move_to_pixel, idx = find_nearest_point(path, move_to_pixel)
        else:

            nearest_clip = self.board.get_clips()[start_clip_id]
            nearest_clip = np.array([nearest_clip["x"], nearest_clip["y"]])
            start_trace = np.array(path[0])
            direction = (start_trace - nearest_clip) / np.linalg.norm(
                start_trace - nearest_clip
            )
            move_to_pixel, idx = find_nearest_point(
                path, nearest_clip + direction * offset_distance
            )

        world_coord = get_world_coord_from_pixel_coord(
            move_to_pixel,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE[arm],
            # depth_map=self.zed_cam.get_depth(),
        )

        print("Going to ", world_coord)

        path_in_world = self.convert_path_to_world_coord(path, arm=arm)

        cable_ori = cable_orientations[idx]

        if display:

            plt.figure()
            path_arr = np.array(path_in_world)

            plt.plot(path_arr[:, 0], path_arr[:, 1], "bo-", label="Cable Path")
            plt.plot(
                world_coord[0],
                world_coord[1],
                "rx",
                markersize=10,
                label="Grasp Target",
            )

            b = np.array(path_in_world[idx - 1])[:2]
            a = np.array(path_in_world[idx + 1])[:2]
            mid = (b + a) / 2

            perp_vec = np.array([np.cos(cable_ori), np.sin(cable_ori)]) * 0.02

            plt.plot(
                [mid[0] - perp_vec[0], mid[0] + perp_vec[0]],
                [mid[1] - perp_vec[1], mid[1] + perp_vec[1]],
                "r-",
                linewidth=2,
                label="Perpendicular Orientation",
            )

            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Cable Path and Grasp Orientation")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.show()

        self.robot.single_hand_grasp(
            arm, world_coord, eef_rot=cable_ori, slow_mode=True
        )

        self.cable_in_arm = arm

        return move_to_pixel, world_coord, idx

    def dual_grasp_cable_node(
        self,
        path,
        cable_orientations,  # ori will slightly be off
        grasp_arm,
        start_clip_id="A",
        idx=None,
        user_pick=False,
        display=False,
    ):
        follow_arm = "right" if grasp_arm=="left" else "left"
        print(f'{follow_arm=}, {grasp_arm}')
        print(f'{cable_orientations=}')
        path_in_world_grasp = self.convert_path_to_world_coord(path, arm=grasp_arm)
        path_in_world_follow = self.convert_path_to_world_coord(path, arm=follow_arm)

        if idx is None and user_pick:

            # grasp arm
            frame = self.zed_cam.get_rgb()
            move_to_pixel_grasp = pick_target_on_path(frame, path)
            move_to_pixel_grasp, idx = find_nearest_point(path, move_to_pixel_grasp)

            # follow_arm
            move_to_pixel_follow = pick_target_on_path(frame, path)
            move_to_pixel_follow, idx = find_nearest_point(path, move_to_pixel_follow)
        else:
            # find the nearest clip to trace

            offset_distance = self.exp_config.grasp_cfg.offset_distance
            min_distance = self.exp_config.grasp_cfg.min_distance
            jump = self.exp_config.grasp_cfg.jump
            exclusion_radius = self.exp_config.grasp_cfg.exclusion_radius

            clip_positions = [
                np.array([clip["x"], clip["y"]])
                for clip in self.board.get_clips().values()
            ]

            nearest_clip = self.board.get_clips()[start_clip_id]
            nearest_clip = np.array([nearest_clip["x"], nearest_clip["y"]])
            start_trace = np.array(path[0])

            direction = (start_trace - nearest_clip) / np.linalg.norm(
                start_trace - nearest_clip
            )

            move_to_pixel_grasp, idx = find_nearest_point(
                path, nearest_clip + direction * offset_distance
            )
            next_idx = idx + jump
            move_to_pixel_follow = np.array(path[next_idx])

            for _ in range(5):

                if np.linalg.norm(
                    move_to_pixel_follow - move_to_pixel_grasp
                ) >= min_distance and all(
                    np.linalg.norm(move_to_pixel_follow - clip) >= exclusion_radius
                    for clip in clip_positions
                ):
                    break
                next_idx += jump

                if next_idx >= len(path):
                    break

                move_to_pixel_follow = np.array(path[next_idx])
            else:
                print("Failed to find a second arm pose\nPlease pick grasp manually")
                move_to_pixel_follow, _ = find_nearest_point(
                    path, pick_target_on_path(frame, path)
                )

        world_coord_grasp = get_world_coord_from_pixel_coord(
            move_to_pixel_grasp,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE[grasp_arm],
        )

        cable_ori_grasp = cable_orientations[idx]

        world_coord_follow = get_world_coord_from_pixel_coord(
            move_to_pixel_follow,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE[follow_arm],
        )

        cable_ori_follow = cable_orientations[next_idx]

        if display:

            plt.figure()
            path_arr = np.array(path_in_world_grasp)
            path_arr2 = np.array(path_in_world_follow)

            plt.plot(path_arr[:, 0], path_arr[:, 1], "bo-", label="Cable Path")
            plt.plot(path_arr2[:, 0], path_arr2[:, 1], "bo-", label="Cable Path")

            plt.plot(
                world_coord_grasp[0],
                world_coord_grasp[1],
                "rx",
                markersize=10,
                label="Grasp Target",
            )

            plt.plot(
                world_coord_follow[0],
                world_coord_follow[1],
                "bx",
                markersize=10,
                label="Follow Target",
            )

            b = np.array(path_in_world_grasp[idx - 1])[:2]
            a = np.array(path_in_world_grasp[idx + 1])[:2]
            mid = (b + a) / 2

            perp_vec = (
                np.array([np.cos(cable_ori_grasp), np.sin(cable_ori_grasp)]) * 0.02
            )

            plt.plot(
                [mid[0] - perp_vec[0], mid[0] + perp_vec[0]],
                [mid[1] - perp_vec[1], mid[1] + perp_vec[1]],
                "r-",
                linewidth=2,
                label="Perpendicular Orientation",
            )

            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Cable Path and Grasp Orientation")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.show()

        world_coords = {
            "left": world_coord_follow if follow_arm == "left" else world_coord_grasp,
            "right": world_coord_follow if follow_arm == "right" else world_coord_grasp,
        }

        eef_rots = {
            "left": cable_ori_follow if follow_arm == "left" else cable_ori_grasp,
            "right": cable_ori_follow if follow_arm == "right" else cable_ori_grasp,
        }

        self.robot.dual_hand_grasp(
            left_world_coord=world_coords["left"],
            left_eef_rot=eef_rots["left"],
            right_world_coord=world_coords["right"],
            right_eef_rot=eef_rots["right"],
            grasp_arm=grasp_arm,
        )

        self.cable_in_arm = grasp_arm

        return move_to_pixel_grasp, world_coord_grasp, idx

    def goto_clip_node(self, arm, side="up", single_hand=True, display=False):

        frame = self.zed_cam.get_rgb()

        move_to_pixel = select_target_point(frame, rule="clip")

        # clip = self.board.find_nearest_clip([move_to_pixel])
        # clip_ori = clip["orientation"]
        # move_to_pixel = [clip["x"], clip["y"]]

        world_coord = get_world_coord_from_pixel_coord(
            move_to_pixel,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE[arm],
            # depth_map=self.zed_cam.get_depth(),
            # is_clip=True,
        )

        offset = {
            "left": [0, 0.02, 0],
            "right": [0, -0.02, 0],
            "down": [-0.02, 0, 0],
            "up": [0.02, 0, 0],
        }[side]

        target_coord = world_coord + np.array(offset)

        path_arr = np.array(
            self.convert_path_to_world_coord(self.board.cable_positions)
        )

        if display:
            plt.figure()
            plt.plot(path_arr[:, 0], path_arr[:, 1], "bo-")
            plt.plot(world_coord[0], world_coord[1], "gx", markersize=10)
            plt.plot(target_coord[0], target_coord[1], "rx", markersize=10)
            plt.arrow(
                world_coord[0],
                world_coord[1],
                offset[0],
                offset[1],
                head_width=0.01,
                head_length=0.015,
                fc="r",
                ec="r",
            )
            plt.axis("equal")
            plt.grid(True)
            plt.show()

            self.robot.go_delta(
                left_delta=[0, 0, 0.05] if arm == "left" else None,
                right_delta=[0, 0, 0.05] if arm == "right" else None,
            )

        self.robot.single_hand_move(arm, target_coord, slow_mode=True)

    def route_cable(
        self,
        routing: list[str],
        primary_arm,
        display=False,
        dual_arm=True,
        save_viz=False,
    ):

        clips = self.board.get_clips()
        start_clip, end_clip = clips[routing[0]], clips[routing[-1]]

        path_in_pixels, path_in_world, cable_orientations = self.update_cable_path(
            display=save_viz,
            arm=primary_arm,
        )

        secondary_arm = "left" if primary_arm == "right" else "right"

        initial_grasp_idx = -1

        if not dual_arm:
            primary_pixel_coord, primary_world_coord, main_initial_grasp_idx = (
                self.grasp_cable_node(
                    path_in_pixels,
                    cable_orientations,
                    arm=primary_arm,
                    display=display,
                )
            )

        else:
            _, _, secondary_initial_grasp_idx = self.dual_grasp_cable_node(
                path_in_pixels,
                cable_orientations,
                grasp_arm=primary_arm,
                display=display,
            )

        # move up to disentangle
        self.robot.go_delta([0, 0, 0.04], [0, 0, 0.04])

        for i in range(1, len(routing) - 1):
            seq, primary_arm = self.route_around_clip(
                routing[i - 1],
                routing[i],
                routing[i + 1],
                path_in_pixels=None,
                cable_orientations=None,
                idx=initial_grasp_idx,
                arm=primary_arm,
                dual_arm=dual_arm,
                display=display,
            )

        secondary_arm = "left" if primary_arm == "right" else "right"
        # finally just go to a pose above the final clip (x, y)
        z_offset = self.exp_config.grasp_cfg.z_offset
        self.robot.open_grippers(secondary_arm)
        self.robot.move_to_home(arm=secondary_arm)
        delta = {primary_arm: [0, 0, z_offset], secondary_arm: [0, 0, 0]}
        self.robot.go_delta(left_delta=delta["left"], right_delta=delta["right"])
        final_pixel_clip_coords = [end_clip["x"], end_clip["y"]]
        final_clip_coords = get_world_coord_from_pixel_coord(
            final_pixel_clip_coords,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE[secondary_arm],
            # depth_map=self.zed_cam.get_depth(),
            # is_clip=True,
        )

        final_clip_coords[2] = z_offset
        self.robot.single_hand_move(
            arm=primary_arm, world_coord=final_clip_coords, slow_mode=True
        )

    def route_around_clip(
        self,
        prev_clip_id: str,
        curr_clip_id: str,
        next_clip_id: str,
        path_in_pixels,
        # path_in_world,
        cable_orientations,
        idx,
        arm,
        dual_arm=False,
        display=False,
    ):
        clips = self.board.get_clips()
        curr_clip = clips[curr_clip_id]
        prev_clip = clips[prev_clip_id]
        next_clip = clips[next_clip_id]

        curr_x, curr_y = curr_clip["x"], curr_clip["y"]
        prev_x, prev_y = prev_clip["x"], prev_clip["y"]
        next_x, next_y = next_clip["x"], next_clip["y"]

        def calculate_sequence():
            """
            determines how to wrap around a give clip
            """

            num2dir = {0: "up", 1: "right", 2: "down", 3: "left"}
            dir2num = {val: key for key, val in num2dir.items()}
            clip_vecs = np.array([[0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0]])
            prev2curr = normalize(np.array([curr_x - prev_x, -(curr_y - prev_y), 0]))
            curr2prev = -prev2curr
            curr2next = normalize(np.array([next_x - curr_x, -(next_y - curr_y), 0]))
            clip_vec = clip_vecs[(curr_clip["orientation"] // 90 + 1) % 4]
            is_clockwise = np.cross(prev2curr, curr2next)[-1] > 0

            net_vector = curr2prev + curr2next
            if abs(net_vector[0]) > abs(net_vector[1]):
                if net_vector[0] > 0:
                    middle_node = dir2num["left"]
                else:
                    middle_node = dir2num["right"]
            else:
                if net_vector[1] > 0:
                    middle_node = dir2num["down"]
                else:
                    middle_node = dir2num["up"]

            if is_clockwise:
                sequence = [
                    num2dir[(middle_node + 1) % 4],
                    num2dir[middle_node],
                    num2dir[(middle_node - 1) % 4],
                ]
            else:
                sequence = [
                    num2dir[(middle_node - 1) % 4],
                    num2dir[middle_node],
                    num2dir[(middle_node + 1) % 4],
                ]

            return sequence, -1 if is_clockwise else 1

        def need_regrasp(curr_x, next_x, arm):
            curr_dir = next_x-curr_x
            if curr_dir>0 and arm=='left':
                return True
            elif curr_dir<0 and arm=='left':
                return False
            elif curr_dir>0 and arm=='right':
                return False
            else:
                return True
        sequence, rotation_dir = calculate_sequence()
        has_regrasped = 0


        for i, s in enumerate(sequence):

            print("Sliding", i, s)
            self.slideto_cable_node(
                path_in_pixels,
                cable_orientations,
                idx,
                clip_id=curr_clip_id,
                arm=arm,
                single_hand=not dual_arm,
                side=s,
                display=display,
            )

            # TODO: Fix the regrasp metric; can't use curr_x, next_x
            swap_arms = s =="left" or s=="right" #need_regrasp(curr_x, next_x, arm)
            if swap_arms:
                print("Swapping Arms")
                arm = self.swap_arms(arm, fixture_dir=s)

        # if switch_hands:
        # print("Swapping Arms")
        # arm = self.swap_arms(arm)


            # expected_rotation = (
            #     self.robot.get_gripper_rotation(arm) + 180 * rotation_dir
            # )

            # if expected_rotation < self.robot.rotation_limits[arm][0]:
            #     print("REGRASPING")
            #     print("Previous eef rotation", self.robot.get_gripper_rotation(arm))
            #     print(f"{expected_rotation=}")
            #     print("Rotating with direction = 1 (positive rotation)")
            #     self.regrasp(arm, direction=1)
            #     has_regrasped = 1
            #     print(f"{has_regrasped=}")
            #     print("New eef rotation", self.robot.get_gripper_rotation(arm))

            #     print()
            # elif expected_rotation > self.robot.rotation_limits[arm][1]:
            #     print("REGRASPING")
            #     print("Previous eef rotation", self.robot.get_gripper_rotation(arm))
            #     print(f"{expected_rotation=}")
            #     print("Rotating with direction = -1 (positive rotation)")
            #     self.regrasp(arm, direction=-1)
            #     has_regrasped = -1
            #     print(f"{has_regrasped=}")
            #     print("New eef rotation", self.robot.get_gripper_rotation(arm))
            # else:
            #     print("NOT REGRASPING")
            #     print("Current eef rotation", self.robot.get_gripper_rotation(arm))
            #     print(f"{expected_rotation=}")

        return sequence, arm

    def slideto_cable_node(
        self,
        cable_path_in_pixel,
        # path_in_world,
        cable_orientations,
        idx,
        arm,
        single_hand=True,
        clip_id=None,
        side="up",
        display=False,
    ):

        plan = self.plan_slide_to_cable_node(
            cable_path_in_pixel,
            idx,
            clip_id=clip_id,
            arm=arm,
            side=side,
            display=display,
        )

        if single_hand:

            self.execute_slide_to_cable_node(
                plan["waypoints"],
                plan["target_coord"],
                cable_orientations,
                arm=arm,
            )
        else:
            # add Literal - both.
            self.execute_dual_slide_to_cable_node(
                plan["waypoints"],
                plan["target_coord"],
                cable_orientations,
                arm=arm,
            )

    def regrasp(self, arm, direction):
        """
        reorients gripper by going to the exact same position, but just with a 180 degree rotation in the ee
        """
        secondary_arm = "right" if arm == "left" else "left"
        curr_secondary_pose = self.robot.get_ee_pose()[
            0 if secondary_arm == "left" else 1
        ].translation
        print("secondary arm pose")
        print(curr_secondary_pose)
        lower_secondary_pose = list(curr_secondary_pose)[:2] + [0.01]
        print(np.array(lower_secondary_pose))
        self.robot.single_hand_move(
            arm=secondary_arm,
            world_coord=np.array(lower_secondary_pose),
        )
        self.robot.open_grippers(side=arm)
        delta = {arm: [0, 0, 0.095], secondary_arm: [0, 0, 0]}
        self.robot.go_delta(left_delta=delta["left"], right_delta=delta["right"])

        # self.robot.grippers_move_to(arm, distance=self.robot.gripper_opening)

        self.robot.set_speed("normal")
        self.robot.rotate_gripper(np.pi * direction, arm=arm)
        # self.robot.grippers_move_to(hand=arm, distance=10)
        delta[arm][-1] *= -1
        self.robot.set_speed("slow")
        self.robot.go_delta(left_delta=delta["left"], right_delta=delta["right"])
        self.robot.close_grippers(side=arm)
        self.robot.set_speed("normal")

    def plan_slide_to_cable_node(
        self, path_in_pixel, idx, clip_id=None, arm=None, side="up", display=False
    ):
        if clip_id is None:
            move_to_pixel = select_target_point(self.workspace_img, rule="clip")
            clip = self.board.get_clips()[self.board.find_nearest_clip([move_to_pixel])]
        else:
            clip = self.board.get_clips()[clip_id]
        clip_pixel = [clip["x"], clip["y"]]

        world_coord = get_world_coord_from_pixel_coord(
            clip_pixel,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE[arm],
            is_clip=True,
        )

        pixel_offset = {
            "left": [-80, 0],  # must be higher than inflation_radius
            "right": [80, 0],
            "down": [0, 80],
            "up": [0, -80],
        }[side]

        goal_pixel = np.array(clip_pixel) + np.array(pixel_offset)

        target_coord = get_world_coord_from_pixel_coord(
            goal_pixel,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE[arm],
            is_clip=True,
        )

        # start_pixel = path_in_pixel[idx]
        current_pose = self.robot.get_ee_pose()[0 if arm == "left" else 1]
        #
        start_pixel = project_points_to_image(
            np.array([current_pose.translation]),
            self.zed_cam.intrinsic._K,
            self.T_CAM_BASE[arm],
            self.workspace_img.shape[:2],
        )[0]
        # start_pixel = select_target_point(self.workspace_img, rule="START START")
        # start_pixel[0] -= self.board.point1[0]
        # start_pixel[1] -= self.board.point1[1]

        waypoints_in_pixels = self.planner.plan_path(
            start_pixel=start_pixel,
            goal_pixel=goal_pixel,
        )

        print(f"waypoints {waypoints_in_pixels}")
        waypoints = self.convert_path_to_world_coord(waypoints_in_pixels, arm=arm)

        if display:

            draw_planner_overlay(
                self.workspace_img,
                self.planner,
                start_pixel,
                goal_pixel,
                waypoints_in_pixels,
                self.planner.resolution,
            )

            path_arr = np.array(
                self.convert_path_to_world_coord(self.board.cable_positions, arm=arm)
            )

            plt.figure(figsize=(8, 6))
            plt.plot(path_arr[:, 0], path_arr[:, 1], "bo-", label="Cable Path")
            plt.plot(
                world_coord[0],
                world_coord[1],
                "gx",
                markersize=10,
                label="Clip Location",
            )
            plt.plot(
                target_coord[0],
                target_coord[1],
                "rx",
                markersize=10,
                label="Target Point",
            )

            for i, waypoint in enumerate(waypoints):
                plt.plot(waypoint[0], waypoint[1], "ko", markersize=5)
                if i > 0:
                    plt.plot(
                        [waypoints[i - 1][0], waypoint[0]],
                        [waypoints[i - 1][1], waypoint[1]],
                        "k--",
                    )

            plt.arrow(
                world_coord[0],
                world_coord[1],
                target_coord[0] - world_coord[0],
                target_coord[1] - world_coord[1],
                head_width=0.01,
                head_length=0.015,
                fc="r",
                ec="r",
                label="Approach Direction",
            )

            plt.xlabel("X (meters)")
            plt.ylabel("Y (meters)")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.title(f"Sliding to Clip (Approach: {side})")
            plt.show()

        return {
            "waypoints": waypoints,
            "target_coord": target_coord,
        }

    def execute_slide_to_cable_node(
        self, waypoints, target_coord, cable_orientations, arm=None, display=False
    ):
        if arm is None:
            arm = self.cable_in_arm

        current_pose = self.robot.get_ee_pose()[0 if arm == "left" else 1]
        target_coord[2] = current_pose.translation[2]

        cur_gripper_pos = self.robot.get_gripper_pose(arm)
        self.robot.grippers_move_to(
            arm, distance=cur_gripper_pos + self.robot.gripper_opening
        )

        eef_orientations = [
            get_perpendicular_orientation(waypoints[i - 1], waypoints[i])
            for i in range(1, len(waypoints))
        ]
        eef_orientations.append(eef_orientations[-1])

        eef_orientations = np.unwrap(eef_orientations)
        eef_orientations = np.mod(eef_orientations, 2 * np.pi).tolist()

        if display:
            path_arr = np.array(waypoints)

            plt.figure(figsize=(10, 8))
            plt.plot(path_arr[:, 0], path_arr[:, 1], "b-", label="Cable Path")

            for idx, (point, orientation) in enumerate(
                zip(waypoints, eef_orientations)
            ):
                dx = 0.02 * np.cos(orientation)
                dy = 0.02 * np.sin(orientation)

                plt.arrow(
                    point[0],
                    point[1],
                    dx,
                    dy,
                    head_width=0.005,
                    head_length=0.01,
                    fc="red",
                    ec="red",
                )
                plt.text(point[0], point[1], f"{idx}", fontsize=12, color="darkgreen")

            plt.xlabel("X (meters)")
            plt.ylabel("Y (meters)")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.title("Cable Path with Orientations")
            plt.show()

        # Move to the first waypoint at the correct height
        self.robot.set_speed("slow")
        poses = [
            RigidTransform(translation=waypoint, rotation=current_pose.rotation)
            for waypoint in [current_pose.translation, waypoints[0]]
        ]
        self.robot.plan_and_execute_linear_waypoints(arm, waypoints=poses)
        self.robot.set_speed("normal")

        # Align orientation to first segment
        poses = [
            RigidTransform(
                translation=waypoints[0],
                rotation=current_pose.rotation,
            ),
            RigidTransform(
                translation=waypoints[0],
                rotation=RigidTransform.x_axis_rotation(-np.pi)
                @ RigidTransform.z_axis_rotation(-eef_orientations[0]),
            ),
        ]
        self.robot.plan_and_execute_linear_waypoints(arm, waypoints=poses)

        # Follow the full path
        poses = [
            RigidTransform(
                translation=wp,
                rotation=RigidTransform.x_axis_rotation(-np.pi)
                @ RigidTransform.z_axis_rotation(-ori),
            )
            for wp, ori in zip(waypoints, eef_orientations)
        ]

        for start_pose, end_pose in zip(poses, poses[1:]):
            self.robot.plan_and_execute_linear_waypoints(
                arm, waypoints=[start_pose, end_pose]
            )
            actual_pose = self.robot.get_ee_pose()[0 if arm == "left" else 1]
            error = np.linalg.norm(actual_pose.translation - end_pose.translation)
            print(f"Pose deviation = {error:.4f} meters")
            # if error > POSITION_TOLERANCE:
            #     raise RuntimeError(f"Failed to reach pose: error = {error:.4f} meters")

        self.robot.set_ee_pose(
            left_pose=(poses[-1] if arm == "left" else None),
            right_pose=(poses[-1] if arm == "right" else None),
        )

    def execute_dual_slide_to_cable_node(
        self,
        waypoints,
        target_coord,
        cable_orientations,
        arm,
        will_regrasp = False,
        display=False,
    ):

        s_arm = "right" if arm == "left" else "left"

        eefs_pose = self.robot.get_ee_pose()

        current_pose = eefs_pose[0 if arm == "left" else 1]
        second_pose = eefs_pose[1 if arm == "left" else 0]

        x_min, x_max = 0.0, 0.6
        y_min, y_max = -0.4, 0.4
        z_min, z_max = 0.0, 0.3

        offset_vector = np.array([0.1, 0.1, 0.1])
        waypoints_secondary = []

        y_threshold = 0.1 

        for i in range(len(waypoints) - 1):

            motion_direction = waypoints[i + 1] - waypoints[i]
            motion_direction /= np.linalg.norm(motion_direction)

            secondary_wp = waypoints[i] + motion_direction * np.linalg.norm(
                offset_vector
            )

            if waypoints_secondary:
                last_valid_wp = waypoints_secondary[-1]
            else:
                last_valid_wp = waypoints[i]

            # Clip X and Y coordinates to stay within the defined bounds
            secondary_wp[0] = (
                np.clip(secondary_wp[0], x_min, x_max)
                if x_min <= secondary_wp[0] <= x_max
                else last_valid_wp[0]
            )
            secondary_wp[1] = (
                np.clip(secondary_wp[1], y_min, y_max)
                if y_min <= secondary_wp[1] <= y_max
                else last_valid_wp[1]
            )
            secondary_wp[2] = 0.1  # Keep Z constant

            if arm == "right":
                # Secondary arm must always have y >= primary arm + threshold
                if secondary_wp[1] < waypoints[i][1] + y_threshold:
                    secondary_wp[1] = waypoints[i][1] + y_threshold
            elif arm == "left":
                # Secondary arm must always have y <= primary arm - threshold
                if secondary_wp[1] > waypoints[i][1] - y_threshold:
                    secondary_wp[1] = waypoints[i][1] - y_threshold

            # Check for crossing between successive waypoints
            if i > 0:
                prev_diff = waypoints_secondary[i - 1][1] - waypoints[i - 1][1]
                curr_diff = secondary_wp[1] - waypoints[i][1]

                # If sign changes, a crossing has occurred. Correct the position with threshold.
                if prev_diff * curr_diff < 0:
                    if arm == "right":
                        secondary_wp[1] = waypoints[i][1] + y_threshold
                    elif arm == "left":
                        secondary_wp[1] = waypoints[i][1] - y_threshold

            waypoints_secondary.append(secondary_wp)

        self.robot.grippers_move_to(s_arm, distance=self.robot.gripper_opening - 2)
        self.robot.grippers_move_to(arm, distance=self.robot.gripper_opening - 2)

        # calc only w.r.t the first arm
        # cur_rot = np.deg2rad(self.robot.get_gripper_rotation(arm))

        eef_orientations = [
            get_perpendicular_orientation(waypoints[i - 1], waypoints[i])
            for i in range(1, len(waypoints))
        ]

        eef_orientations = [
            np.array(e) for e in eef_orientations
        ]

        eef_orientations.append(eef_orientations[-1])

        eef_orientations = np.unwrap(eef_orientations)
        eef_orientations = np.mod(eef_orientations, 2 * np.pi).tolist()


        if display:
            path_arr = np.array(waypoints)
            path_arr2 = np.array(waypoints_secondary)

            plt.figure(figsize=(10, 8))
            plt.plot(path_arr[:, 0], path_arr[:, 1], "b-", label="Cable Path")
            plt.plot(path_arr2[:, 0], path_arr2[:, 1], "r-", label="Cable Path2")

            for idx, (point, orientation) in enumerate(
                zip(waypoints, eef_orientations)
            ):
                dx = 0.02 * np.cos(orientation)
                dy = 0.02 * np.sin(orientation)

                plt.arrow(
                    point[0],
                    point[1],
                    dx,
                    dy,
                    head_width=0.005,
                    head_length=0.01,
                    fc="red",
                    ec="red",
                )
                plt.text(point[0], point[1], f"{idx}", fontsize=12, color="darkgreen")

            for idx, (point, orientation) in enumerate(
                zip(waypoints_secondary, eef_orientations)
            ):
                dx = 0.02 * np.cos(orientation)
                dy = 0.02 * np.sin(orientation)

                plt.arrow(
                    point[0],
                    point[1],
                    dx,
                    dy,
                    head_width=0.005,
                    head_length=0.01,
                    fc="red",
                    ec="red",
                )
                plt.text(point[0], point[1], f"{idx}", fontsize=12, color="darkgreen")

            plt.xlabel("X (meters)")
            plt.ylabel("Y (meters)")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.title("Cable Path with Orientations")
            plt.show()

        # Move to the first waypoint at the correct height
        # self.robot.set_speed("slow")

        poses = [
            RigidTransform(translation=waypoint, rotation=current_pose.rotation)
            for waypoint in [current_pose.translation, waypoints[0]]
        ]

        poses_secondary = [
            RigidTransform(translation=waypoint, rotation=second_pose.rotation)
            for waypoint in [second_pose.translation, waypoints_secondary[0]]
        ]

        self.robot.plan_and_execute_linear_waypoints(s_arm, waypoints=poses_secondary)

        poses_secondary = [
            RigidTransform(translation=wp, rotation=second_pose.rotation)
            for wp in waypoints_secondary
        ]
        # move secondary arm before moving primary arm to reduce the likelihood of cable tangling
        # TODO check why 3
        self.robot.plan_and_execute_linear_waypoints(
            s_arm, waypoints=poses_secondary[0:3]
        )


        self.robot.plan_and_execute_linear_waypoints(arm, waypoints=poses)

        self.robot.set_speed("normal")

        # Align orientation to the first waypoint
        poses = [
            RigidTransform(
                translation=waypoints[0],
                rotation=current_pose.rotation,
            ),
            RigidTransform(
                translation=waypoints[0],
                rotation=RigidTransform.x_axis_rotation(-np.pi)
                @ RigidTransform.z_axis_rotation(-eef_orientations[0]),
            ),
        ]

        self.robot.plan_and_execute_linear_waypoints(arm, waypoints=poses)

        poses = [
            RigidTransform(
                translation=wp,
                rotation=RigidTransform.x_axis_rotation(-np.pi)
                @ RigidTransform.z_axis_rotation(-ori),
            )
            for wp, ori in zip(waypoints, eef_orientations)
        ]

        poses_secondary = [
            RigidTransform(translation=wp, rotation=second_pose.rotation)
            for wp in waypoints_secondary
        ]

        # poses_secondary = [
        #     RigidTransform(
        #         translation=wp,
        #         rotation=RigidTransform.x_axis_rotation(-np.pi)
        #         @ RigidTransform.z_axis_rotation(-ori),
        #     )
        #     for wp, ori in zip(waypoints_secondary, eef_second_orientations)
        # ]
        # self.robot.plan_and_execute_linear_waypoints(
        #     arms="left", waypoints=poses_secondary
        # )

        # exit()

        for i in range(len(poses) - 2):

            # if arm == "right":
            #     self.robot.plan_and_execute_linear_waypoints(
            #         arms="both",
            #         waypoints=(poses_secondary[i : i + 3], poses[i : i + 3]),
            #     )
            # else:
            #     self.robot.plan_and_execute_linear_waypoints(
            #         arms="both",
            #         waypoints=(poses[i : i + 3], poses_secondary[i : i + 3]),
            #     )


            self.robot.plan_and_execute_linear_waypoints(
                    arm, waypoints=poses[i : i + 3]
                )


            if not i + 4 > len(poses_secondary):
                self.robot.plan_and_execute_linear_waypoints(
                    s_arm, waypoints=poses_secondary[i + 1 : i + 4]
                )

            # actual_pose_primary = self.robot.get_ee_pose()[0 if arm == "left" else 1]
            # actual_pose_secondary = self.robot.get_ee_pose()[
            #     1 if s_arm == "left" else 0
            # ]

            # error_primary = np.linalg.norm(
            #     actual_pose_primary.translation - end_pose_primary.translation
            # )
            # error_secondary = np.linalg.norm(
            #     actual_pose_secondary.translation - end_pose_secondary.translation
            # )

            # print(f"Step {i+1}:")
            # print(f"  Primary Pose deviation = {error_primary:.4f} meters")
            # print(f"  Secondary Pose deviation = {error_secondary:.4f} meters")

        self.robot.set_ee_pose(
            left_pose=(poses[-1] if arm == "left" else poses_secondary[-1]),
            right_pose=(poses[-1] if arm == "right" else poses_secondary[-1]),
        )

        # for start_pose, end_pose in zip(poses, poses[1:]):
        #     self.robot.plan_and_execute_linear_waypoints(
        #         arm, waypoints=[start_pose, end_pose]
        #     )
        #     actual_pose = self.robot.get_ee_pose()[0 if arm == "left" else 1]
        #     error = np.linalg.norm(actual_pose.translation - end_pose.translation)
        #     print(f"Pose deviation = {error:.4f} meters")

        # self.robot.set_ee_pose(
        #     left_pose=(poses[-1] if arm == "left" else None),
        #     right_pose=(poses[-1] if arm == "right" else None),
        # )

    def swap_arms(
        self, arm, fixture_dir
    ):
        pass
        """
        Go down, let go with both arms, record pixel position of each, swap rolls, then call the analytic grasp finder function
        """

        # Function: 

        # align the second arm with offset
        eefs_pose = self.robot.get_ee_pose()
        current_pose = eefs_pose[0 if arm == "left" else 1]
        second_pose = eefs_pose[1 if arm == "left" else 0]

        align_pose = second_pose.copy()
        # TODO: Update The line below to accommodate routings either away or toward the robot
        align_pose.translation[0] = max(current_pose.translation[0] + 0.15, 0.6)

        s_arm = 'left' if arm=='right' else 'right'
        
        poses_dict = {arm:current_pose,
                      s_arm:align_pose}
        
        self.robot.set_ee_pose(poses_dict['left'], poses_dict['right'])
        ######################

        # 
        eefs_pose = self.robot.get_ee_pose()
        current_pose = eefs_pose[0 if arm == "left" else 1]
        second_pose = eefs_pose[1 if arm == "left" else 0]

        align_pose.translation[1] = current_pose.translation[1]
        poses_dict = {arm:current_pose,
                      s_arm:align_pose}
        self.robot.set_ee_pose(poses_dict['left'], poses_dict['right'])
        self.robot.rotate_gripper(angle = -np.pi/2, arm= s_arm)

        eefs_pose = self.robot.get_ee_pose()
        current_pose = eefs_pose[0 if arm == "left" else 1]
        second_pose = eefs_pose[1 if arm == "left" else 0]

        #
        new_pose = current_pose.copy()
        new_pose.translation[2] = -0.02
        new_second_pose = second_pose.copy()
        new_second_pose.translation[2] = -0.02

        s_arm = 'left' if arm=='right' else 'right'
        poses_dict = {arm:new_pose,
                      s_arm:new_second_pose}
        self.robot.set_ee_pose(poses_dict['left'], poses_dict['right'])


        # self.robot.go_delta([0,0,0.0], [0,0,0.04])
        # self.robot.go_delta([0,0,0.04], [0,0,0.0])

        # Move up, go_delta not working
        self.robot.open_grippers(arm=s_arm)
        eefs_pose = self.robot.get_ee_pose()
        primary_pose = eefs_pose[0 if arm == "left" else 1]
        secondary_pose = eefs_pose[1 if arm == "left" else 0]

        second_pose.translation[2] += 0.1
        poses_dict = {arm:primary_pose,
                      s_arm:secondary_pose}
        self.robot.set_ee_pose(poses_dict['left'], poses_dict['right'])

        s_pose = self.robot.get_ee_pose()[1 if arm == "left" else 0]
        s_pose.translation[0] -= 0.05
        self.robot.move_to_home(arm=s_arm)

        eefs_pose = self.robot.get_ee_pose()
        primary_pose = eefs_pose[0 if arm == "left" else 1]
        secondary_pose = eefs_pose[1 if arm == "left" else 0]

        if fixture_dir == "left":
            primary_pose.translation[1] += 0.04
        else:
            primary_pose.translation[1] -= 0.04
        
        poses_dict = {arm:primary_pose,
                      s_arm:secondary_pose}
        self.robot.set_ee_pose(poses_dict['left'], poses_dict['right'])

        self.robot.open_grippers(arm=arm)
        eefs_pose = self.robot.get_ee_pose()
        primary_pose = eefs_pose[0 if arm == "left" else 1]
        secondary_pose = eefs_pose[1 if arm == "left" else 0]

        primary_pose.translation[2] += 0.1
        poses_dict = {arm:primary_pose,
                      s_arm:secondary_pose}
        self.robot.set_ee_pose(poses_dict['left'], poses_dict['right'])

        
        arm_pose = self.robot.get_ee_pose()[0 if arm == "left" else 1]
        self.robot.move_to_home(arm=arm)



        arm_pixel_pose = project_points_to_image(
            np.array([arm_pose.translation]),
            self.zed_cam.intrinsic._K,
            self.T_CAM_BASE[arm],
            self.workspace_img.shape[:2],
        )[0]

        s_arm_pixel_pose = project_points_to_image(
            np.array([s_pose.translation]),
            self.zed_cam.intrinsic._K,
            self.T_CAM_BASE[s_arm],
            self.workspace_img.shape[:2],
        )[0]

        self.perform_nearest_analytic_grasp_dual(prev_grasp_point=s_arm_pixel_pose,
                                                 prev_follow_point=arm_pixel_pose,
                                                 grasp_arm="left" if arm == "left" else "right",
                                                 follow_arm="right" if arm == "left" else "left",
                                                 visualize=True
                                                 )
        return s_arm #new primary arm

    def trace_cable(
        self,
        img=None,
        start_points=None,
        end_points=None,
        viz=False,
        user_pick=False,
        save=False,
    ):

        if viz and save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_folder = f"{self.exp_config.save_folder}/run_{timestamp}"
            os.makedirs(save_folder, exist_ok=True)
        else:
            save_folder = None
        p1, p2 = self.board.point1, self.board.point2

        clips = list(self.board.get_clips().values())
        for clip in clips:
            clip["x"] -= p1[0]
            clip["y"] -= p1[1]

        if img is None:
            img = self.zed_cam.rgb_image.copy()

        img = crop_img(img, p1, p2)  # TODO: fix!

        if start_points == None and user_pick:
            start_points = select_target_point(img, rule="start")
            filtered_points = [start_points]
        else:
            nearest_clip = self.board.get_clips()["A"]
            nearest_clip["x"] -= p1[0]
            nearest_clip["y"] -= p1[1]
            valid_points = find_nearest_white_pixel(img, nearest_clip)
            # remove points near clips
            filtered_points = [
                p
                for p in valid_points
                if all(
                    np.linalg.norm(
                        np.array([p[0], p[1]]) - np.array([clip["x"], clip["y"]])
                    )
                    >= 20
                    for clip in clips
                )
            ]
            start_points = random.choice(filtered_points)

        print("Start points", start_points)

        print(f"Starting trace at start point: {start_points}")
        status = None
        for _ in range(len(filtered_points)):
            try:
                path, status = self.tracer.trace(
                    img=img.copy(),
                    start_points=start_points,
                    end_points=end_points,
                    clips=clips,
                    save_folder=save_folder,
                    idx=0,
                    viz=viz,
                )
                break
            except:
                start_points = random.choice(filtered_points)
                print("Failed to trace with Analytical Tracer, trying again :(")
                continue

        print("Tracing status:", status)
        connection = 0

        if clips is not None:

            while status == TraceEnd.CLIP:
                connection += 1
                nearest_clip = self.board.find_nearest_clip(path)

                if nearest_clip is None:
                    print("Error: No nearest clip found. Stopping tracing.")
                    break

                skip_distance = 40
                print("Clip Orientation:", nearest_clip["orientation"])
                clip_angle = np.deg2rad(nearest_clip["orientation"])

                path_angle = get_path_angle(path, N=3)
                print("Path Estimated Orientation:", np.rad2deg(path_angle))

                options = [
                    (clip_angle, "Clip ori"),
                    (path_angle, "Path ori"),
                ]

                start_options = []
                for ang, label in options:
                    opt_x = int(path[-1][0] + skip_distance * np.cos(ang))
                    opt_y = int(path[-1][1] + skip_distance * np.sin(ang))
                    start_options.append((opt_x, opt_y, label))

                new_x, new_y = start_options[-1][:2]

                start_points = [new_x, new_y]
                path = np.vstack([path, start_points])

                if viz:
                    img_display = img.copy()
                    cv2.polylines(
                        img_display,
                        [path.astype(np.int32)],
                        isClosed=False,
                        color=(0, 255, 0),
                        thickness=2,
                    )

                    for x, y, label in start_options:
                        color = (0, 0, 255) if label == "Clip ori" else (255, 0, 0)
                        cv2.circle(img_display, (x, y), 5, color, -1)
                        cv2.putText(
                            img_display,
                            label,
                            (x + 20, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )

                    cv2.circle(
                        img_display,
                        (nearest_clip["x"], nearest_clip["y"]),
                        5,
                        (255, 255, 255),
                        -1,
                    )

                    cv2.imshow("Trace Restart Options", img_display)
                    plt.imsave(f"{save_folder}/restart_{connection}.png", img_display)

                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                print(f"Resuming trace at new start point: {start_points}")

                new_path, status = self.tracer.trace(
                    img=img.copy(),
                    start_points=start_points,
                    end_points=end_points,
                    clips=clips,
                    save_folder=save_folder,
                    last_path=path,  # comment to use analytic_tracer
                    idx=connection,
                    viz=viz,
                )

                path = np.vstack([path, new_path])
                print("Tracing status:", status)

        # TODO: find a better way
        path = [(x + p1[0], y + p1[1]) for x, y in path]

        return path, status

    def get_nearest_analytic_grasp_point(self, start_point, img=None, visualize=False):
        '''
        Finds the nearest grasp point on the cable to a given position (start_point) using the analytic 
        color thresholding method

        Arguments:
        start_point ((int, int)): coordinates of the point to find a nearby grasp to, in pixel space
        img (Image): Image to use for grasp identification, None by default
        visualize (bool): indicates whether or not to visualize the planned grasps
        '''
        if img is None:
            img = self.zed_cam.rgb_image.copy()
        squared_magnitude = lambda my_tuple: sum([i**2 for i in my_tuple])

        x, y = start_point[0], start_point[1]

        directions = sum([[(a, b) for a in [-1, 0, 1]] for b in [-1, 0, 1]], start=[])

        heap = [(0, (x, y))]
        visited = set()
        visited.add((x, y))

        direction = (1, 0)
        steps_in_interval = 0
        interval_length = 1

        while heap:
            dist, pos = heapq.heappop(heap)
            if squared_magnitude(img[pos[1]][pos[0]]) > 140000:
                x = pos[0]
                y = pos[1]
                break
            for dx, dy in directions:
                nx, ny = pos[0] + dx, pos[1] + dy
                neighbor = (nx, ny)
                if 0 <= nx < img.shape[1] and 0 <= ny < img.shape[0]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_dist = squared_magnitude(
                            (neighbor[0] - start_point[0], neighbor[1] - start_point[1])
                        )
                        heapq.heappush(heap, (new_dist, neighbor))

        delta_x = start_point[0] - x
        delta_y = start_point[1] - y

        predicted_x = x
        predicted_y = y
        predicted_orientation = 180 / np.pi * np.arctan2(delta_y, delta_x)


        if visualize:
            def draw_point(
                img, x, y, label="Default", color=(0, 255, 255), orientation=0
            ):
                center = (x, y)
                cv2.circle(img, center, 10, color, -1)

                arrow_length = 30
                angle_rad = np.deg2rad(orientation)

                arrow_start = (
                    int(center[0] - (arrow_length / 2) * np.cos(angle_rad)),
                    int(center[1] - (arrow_length / 2) * np.sin(angle_rad)),
                )
                arrow_end = (
                    int(center[0] + (arrow_length / 2) * np.cos(angle_rad)),
                    int(center[1] + (arrow_length / 2) * np.sin(angle_rad)),
                )

                cv2.arrowedLine(img, arrow_start, arrow_end, (255, 255, 0), 2)

                cv2.putText(
                    img,
                    label,
                    (x + 15, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            img_display = img.copy()

            # Draw instructions on top-left
            instructions = ["Visualizing Nearest Cable Point"]
            for i, text in enumerate(instructions):
                cv2.putText(
                    img_display,
                    text,
                    (10, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            draw_point(
                    img_display,
                    start_point[0],
                    start_point[1],
                    "Previous",
                    (0, 0, 255),
                    0
                )
            draw_point(
                    img_display,
                    predicted_x,
                    predicted_y,
                    "Planned",
                    (255, 0, 0),
                    predicted_orientation
                )
            
            cv2.imshow("Board Setup", img_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return (predicted_x, predicted_y), predicted_orientation
    
    def perform_nearest_analytic_grasp_dual(
        self,
        prev_grasp_point,
        prev_follow_point,
        grasp_arm,
        follow_arm,
        visualize=False
    ):
        # grasp arm
        move_to_pixel_grasp, cable_ori_grasp = self.get_nearest_analytic_grasp_point(prev_grasp_point, visualize=visualize)
        cable_ori_grasp = np.deg2rad(-cable_ori_grasp + 90)

        # follow_arm
        move_to_pixel_follow, cable_ori_follow = self.get_nearest_analytic_grasp_point(prev_follow_point, visualize=visualize)
        cable_ori_follow = np.deg2rad(-cable_ori_follow + 90)

        
        world_coord_grasp = get_world_coord_from_pixel_coord(
            move_to_pixel_grasp,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE[grasp_arm],
        )

        world_coord_follow = get_world_coord_from_pixel_coord(
            move_to_pixel_follow,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE[follow_arm],
        )

        
        world_coords = {
            "left": world_coord_follow if follow_arm == "left" else world_coord_grasp,
            "right": world_coord_follow if follow_arm == "right" else world_coord_grasp,
        }

        print(cable_ori_grasp)

        eef_rots = {
            "left": cable_ori_follow if follow_arm == "left" else cable_ori_grasp,
            "right": cable_ori_follow if follow_arm == "right" else cable_ori_grasp,
        }

        self.robot.dual_hand_grasp(
            left_world_coord=world_coords["left"],
            left_eef_rot=eef_rots["left"],
            right_world_coord=world_coords["right"],
            right_eef_rot=eef_rots["right"],
            grasp_arm=grasp_arm,
        )

        self.cable_in_arm = grasp_arm

        return move_to_pixel_grasp, world_coord_grasp