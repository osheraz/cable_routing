from math import e
from re import S
import rospy
from std_srvs.srv import Trigger, TriggerResponse

import cv2
from sympy import im
from tqdm import tqdm
from cable_routing.env.robots.yumi import YuMiRobotEnv
import tyro
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
import os
import random
import threading
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
    is_cable_pixel,
    Visualizer,
)
from cable_routing.env.ext_camera.utils.pcl_utils import (
    project_points_to_image,
    get_rotation_matrix,
)
from cable_routing.handloom.handloom_pipeline.tracer import (
    TraceEnd,
)

from cable_routing.env.robots.misc import (
    calculate_sequence,
    need_regrasp,
    run_with_fallback,
)
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import traceback
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored, cprint


#################################################################
# TODO: This class should split into multiple classes, one for each module
# Splitting the code into smaller classes will make it easier to maintain and understand.
# It will also help in unit testing and debugging.
# The ExperimentEnv class is responsible for setting up the environment
# managing the robot, camera, and board, and handling the routing of cables.
#################################################################


class ExperimentEnv:

    def __init__(
        self,
        exp_config: ExperimentConfig,
    ):

        rospy.logwarn("Setting up the environment")
        self.exp_config = exp_config
        self.robot = YuMiRobotEnv(exp_config.robot_cfg)

        rospy.sleep(2.0)

        self.zed_cam = ZedCameraSubscriber()
        while self.zed_cam.rgb_image is None or self.zed_cam.depth_image is None:
            rospy.sleep(0.1)
            rospy.loginfo("Waiting for images from ZED camera...")

        # Camera to base transform
        self.T_CAM_BASE = {
            "left": RigidTransform.load(
                exp_config.cam_to_robot_left_trans_path
            ).as_frames(from_frame="zed", to_frame="base_link"),
            "right": RigidTransform.load(
                exp_config.cam_to_robot_right_trans_path
            ).as_frames(from_frame="zed", to_frame="base_link"),
        }

        # handloom pipeline wrapper
        self.tracer = CableTracer()

        # slide planner
        self.planner = BoardPlanner(show_animation=False)

        self.board = Board(config_path=exp_config.board_cfg_path)
        rospy.logwarn("Env is ready")

        self.accumulated_length_px = 0.0
        self.cable_in_arm = None
        self.swapped = False
        self.swaps = 0
        self.workspace_img = self.zed_cam.get_rgb().copy()

        # self.visualizer = Visualizer()

        # manual reset service with ros
        self._start_manual_reset_service()

        # in case we need to monitor the reset_triggered variable
        # self._start_reset_listener()

        # in case we want to check the calibration
        # self.check_calibration("left")
        # self.check_calibration("right")

        # In case we need to adjust the camera extrinsic manually:
        # self._adjust_extrinsic("right", yaw=-5.0)

    def _start_manual_reset_service(self):
        """Start a ROS service to manually reset the robot."""
        rospy.Service("/manual_reset_yumi", Trigger, self._handle_manual_reset)

    def _handle_manual_reset(self, req):
        """Handle the manual reset request."""

        rospy.logwarn("[ManualReset] Manual reset requested via ROS service.")
        try:
            self.robot._on_interface_reset_request()
            return TriggerResponse(
                success=True, message="Reset completed successfully."
            )
        except Exception as e:
            rospy.logerr(f"[ManualReset] Reset failed: {e}")
            return TriggerResponse(success=False, message=str(e))

    def _start_reset_listener(self):
        """Start a thread to monitor the reset_triggered variable."""

        def monitor():
            last_state = None
            while True:
                current_state = self.robot.reset_triggered
                if current_state != last_state:
                    print(f"[Listener] reset_triggered changed: {current_state}")
                    last_state = current_state
                time.sleep(0.05)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def _adjust_extrinsic(self, arm, roll=0.0, pitch=0.0, yaw=0.0):
        """Adjust the camera extrinsic parameters."""
        rotation_tilt = get_rotation_matrix(roll, pitch, yaw)
        self.T_CAM_BASE[arm].rotation = rotation_tilt @ self.T_CAM_BASE[arm].rotation

    def get_extrinsic(self, arm):
        """Get the camera to base transform for the specified arm."""
        return self.T_CAM_BASE[arm]

    def set_board_region(self, img=None):
        """Set the region of interest for the board."""
        if img is None:
            img = self.zed_cam.rgb_image

        _, self.point1, self.point2 = crop_board(img)

    def compute_routing_length_px(self, routing):
        """Compute the total length of the routing path in pixels."""

        clips = self.board.get_clips()

        total_length = 0.0
        for i in range(len(routing) - 1):
            clip_a = clips[routing[i]]
            clip_b = clips[routing[i + 1]]

            ax, ay = clip_a["x"], clip_a["y"]
            bx, by = clip_b["x"], clip_b["y"]

            dist = np.linalg.norm(np.array([bx - ax, by - ay]))
            total_length += dist

        return total_length

    def update_routing_progress_px(self, routing, current_idx):
        """Update the routing progress based on the current index."""
        # Check if the current index is within the valid range
        if current_idx <= 0 or current_idx >= len(routing):
            return
        # Calculate the accumulated length in pixels
        clips = self.board.get_clips()
        a = np.array(
            [clips[routing[current_idx - 1]]["x"], clips[routing[current_idx - 1]]["y"]]
        )
        b = np.array(
            [clips[routing[current_idx]]["x"], clips[routing[current_idx]]["y"]]
        )
        self.accumulated_length_px += np.linalg.norm(b - a)
        # Calculate the total length of the routing path
        total = self.compute_routing_length_px(routing)
        print(
            f"[Progress] {self.accumulated_length_px:.1f} / {total:.1f} pixels routed"
        )

        return self.accumulated_length_px / total

    def check_calibration(self, arm):
        """Check if the camera is calibrated correctly."""
        """We are going to poke all the clips/plugs one by one"""

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

                self.robot.single_hand_grasp(
                    arm, world_coord, eef_rot=np.deg2rad(clip_ori), slow_mode=True
                )
                self.robot.move_to_home()
                # abort = input("Abort? (y/n): ") == "y"

    #################################################################
    # TODO: Refactor - split here into a separate metaclass with inheritance
    #################################################################

    def update_cable_path(
        self,
        arm,
        start_points=None,
        end_points=None,
        user_pick=False,
        save_vis=False,
        display=False,
    ):
        """Update the cable path by tracing it from start to end points using HANDLOOM"""

        # Trace the cable path using start point around "A".
        path_in_pixels, _ = self.trace_cable(
            start_points=start_points,
            end_points=end_points,
            user_pick=user_pick,
            viz=display,
            save=save_vis,
        )

        self.board.set_cable_path(path_in_pixels)

        # convert the pixel path to world coordinates
        path_in_world = self.convert_path_to_world_coord(path_in_pixels, arm=arm)

        # get the cable orientations (prependicular to the path)
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

        # Display the cable path and orientations
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
        """Convert the cable path from pixel coordinates to world coordinates."""
        world_path = []
        for pixel_coord in path:
            world_coord = get_world_coord_from_pixel_coord(
                pixel_coord, self.zed_cam.intrinsic, self.T_CAM_BASE[arm]
            )
            world_path.append(world_coord)

        return world_path

    def route_cable(
        self,
        routing: list[str],
        primary_arm,
        display=False,
        dual_arm=True,
        save_viz=False,
    ):
        """Route the cable around the clips in the specified order."""

        # get the pixel position of all the clips
        clips = self.board.get_clips()
        start_clip, end_clip = clips[routing[0]], clips[routing[-1]]
        progress = 0

        # get the pixel position of the cable path
        path_in_pixels, path_in_world, cable_orientations = self.update_cable_path(
            save_vis=save_viz,
            display=display,
            arm=primary_arm,
        )

        secondary_arm = "left" if primary_arm == "right" else "right"
        initial_grasp_idx = -1

        # Grasp the cable node at the start clip with 1/2 arms
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

            # Most of the time, we will be using dual arm grasp
            _, _, secondary_initial_grasp_idx = self.dual_grasp_cable_node(
                path_in_pixels,
                cable_orientations,
                grasp_arm=primary_arm,
                display=display,
            )

        # Move up a bit to disentangle
        self.robot.go_delta([0, 0, 0.04], [0, 0, 0.04])

        # Apply routing
        for i in range(1, len(routing) - 1):
            # break each clip into 3 segments.

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
                progress=progress,
            )
            # update the routing progress
            progress = self.update_routing_progress_px(routing, i)

        # Closing ceremony - go to a pose above the final clip (x, y)
        secondary_arm = "left" if primary_arm == "right" else "right"
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
        cable_orientations,
        idx,
        arm,
        dual_arm=False,
        display=False,
        progress=0.0,
    ):

        clips = self.board.get_clips()
        curr_clip = clips[curr_clip_id]
        prev_clip = clips[prev_clip_id]
        next_clip = clips[next_clip_id]

        # get the sequence - clock\counterclockwise rotation
        sequence, _ = calculate_sequence(curr_clip, prev_clip, next_clip)

        for i, s in enumerate(sequence):

            cprint(
                f"Sliding {i}: --> {s}; From: {prev_clip_id}, Curr: {curr_clip_id}  Next: {next_clip_id}",
                "green",
            )

            self.slideto_cable_node(
                path_in_pixels,
                cable_orientations,
                idx,
                clip_id=curr_clip_id,
                arm=arm,
                single_hand=not dual_arm,
                side=s,
                display=display,
                swapped=self.swapped,
            )

            swap_arms = s == "left" or s == "right"
            swap_arms &= need_regrasp(curr_clip, next_clip, prev_clip, arm)
            swap_arms &= progress < 0.5

            if swap_arms:
                arm = self.swap_arms(
                    arm, prev_clip, curr_clip, next_clip, fixture_dir=s
                )
                self.swapped = not self.swapped
                self.swaps += 1
                # skip "up"
                break

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
        swapped=False,
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
            self.execute_dual_slide_to_cable_node(
                plan["waypoints"],
                plan["target_coord"],
                cable_orientations,
                arm=arm,
                swapped=swapped,
            )

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

    def _generate_secondary_waypoints(self, waypoints, current_pose, arm):
        x_min, x_max = self.exp_config.grasp_cfg.x_min, self.exp_config.grasp_cfg.x_max
        y_min, y_max = self.exp_config.grasp_cfg.y_min, self.exp_config.grasp_cfg.y_max
        y_threshold = self.exp_config.grasp_cfg.y_threshold

        waypoints_secondary = []

        for i in range(len(waypoints) - 1):
            motion_direction = waypoints[i + 1] - waypoints[i]
            motion_direction /= np.linalg.norm(motion_direction)
            secondary_wp = waypoints[i] + motion_direction * np.linalg.norm(
                self.exp_config.grasp_cfg.offset_vector
            )

            last_valid_wp = (
                waypoints_secondary[-1] if waypoints_secondary else waypoints[i]
            )

            # Clamp to bounds
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

            # Adjust z
            cur_x = current_pose.translation[0]
            min_x, max_x = 0.33, 0.0
            z_min = 0.02
            clamped_x = max(min_x, min(cur_x, max_x))
            scale = (clamped_x - min_x) / (max_x - min_x)
            z_max = self.exp_config.grasp_cfg.z_slide
            secondary_wp[2] = z_max * (1 - scale) + z_min * scale

            # Enforce offset constraints
            if arm == "right" and secondary_wp[1] < waypoints[i][1] + y_threshold:
                secondary_wp[1] = waypoints[i][1] + y_threshold
            elif arm == "left" and secondary_wp[1] > waypoints[i][1] - y_threshold:
                secondary_wp[1] = waypoints[i][1] - y_threshold

            # Prevent path crossing
            if i > 0:
                prev_diff = waypoints_secondary[i - 1][1] - waypoints[i - 1][1]
                curr_diff = secondary_wp[1] - waypoints[i][1]
                if prev_diff * curr_diff < 0:
                    if arm == "right":
                        secondary_wp[1] = waypoints[i][1] + y_threshold
                    elif arm == "left":
                        secondary_wp[1] = waypoints[i][1] - y_threshold

            waypoints_secondary.append(secondary_wp)

        return waypoints_secondary

    def _compute_orientations(self, waypoints, swapped, display=False):

        eef_orientations = [
            get_perpendicular_orientation(waypoints[i - 1], waypoints[i])
            for i in range(1, len(waypoints))
        ]
        raw = eef_orientations.copy()
        eef_orientations = [np.array(e) for e in eef_orientations]
        eef_orientations.append(eef_orientations[-1])

        eef_orientations = np.unwrap(eef_orientations)
        eef_orientations = np.mod(eef_orientations, 2 * np.pi).tolist()

        smoothed = [eef_orientations[0]]
        for ori in eef_orientations[1:]:
            if abs(ori - smoothed[-1]) > np.pi:
                smoothed.append(smoothed[-1])
            else:
                smoothed.append(ori)
        eef_orientations = smoothed

        eef_second = eef_orientations.copy()

        if swapped:
            eef_orientations = [(ori - np.pi) % (2 * np.pi) for ori in eef_orientations]
        if self.swaps % 2 == 0 and self.swaps > 1:
            eef_second = [(ori - np.pi) % (2 * np.pi) for ori in eef_orientations]

        if display:
            plt.figure(figsize=(8, 3))
            plt.plot(eef_orientations, label="eef_first")
            # plt.plot(raw, label="raw")

            plt.plot(eef_second, label="eef_second", linestyle="--")
            plt.legend()
            plt.title("Orientation Comparison")
            plt.grid(True)
            plt.show()

        return eef_orientations, eef_second

    def execute_dual_slide_to_cable_node(
        self,
        waypoints,
        target_coord,
        cable_orientations,
        arm,
        will_regrasp=False,
        display=False,
        swapped=False,
    ):
        """
        Plans and executes a dual-arm sliding motion along a cable path.
        One arm performs the primary motion while the secondary arm follows a constrained,
        tangential path to avoid interference or cable twisting.

        Args:
            waypoints: List of primary motion 3D coordinates.
            target_coord: Target end point (unused here but could be for future refinement).
            cable_orientations: Orientation hints for end-effector alignment (also not used here).
            arm: The primary arm ('left' or 'right') to execute the motion.
            will_regrasp: Flag for whether a regrasp is expected (unused here).
            display: Whether to show a matplotlib plot of the path and orientations.
        """

        s_arm = "right" if arm == "left" else "left"

        # Get current end-effector poses for both arms
        eefs_pose = self.robot.get_ee_pose()
        current_pose = eefs_pose[0 if arm == "left" else 1]  # sliding hand
        second_pose = eefs_pose[1 if arm == "left" else 0]  # supporting hand

        # Bounds and thresholds for secondary arm path generation
        waypoints_secondary = self._generate_secondary_waypoints(
            waypoints, current_pose, arm
        )

        # Compute end-effector orientation based on motion direction
        eef_orientations, eef_second = self._compute_orientations(waypoints, swapped)

        # Optional visualization of paths and orientations
        if display:
            self.visualizer.visualize(waypoints, waypoints_secondary, eef_orientations)

        # Executing the motion

        # Slightly open both grippers before motion
        self.robot.grippers_move_to(s_arm, distance=self.robot.gripper_opening)  # )- 2)
        self.robot.grippers_move_to(arm, distance=self.robot.gripper_opening)  # - 2)

        # Move supporting arm to start position
        poses_secondary = [
            RigidTransform(translation=waypoint, rotation=second_pose.rotation)
            for waypoint in [second_pose.translation, waypoints_secondary[0]]
        ]
        result = run_with_fallback(
            self.robot.plan_and_execute_linear_waypoints,
            10,
            s_arm,
            waypoints=poses_secondary,
            fallback_func=self.robot.set_ee_pose,
        )

        # Align orientation of secondary arm
        poses_secondary_align = [
            RigidTransform(
                translation=waypoints_secondary[0], rotation=second_pose.rotation
            ),
            RigidTransform(
                translation=waypoints_secondary[0],
                rotation=RigidTransform.x_axis_rotation(-np.pi)
                @ RigidTransform.z_axis_rotation(-eef_second[0]),
            ),
        ]

        result = run_with_fallback(
            self.robot.plan_and_execute_linear_waypoints,
            10,
            s_arm,
            waypoints=poses_secondary_align,
            fallback_func=self.robot.set_ee_pose,
        )

        # Advance secondary arm slightly before primary moves
        poses_secondary = [
            RigidTransform(
                translation=wp,
                rotation=RigidTransform.x_axis_rotation(-np.pi)
                @ RigidTransform.z_axis_rotation(-ori),
            )
            for wp, ori in zip(waypoints_secondary, eef_second)
        ]

        result = run_with_fallback(
            self.robot.plan_and_execute_linear_waypoints,
            10,
            s_arm,
            waypoints=poses_secondary[0:3],
            fallback_func=self.robot.set_ee_pose,
        )

        # Move primary arm to start position
        poses = [
            RigidTransform(translation=waypoint, rotation=current_pose.rotation)
            for waypoint in [current_pose.translation, waypoints[0]]
        ]
        result = run_with_fallback(
            self.robot.plan_and_execute_linear_waypoints,
            10,
            arm,
            waypoints=poses,
            fallback_func=self.robot.set_ee_pose,
        )

        self.robot.set_speed("normal")

        # Align primary arm orientation to start of motion
        poses = [
            RigidTransform(translation=waypoints[0], rotation=current_pose.rotation),
            RigidTransform(
                translation=waypoints[0],
                rotation=RigidTransform.x_axis_rotation(-np.pi)
                @ RigidTransform.z_axis_rotation(-eef_orientations[0]),
            ),
        ]
        result = run_with_fallback(
            self.robot.plan_and_execute_linear_waypoints,
            10,
            arm,
            waypoints=poses,
            fallback_func=self.robot.set_ee_pose,
        )

        # Prepare full motion paths for both arms
        poses = [
            RigidTransform(
                translation=wp,
                rotation=RigidTransform.x_axis_rotation(-np.pi)
                @ RigidTransform.z_axis_rotation(-ori),
            )
            for wp, ori in zip(waypoints, eef_orientations)
        ]
        # poses_secondary = [
        #     RigidTransform(translation=wp, rotation=second_pose.rotation)
        #     for wp in waypoints_secondary
        # ]

        # Interleaved execution of both arms along the path (3-waypoint sliding windows)
        for i in range(len(poses) - 2):

            result = run_with_fallback(
                self.robot.plan_and_execute_linear_waypoints,
                10,
                arm,
                waypoints=poses[i : i + 3],
                fallback_func=self.robot.set_ee_pose,
            )

            if not i + 4 > len(poses_secondary):
                result = run_with_fallback(
                    self.robot.plan_and_execute_linear_waypoints,
                    10,
                    s_arm,
                    waypoints=poses_secondary[i + 1 : i + 4],
                    fallback_func=self.robot.set_ee_pose,
                )

        # Final alignment of both arms at the end

        run_with_fallback(
            self.robot.set_ee_pose,
            10,
            left_pose=(poses[-1] if arm == "left" else poses_secondary[-1]),
            right_pose=(poses[-1] if arm == "right" else poses_secondary[-1]),
        )

        # result = run_with_fallback(
        #     self.robot.plan_and_execute_linear_waypoints,
        #     10,
        #     s_arm,
        #     waypoints=[poses_secondary[-1], poses_secondary[-1]],
        # )
        # result = run_with_fallback(
        #     self.robot.plan_and_execute_linear_waypoints,
        #     10,
        #     arm,
        #     waypoints=[poses[-1], poses[-1]],
        # )

    #################################################################
    # TODO: Refactor - split here into a separate "primitive" class
    #################################################################

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
        """Grasp the cable node at the specified index with one hand."""

        offset_distance = self.exp_config.grasp_cfg.offset_distance

        if idx is None and user_pick:
            # support for user picking
            frame = self.zed_cam.get_rgb()
            move_to_pixel = pick_target_on_path(frame, path)
            move_to_pixel, idx = find_nearest_point(path, move_to_pixel)
        else:
            # find the nearest clip to trace (around clip A)
            nearest_clip = self.board.get_clips()[start_clip_id]
            nearest_clip = np.array([nearest_clip["x"], nearest_clip["y"]])
            start_trace = np.array(path[0])
            direction = (start_trace - nearest_clip) / np.linalg.norm(
                start_trace - nearest_clip
            )
            # find the nearest point on the path
            move_to_pixel, idx = find_nearest_point(
                path, nearest_clip + direction * offset_distance
            )

        # move pixel to world coordinates
        world_coord = get_world_coord_from_pixel_coord(
            move_to_pixel,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE[arm],
            # depth_map=self.zed_cam.get_depth(), # by passing depth map, we can get the z coord
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
        """Grasp the cable node with both arms."""

        follow_arm = "right" if grasp_arm == "left" else "left"

        # convert the path to world coordinates (for both arms)
        path_in_world_grasp = self.convert_path_to_world_coord(path, arm=grasp_arm)
        path_in_world_follow = self.convert_path_to_world_coord(path, arm=follow_arm)

        if idx is None and user_pick:
            # support for user picking
            # grasp arm
            frame = self.zed_cam.get_rgb()
            move_to_pixel_grasp = pick_target_on_path(frame, path)
            move_to_pixel_grasp, idx = find_nearest_point(path, move_to_pixel_grasp)

            # follow_arm
            move_to_pixel_follow = pick_target_on_path(frame, path)
            move_to_pixel_follow, idx = find_nearest_point(path, move_to_pixel_follow)
        else:

            # we need to find 2 grasp points which are far enough apart
            # and not too close to the clips

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

            # find the next point on the path
            for _ in range(5):
                # check if the next point is far enough from the grasp point
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
                # find the next point on the path
                move_to_pixel_follow = np.array(path[next_idx])
            else:
                print("Failed to find a second arm pose\nPlease pick grasp manually")
                move_to_pixel_follow, _ = find_nearest_point(
                    path, pick_target_on_path(frame, path)
                )

        # convert the pixel coordinates to world coordinates
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

        # support for both directions
        world_coords = {
            "left": world_coord_follow if follow_arm == "left" else world_coord_grasp,
            "right": world_coord_follow if follow_arm == "right" else world_coord_grasp,
        }

        eef_rots = {
            "left": cable_ori_follow if follow_arm == "left" else cable_ori_grasp,
            "right": cable_ori_follow if follow_arm == "right" else cable_ori_grasp,
        }

        # perform the dual hand grasp
        self.robot.dual_hand_grasp(
            left_world_coord=world_coords["left"],
            left_eef_rot=eef_rots["left"],
            right_world_coord=world_coords["right"],
            right_eef_rot=eef_rots["right"],
            grasp_arm=grasp_arm,
        )
        self.cable_in_arm = grasp_arm

        return move_to_pixel_grasp, world_coord_grasp, idx

    def swap_arms(self, arm, prev_clip, curr_clip, next_clip, fixture_dir):
        """
        Go down, let go with both arms, record pixel position of each,
        swap rolls, then call the analytic grasp finder function
        """
        curr_x, curr_y = curr_clip["x"], curr_clip["y"]
        prev_x, prev_y = prev_clip["x"], prev_clip["y"]
        next_x, next_y = next_clip["x"], next_clip["y"]
        s_arm = "left" if arm == "right" else "right"

        cprint(f"Before Swap: Leading = {arm}, Following = {s_arm}", "green")
        cprint("Swapping...", "green")
        cprint(f"After  Swap: Leading = {s_arm}, Following = {arm}", "green")

        eefs_pose = self.robot.get_ee_pose()
        current_pose = eefs_pose[0 if arm == "left" else 1]
        second_pose = eefs_pose[1 if arm == "left" else 0]
        init_arm_pose = current_pose.copy()

        # Release the second arm
        new_second_pose = second_pose.copy()
        new_second_pose.translation[0] = current_pose.translation[0]
        new_second_pose.translation[2] = 0.01
        s_arm = "left" if arm == "right" else "right"
        poses_dict = {arm: None, s_arm: new_second_pose}
        self.robot.set_ee_pose(poses_dict["left"], poses_dict["right"])
        self.robot.open_grippers(arm=s_arm)

        # Jiggle motion: quick back-and-forth in Y direction
        for i in range(3):
            offset = (-1) ** i * 0.005
            jiggle_pose = new_second_pose.copy()
            jiggle_pose.translation[0] += offset
            jiggle_pose.translation[2] += 0.005  # small upward nudge
            self.robot.set_ee_pose(
                left_pose=jiggle_pose if s_arm == "left" else None,
                right_pose=jiggle_pose if s_arm == "right" else None,
            )
            time.sleep(0.1)

        # Slide the first arm to safe position
        # TODO: actually it has to be with the routing directions
        align_pose = current_pose.copy()
        align_pose.translation[2] = 0.01

        # if curr_y > next_y:
        align_pose.translation[0] = min(
            current_pose.translation[0] + 0.22, self.exp_config.grasp_cfg.x_max
        )
        # else:
        #     align_pose.translation[0] = max(
        #         current_pose.translation[0] - 0.25, self.exp_config.grasp_cfg.x_min
        #     )
        poses_dict = {arm: align_pose, s_arm: None}
        self.robot.set_ee_pose(poses_dict["left"], poses_dict["right"])
        # now the y value.
        # eefs_pose = self.robot.get_ee_pose()
        # second_pose = eefs_pose[1 if arm == "left" else 0]
        # # Determine new Y position: move above or below current_pose
        # if curr_x > next_x:
        #     align_pose.translation[1] = max(
        #         current_pose.translation[1] - 0.15, self.exp_config.grasp_cfg.y_min
        #     )
        # else:
        #     align_pose.translation[1] = min(
        #         current_pose.translation[1] + 0.15, self.exp_config.grasp_cfg.y_max
        #     )

        # poses_dict = {arm: align_pose, s_arm: None}
        # self.robot.set_ee_pose(poses_dict["left"], poses_dict["right"])
        # second_pose = self.robot.get_ee_pose()[1 if arm == "left" else 0]

        # Move down both to the board - slightlty above clips

        # self.robot.rotate_gripper(angle=-np.pi / 2, arm=s_arm)
        # self.robot.rotate_gripper(angle=np.pi / 2, arm=s_arm)

        # Both home at this point.
        s_arm_pixel_pose = project_points_to_image(
            np.array([init_arm_pose.translation]),
            self.zed_cam.intrinsic._K,
            self.T_CAM_BASE[s_arm],
            self.workspace_img.shape[:2],
        )[0]

        self.perform_nearest_analytic_grasp_dual(
            prev_grasp_point=None,
            prev_follow_point=s_arm_pixel_pose,
            grasp_arm="left" if arm == "left" else "right",
            follow_arm="right" if arm == "left" else "left",
            visualize=False,
        )

        self.robot.grippers_move_to(
            "left" if arm == "left" else "right",
            distance=self.robot.gripper_opening - 2,
        )

        return s_arm

    def swap_arms_release(self, arm, prev_clip, curr_clip, next_clip, fixture_dir):
        """
        Go down, let go with both arms, record pixel position of each,
        swap rolls, then call the analytic grasp finder function
        """
        curr_x, curr_y = curr_clip["x"], curr_clip["y"]
        prev_x, prev_y = prev_clip["x"], prev_clip["y"]
        next_x, next_y = next_clip["x"], next_clip["y"]
        s_arm = "left" if arm == "right" else "right"

        print("Swapping")

        # first, move the supporting arm to safe release position
        # shady - first move the x value
        eefs_pose = self.robot.get_ee_pose()
        current_pose = eefs_pose[0 if arm == "left" else 1]
        second_pose = eefs_pose[1 if arm == "left" else 0]

        align_pose = second_pose.copy()
        # align_pose.translation[0] = min(
        #     current_pose.translation[0] + 0.15, self.exp_config.grasp_cfg.x_max
        # )

        # Determine new X position: move to the side of current_pose
        if next_x > curr_x:
            align_pose.translation[0] = min(
                current_pose.translation[0] + 0.15, self.exp_config.grasp_cfg.x_max
            )
        else:
            align_pose.translation[0] = max(
                current_pose.translation[0] - 0.15, self.exp_config.grasp_cfg.x_min
            )

        poses_dict = {arm: None, s_arm: align_pose}
        self.robot.set_ee_pose(poses_dict["left"], poses_dict["right"])

        # now the y value.
        eefs_pose = self.robot.get_ee_pose()
        second_pose = eefs_pose[1 if arm == "left" else 0]
        # Determine new Y position: move above or below current_pose
        if curr_y > next_y:
            align_pose.translation[1] = max(
                current_pose.translation[1] - 0.15, self.exp_config.grasp_cfg.y_min
            )
        else:
            align_pose.translation[1] = min(
                current_pose.translation[1] + 0.15, self.exp_config.grasp_cfg.y_max
            )

        poses_dict = {arm: None, s_arm: align_pose}
        self.robot.set_ee_pose(poses_dict["left"], poses_dict["right"])
        second_pose = self.robot.get_ee_pose()[1 if arm == "left" else 0]

        # Move down both to the board - slightlty above clips
        new_pose = current_pose.copy()
        current_pose.translation[2] = 0.01
        new_second_pose = second_pose.copy()
        new_second_pose.translation[2] = 0.01
        s_arm = "left" if arm == "right" else "right"
        poses_dict = {arm: new_pose, s_arm: new_second_pose}
        self.robot.set_ee_pose(poses_dict["left"], poses_dict["right"])

        # Jiggle Jiggle - second arm
        self.robot.open_grippers(arm=s_arm)
        # self.robot.rotate_gripper(angle=-np.pi / 2, arm=s_arm)
        # self.robot.rotate_gripper(angle=np.pi / 2, arm=s_arm)

        # Move up and then home - second arm
        eefs_pose = self.robot.get_ee_pose()
        primary_pose = eefs_pose[0 if arm == "left" else 1]
        secondary_pose = eefs_pose[1 if arm == "left" else 0]

        last_s_pose = secondary_pose.copy()  # record for grasp retrace

        secondary_pose.translation[2] += 0.1
        poses_dict = {arm: None, s_arm: secondary_pose}
        self.robot.set_ee_pose(poses_dict["left"], poses_dict["right"])
        self.robot.move_to_home(arm=s_arm)

        # # First arm  - move a bit from the clip
        # if fixture_dir == "left":
        #     primary_pose.translation[1] += 0.04
        # else:
        #     primary_pose.translation[1] += 0.04
        # poses_dict = {arm: primary_pose, s_arm: None}
        # self.robot.set_ee_pose(poses_dict["left"], poses_dict["right"])

        last_arm_pose = primary_pose.copy()

        # Jiggle Jiggle - arm
        self.robot.open_grippers(arm=arm)
        # self.robot.rotate_gripper(angle=-np.pi / 2, arm=arm)
        # self.robot.rotate_gripper(angle=np.pi / 2, arm=arm)

        # Move up - them home - first arm
        eefs_pose = self.robot.get_ee_pose()
        primary_pose = self.robot.get_ee_pose()[0 if arm == "left" else 1]
        primary_pose.translation[2] += 0.1
        poses_dict = {arm: primary_pose, s_arm: None}
        self.robot.set_ee_pose(poses_dict["left"], poses_dict["right"])
        self.robot.move_to_home(arm=arm)

        # Both home at this point.
        arm_pixel_pose = project_points_to_image(
            np.array([last_arm_pose.translation]),
            self.zed_cam.intrinsic._K,
            self.T_CAM_BASE[arm],
            self.workspace_img.shape[:2],
        )[0]

        s_arm_pixel_pose = project_points_to_image(
            np.array([last_s_pose.translation]),
            self.zed_cam.intrinsic._K,
            self.T_CAM_BASE[s_arm],
            self.workspace_img.shape[:2],
        )[0]

        self.perform_nearest_analytic_grasp_dual(
            prev_grasp_point=s_arm_pixel_pose,
            prev_follow_point=arm_pixel_pose,
            grasp_arm="left" if arm == "left" else "right",
            follow_arm="right" if arm == "left" else "left",
            visualize=False,
        )
        return s_arm

    def regrasp(self, arm, direction):
        """
        reorients gripper by going to the exact same position,
        but just with a 180 degree rotation in the ee
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

    def perform_nearest_analytic_grasp_dual(
        self,
        prev_grasp_point,
        prev_follow_point,
        grasp_arm,
        follow_arm,
        visualize=False,
    ):
        # grasp arm
        world_coord_follow, world_coord_grasp = None, None
        cable_ori_follow, cable_ori_grasp = None, None
        move_to_pixel_grasp = None

        if prev_grasp_point is not None:
            move_to_pixel_grasp, cable_ori_grasp = (
                self.get_nearest_analytic_grasp_point(
                    prev_grasp_point, visualize=visualize
                )
            )
            cable_ori_grasp = np.deg2rad(-cable_ori_grasp + 90)
            assert print("should be here")
            world_coord_grasp = get_world_coord_from_pixel_coord(
                move_to_pixel_grasp,
                self.zed_cam.intrinsic,
                self.T_CAM_BASE[grasp_arm],
            )

        # follow_arm
        if prev_follow_point is not None:
            move_to_pixel_follow, cable_ori_follow = (
                self.get_nearest_analytic_grasp_point(
                    prev_follow_point, visualize=visualize
                )
            )
            cable_ori_follow = np.deg2rad(-cable_ori_follow + 90)

            world_coord_follow = get_world_coord_from_pixel_coord(
                move_to_pixel_follow,
                self.zed_cam.intrinsic,
                self.T_CAM_BASE[follow_arm],
            )

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

        return move_to_pixel_grasp, world_coord_grasp

    #################################################################
    # TODO: Refactor - split here into a separate tracing class
    #################################################################

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
        p1, p2 = self.board.point1, self.board.point2  # (582, 5), (1391, 767)

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

        cprint(f"Starting trace at start point: {start_points}", "red")
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

        cprint(f"Tracing status: {status}", "red")
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
        """
        Finds the nearest grasp point on the cable to a given position (start_point) using the analytic
        color thresholding method

        Arguments:
        start_point ((int, int)): coordinates of the point to find a nearby grasp to, in pixel space
        img (Image): Image to use for grasp identification, None by default
        visualize (bool): indicates whether or not to visualize the planned grasps
        """
        if img is None:
            img = self.zed_cam.rgb_image.copy()
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = img.copy()

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

            if is_cable_pixel(gray_img, pos):
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
                img_display, start_point[0], start_point[1], "Previous", (0, 0, 255), 0
            )
            draw_point(
                img_display,
                predicted_x,
                predicted_y,
                "Planned",
                (255, 0, 0),
                predicted_orientation,
            )

            cv2.imshow("Board Setup", img_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return (predicted_x, predicted_y), predicted_orientation
