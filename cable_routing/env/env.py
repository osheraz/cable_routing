from math import e
import rospy
import numpy as np
import cv2
from sympy import im
from tqdm import tqdm
from cable_routing.env.robots.yumi import YuMiRobotEnv
import tyro
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
from scipy.spatial import KDTree
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
)  # TODO: Split to env_utils, img_utils etc..
from cable_routing.env.ext_camera.utils.pcl_utils import project_points_to_image
from cable_routing.handloom.handloom_pipeline.tracer import (
    TraceEnd,
)


class ExperimentEnv:
    """Superclass for all Robots environments."""

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

        self.T_CAM_BASE = RigidTransform.load(
            exp_config.cam_to_robot_trans_path
        ).as_frames(from_frame="zed", to_frame="base_link")

        self.tracer = CableTracer()
        self.planner = BoardPlanner(show_animation=False)

        self.board = Board(config_path=exp_config.board_cfg_path)
        rospy.logwarn("Env is ready")

        self.cable_in_arm = None

        self.workspace_img = self.zed_cam.get_rgb().copy()

        ######################################################
        # self.set_board_region()
        # TODO: find a better way..
        # TODO: add support for this in cfg
        # TODO: modify to Jaimyn code
        # TODO: move all plots to vis_utils
        ######################################################

    def get_extrinsic(self):

        return self.T_CAM_BASE

    def set_board_region(self, img=None):

        if img is None:
            img = self.zed_cam.rgb_image

        _, self.point1, self.point2 = crop_board(img)

        # print(self.point1, self.point2)

    def check_calibration(self):
        """we are going to poke all the clips/plugs etc"""

        # TODO - above board
        self.robot.move_to_home()

        self.board.visualize_board(self.zed_cam.rgb_image)

        clips = self.board.get_clips()

        abort = False
        for clip in clips:

            if not abort:
                clip_type = clip["type"]
                clip_ori = clip["orientation"]
                pixel_coord = (clip["x"], clip["y"])
                world_coord = get_world_coord_from_pixel_coord(
                    pixel_coord, self.zed_cam.intrinsic, self.T_CAM_BASE, is_clip=True
                )

                print(
                    f"Poking clip {clip_type}, orientation: {clip_ori} at: {world_coord}"
                )

                # TODO add support for ori
                arm = "right" if world_coord[1] < 0 else "left"
                self.robot.single_hand_grasp(
                    arm, world_coord, eef_rot=np.deg2rad(clip_ori), slow_mode=True
                )
                self.robot.move_to_home()
                #                 arm, world_coord, eef_rot=cable_ori, slow_mode=True

                # abort = input("Abort? (y/n): ") == "y"

    def update_cable_path(self, start_points=None, end_points=None, display=False):

        path_in_pixels, _ = self.trace_cable(
            start_points=start_points, end_points=end_points, viz=display
        )

        self.board.set_cable_path(path_in_pixels)

        path_in_world = self.convert_path_to_world_coord(path_in_pixels)

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

    def convert_path_to_world_coord(self, path):

        world_path = []
        for pixel_coord in path:
            world_coord = get_world_coord_from_pixel_coord(
                pixel_coord, self.zed_cam.intrinsic, self.T_CAM_BASE
            )
            world_path.append(world_coord)

        return world_path

    def grasp_cable_node(
        self, path, cable_orientations, arm=None, single_hand=True, display=False
    ):

        frame = self.zed_cam.get_rgb()

        move_to_pixel = pick_target_on_path(frame, path)
        move_to_pixel, idx = find_nearest_point(path, move_to_pixel)

        world_coord = get_world_coord_from_pixel_coord(
            move_to_pixel,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE,
            # depth_map=self.zed_cam.get_depth(),
        )

        print("Going to ", world_coord)

        path_in_world = self.convert_path_to_world_coord(path)

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

        if single_hand:

            if arm is None:
                arm = "right" if world_coord[1] < 0 else "left"

            self.robot.single_hand_grasp(
                arm, world_coord, eef_rot=cable_ori, slow_mode=True
            )

            # should update upon success
            self.cable_in_arm = arm

        return move_to_pixel, world_coord, idx

    def release_cable_node(
        self, path, cable_orientations, arm, single_hand=True, display=False
    ):

        frame = self.zed_cam.get_rgb()

        move_to_pixel = pick_target_on_path(frame, path)
        clip = self.board.find_nearest_clip([move_to_pixel])
        move_to_pixel = [clip["x"], clip["y"]]

        # TODO - should pick a safe grasp point automatically.
        move_to_pixel, idx = find_nearest_point(path, move_to_pixel)

        world_coord = get_world_coord_from_pixel_coord(
            move_to_pixel,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE,
            depth_map=self.zed_cam.get_depth(),
        )

        print("Going to ", world_coord)

        path_in_world = self.convert_path_to_world_coord(path)

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

        if single_hand:

            self.robot.single_hand_grasp(
                arm, world_coord, eef_rot=cable_ori, slow_mode=True
            )

            # should update upon success
            self.robot.go_delta(
                left_delta=[0, 0, 0.05] if arm == "left" else None,
                right_delta=[0, 0, 0.05] if arm == "right" else None,
            )

            self.robot.open_grippers(arm)

        return move_to_pixel, world_coord, idx

    def goto_clip_node(self, arm, side="up", single_hand=True, display=True):

        frame = self.zed_cam.get_rgb()

        move_to_pixel = select_target_point(frame, rule="clip")

        # clip = self.board.find_nearest_clip([move_to_pixel])
        # clip_ori = clip["orientation"]
        # move_to_pixel = [clip["x"], clip["y"]]

        world_coord = get_world_coord_from_pixel_coord(
            move_to_pixel,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE,
            # depth_map=self.zed_cam.get_depth(),
            # is_clip=True,
        )

        distance = 0.02
        offset = {
            "left": [0, distance, 0],
            "right": [0, -distance, 0],
            "down": [-distance, 0, 0],
            "up": [distance, 0, 0],
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

    def plan_slide_to_cable_node(
        self, path_in_pixel, idx, arm=None, side="up", display=True
    ):

        move_to_pixel = select_target_point(self.workspace_img, rule="clip")
        clip = self.board.find_nearest_clip([move_to_pixel])
        clip_pixel = [clip["x"], clip["y"]]

        world_coord = get_world_coord_from_pixel_coord(
            clip_pixel,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE,
            is_clip=True,
        )

        pixel_offset = {
            "left": [-60, 0],  # must be higher than inflation_radius
            "right": [60, 0],
            "down": [0, 60],
            "up": [0, -60],
        }[side]

        goal_pixel = np.array(clip_pixel) + np.array(pixel_offset)

        target_coord = get_world_coord_from_pixel_coord(
            goal_pixel,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE,
            is_clip=True,
        )

        start_pixel = path_in_pixel[idx]
        current_pose = self.robot.get_ee_pose()[0 if arm == "left" else 1]
        #
        start_pixel = project_points_to_image(
            np.array([current_pose.translation]),
            self.zed_cam.intrinsic._K,
            self.T_CAM_BASE,
            self.workspace_img.shape[:2],
        )[0]
        # start_pixel = select_target_point(self.workspace_img, rule="START START")
        # start_pixel[0] -= self.point1[0]
        # start_pixel[1] -= self.point1[1]

        waypoints_in_pixels = self.planner.plan_path(
            start_pixel=start_pixel,  # start_pixel,
            goal_pixel=goal_pixel,  #  move_to_pixel - directly to the point
        )
        waypoints = self.convert_path_to_world_coord(waypoints_in_pixels)

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
                self.convert_path_to_world_coord(self.board.cable_positions)
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
        self, waypoints, target_coord, cable_orientations, arm=None, display=True
    ):
        if arm is None:
            arm = self.cable_in_arm

        current_pose = self.robot.get_ee_pose()[0 if arm == "left" else 1]
        target_coord[2] = current_pose.translation[2]  # Preserve current Z height

        # Open gripper slightly -- fix!
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
                rotation=RigidTransform.x_axis_rotation(np.pi)
                @ RigidTransform.z_axis_rotation(-eef_orientations[0]),
            ),
        ]
        self.robot.plan_and_execute_linear_waypoints(arm, waypoints=poses)

        # Follow the full path
        poses = [
            RigidTransform(
                translation=wp,
                rotation=RigidTransform.x_axis_rotation(np.pi)
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

    def slideto_cable_node(
        self,
        cable_path_in_pixel,
        path_in_world,
        cable_orientations,
        idx,
        arm=None,
        side="up",
        single_hand=True,
        display=False,
    ):
        plan = self.plan_slide_to_cable_node(
            cable_path_in_pixel, idx, arm=arm, side=side, display=display
        )
        self.execute_slide_to_cable_node(
            plan["waypoints"], plan["target_coord"], cable_orientations, arm=arm
        )

    def trace_cable(self, img=None, start_points=None, end_points=None, viz=False):

        # TODO: clean implementation
        p1, p2 = self.board.point1, self.board.point2

        clips = self.board.get_clips().copy()
        for clip in clips:
            clip["x"] -= p1[0]
            clip["y"] -= p1[1]

        if img is None:
            img = self.zed_cam.rgb_image

        img = crop_img(img, p1, p2)  # TODO: fix!

        if start_points == None:
            start_points = select_target_point(img, rule="start")

        if end_points == None:
            end_points = select_target_point(img, rule="end")

        # TODO: fix!
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_folder = f"{self.exp_config.save_folder}/run_{timestamp}"
        if viz:
            os.makedirs(save_folder, exist_ok=True)

        print(f"Starting trace at start point: {start_points}")
        connection = 0

        path, status = self.tracer.trace(
            img=img.copy(),
            start_points=start_points,
            end_points=end_points,
            clips=clips,
            save_folder=save_folder,
            idx=connection,
            viz=viz,
        )

        print("Tracing status:", status)

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

    def trace_cable_from_clips(
        self, img=None, start_points=None, end_points=None, viz=True
    ):
        """Traces a cable path from clips by finding valid start points next to them."""
        # TODO: fix bugs NOT WORKING
        p1, p2 = self.board.point1, self.board.point2
        clips = self.board.get_clips()

        for clip in clips:
            clip["x"] -= p1[0]
            clip["y"] -= p1[1]

        if img is None:
            img = self.zed_cam.rgb_image

        img = crop_img(img, p1, p2)

        if end_points == None:
            end_points = select_target_point(img, rule="end")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_folder = f"{self.exp_config.save_folder}/run_{timestamp}"
        os.makedirs(save_folder, exist_ok=True)

        def mask_clip_region(image, clip, mask_size=30):
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            x, y = clip["x"], clip["y"]
            angle = np.deg2rad(clip["orientation"])

            normal_x, normal_y = np.cos(angle), np.sin(angle)

            for i in range(-mask_size, mask_size):
                for j in range(-mask_size, mask_size):
                    px, py = x + i, y + j
                    if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                        relative_pos = (px - x) * normal_x + (py - y) * normal_y
                        if relative_pos > 0:
                            mask[py, px] = 0

            return mask

        def center_pixels_on_cable(image, pixels, display=True):
            # for each pixel, find closest pixel on cable
            image_mask = image[:, :, 0] > 100
            # erode white pixels
            kernel = np.ones((2, 2), np.uint8)
            image_mask = cv2.erode(image_mask.astype(np.uint8), kernel, iterations=1)
            white_pixels = np.argwhere(image_mask)

            # # visualize this
            if display:
                pixels = np.atleast_2d(pixels)
                plt.imshow(image_mask)
                for pixel in pixels:
                    plt.scatter(*pixel[::-1], c="r")
                plt.show()

            processed_pixels = []
            for pixel in pixels:
                # find closest pixel on cable
                distances = np.linalg.norm(white_pixels - pixel, axis=1)
                closest_pixel = white_pixels[np.argmin(distances)]
                processed_pixels.append([closest_pixel])
            return np.array(processed_pixels)

        def find_nearest_white_pixel(image, clip):
            if len(image.shape) == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                image_gray = image

            clip_pixel = np.array([[clip["y"], clip["x"]]])

            mask = mask_clip_region(image_gray, clip)

            masked_image = cv2.bitwise_and(image_gray, image_gray, mask=mask)

            centered_pixels = center_pixels_on_cable(
                masked_image[..., None], clip_pixel, display=False
            )

            nearest_pixel = centered_pixels[0][0]

            vis = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

            cv2.circle(vis, (clip["x"], clip["y"]), 10, (0, 0, 255), -1)
            cv2.circle(vis, (nearest_pixel[1], nearest_pixel[0]), 5, (0, 255, 0), -1)

            mask_color = np.zeros_like(vis)
            mask_color[mask == 0] = [255, 0, 0]

            vis = cv2.addWeighted(vis, 1.0, mask_color, 0.5, 0)

            cv2.imshow("Image with Mask and Nearest Cable Pixel", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return (nearest_pixel[1], nearest_pixel[0])

        paths = []

        for clip in clips:

            nearest_white = find_nearest_white_pixel(img, clip)

            if nearest_white is None:
                print(
                    f"Warning: No valid start point found near clip at {clip['x'], clip['y']}"
                )
                continue

            start_x, start_y = nearest_white
            print(f"Starting trace near clip at: {start_x, start_y}")

            path, status = self.tracer.trace(
                img=img.copy(),
                start_points=[[start_y, start_x]],
                end_points=end_points,
                clips=clips,
                save_folder=save_folder,
            )

            paths.append(path)

            if viz:
                img_display = img.copy()
                cv2.polylines(
                    img_display,
                    [np.array(path, dtype=np.int32)],
                    isClosed=False,
                    color=(0, 255, 0),
                    thickness=2,
                )

                cv2.circle(img_display, (start_x, start_y), 10, (0, 0, 255), -1)
                cv2.putText(
                    img_display,
                    "Start",
                    (start_x + 15, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

                cv2.circle(img_display, (clip["x"], clip["y"]), 10, (255, 255, 255), -1)

                plt.imsave(
                    f"{save_folder}/clip_start_{clip['x']}_{clip['y']}.png", img_display
                )
                cv2.imshow("Tracing Start Point", img_display)
                cv2.waitKey(500)
                cv2.destroyAllWindows()

        # Merge all paths
        final_path = []
        for path in paths:
            final_path.extend(path)

        # Convert path coordinates back to the original image frame
        final_path = [(x + p1[0], y + p1[1]) for x, y in final_path]

        return final_path, status
