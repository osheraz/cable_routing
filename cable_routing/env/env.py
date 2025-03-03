import rospy
import numpy as np
import cv2
from tqdm import tqdm
from cable_routing.env.robots.yumi import YuMiRobotEnv
import tyro
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import math
from autolab_core import RigidTransform, Point, CameraIntrinsics
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber
from cable_routing.handloom.handloom_pipeline.single_tracer import CableTracer
from cable_routing.env.board.new_board import Board
from cable_routing.env.ext_camera.utils.img_utils import (
    crop_img,
    crop_board,
    select_target_point,
    get_world_coord_from_pixel_coord,
    pick_target_on_path,
    find_nearest_point,
    get_perpendicular_ori,
    get_path_angle,
)  # TODO: Split to env_utils, img_utils etc..
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

        self.robot = YuMiRobotEnv(exp_config.robot_cfg)
        rospy.sleep(2)

        # todo: add cfg support for this
        self.zed_cam = ZedCameraSubscriber()
        while self.zed_cam.rgb_image is None or self.zed_cam.depth_image is None:
            rospy.sleep(0.1)
            rospy.loginfo("Waiting for images from ZED camera...")

        self.T_CAM_BASE = RigidTransform.load(
            "/home/osheraz/cable_routing/data/zed/zed_to_world.tf"
        ).as_frames(from_frame="zed", to_frame="base_link")

        self.tracer = CableTracer()

        ######################################################
        # self.set_board_region()
        # TODO: find a better way..
        # TODO: add support for this in cfg
        # TODO: modify to Jaimyn code
        # TODO: move all plots to vis_utils
        ######################################################

        self.board = Board(
            config_path="/home/osheraz/cable_routing/data/board_config.json"
        )
        rospy.logwarn("Env is ready")

        self.cable_in_arm = None

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
                self.robot.single_hand_grasp(
                    world_coord, eef_rot=np.deg2rad(clip_ori), slow_mode=True
                )
                self.robot.move_to_home()

                # abort = input("Abort? (y/n): ") == "y"

    def update_cable_path(self, start_points=None, end_points=None):

        path, _ = self.trace_cable(start_points=start_points, end_points=end_points)

        self.board.set_cable_path(path)

        return path

    def convert_path_to_world_coord(self, path):

        world_path = []
        for pixel_coord in path:
            world_coord = get_world_coord_from_pixel_coord(
                pixel_coord, self.zed_cam.intrinsic, self.T_CAM_BASE
            )
            world_path.append(world_coord)

        return world_path

    def goto_cable_node(self, path, single_hand=True, display=True):

        frame = self.zed_cam.get_rgb()

        move_to_pixel = pick_target_on_path(frame, path)
        move_to_pixel, idx = find_nearest_point(path, move_to_pixel)

        world_coord = get_world_coord_from_pixel_coord(
            move_to_pixel,
            self.zed_cam.intrinsic,
            self.T_CAM_BASE,
            depth_map=self.zed_cam.get_depth(),
        )

        print("Going to ", world_coord)

        path_in_world = self.convert_path_to_world_coord(path)

        cable_ori = get_perpendicular_ori(
            path_in_world[idx - 1], path_in_world[idx + 1]
        )
        path_arr = np.array(path_in_world)

        if display:
            plt.figure()
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

            arm = "right" if world_coord[1] < 0 else "left"

            self.robot.single_hand_grasp(
                arm, world_coord, eef_rot=cable_ori + np.pi / 2, slow_mode=True
            )

            # should update upon success
            self.cable_in_arm = arm

    def goto_clip_node(self, side="up", single_hand=True, display=True):

        frame = self.zed_cam.get_rgb()

        move_to_pixel = select_target_point(frame, rule="clip")

        clip = self.board.find_nearest_clip([move_to_pixel])

        clip_ori = clip["orientation"]
        move_to_pixel = [clip["x"], clip["y"]]

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

        self.robot.single_hand_move(self.cable_in_arm, target_coord, slow_mode=True)

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
        save_folder = f"/home/osheraz/cable_routing/trace_test/run_{timestamp}"
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
        # TODO: fix bugs
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
        save_folder = f"/home/osheraz/cable_routing/trace_test/run_{timestamp}"
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
