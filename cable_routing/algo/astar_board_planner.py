import numpy as np
import matplotlib.pyplot as plt
import cv2

from cable_routing.env.board.new_board import Board
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.algo.astar import AStarPlanner
from cable_routing.env.ext_camera.utils.img_utils import crop_img, select_target_point
import scipy.interpolate as si


class BoardPlanner:
    def __init__(
        self,
        config_path=None,
        resolution=20.0,
        robot_radius=20.0,
        inflation_radius=40.0,
        show_animation=False,
    ):
        self.show_animation = show_animation
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.inflation_radius = inflation_radius

        self.cfg = ExperimentConfig
        self.board = Board(config_path=config_path or self.cfg.board_cfg_path)

        # Detect board corners and crop the image
        self.p1 = self.board.point1
        self.p2 = self.board.point2
        self.full_img = cv2.imread(self.cfg.bg_img_path)
        self.img = crop_img(self.full_img.copy(), self.p1, self.p2)

        # Pre-load obstacles directly in pixel space (cropped coordinates)
        self.ox, self.oy = self._extract_obstacles()

    def _extract_obstacles(self):

        ox, oy = [], []
        clips = self.board.get_clips()
        print("printing clips")
        print(clips)

        for id, clip in clips.items():
            clip_x = clip["x"] - self.p1[0]
            clip_y = clip["y"] - self.p1[1]

            for angle in range(0, 360, 1):
                rad = np.deg2rad(angle)
                x = int(clip_x + self.inflation_radius * np.cos(rad))
                y = int(clip_y + self.inflation_radius * np.sin(rad))
                ox.append(x)
                oy.append(y)

            ox.append(clip_x)
            oy.append(clip_y)

        height, width = self.img.shape[:2]
        for x in range(width):
            ox.extend([x, x])
            oy.extend([0, height - 1])

        for y in range(height):
            ox.extend([0, width - 1])
            oy.extend([y, y])

        return ox, oy

    def _simplify_path(
        self,
        path_pixels,
        curvature_threshold=0.04,
        distance_threshold=1.0,
        max_step_threshold=50.0,
    ):
        if len(path_pixels) < 3:
            return path_pixels

        simplified_path = [path_pixels[0]]

        for i in range(1, len(path_pixels) - 1):
            prev_point = np.array(simplified_path[-1])
            curr_point = np.array(path_pixels[i])
            next_point = np.array(path_pixels[i + 1])

            vec1 = curr_point - prev_point
            vec2 = next_point - curr_point

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 < distance_threshold:
                continue

            cos_angle = np.dot(vec1, vec2) / ((norm1 + 1e-6) * (norm2 + 1e-6))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

            curvature = angle / norm1

            if (
                curvature > curvature_threshold
                or np.linalg.norm(curr_point - prev_point) > max_step_threshold
            ):
                simplified_path.append(path_pixels[i])

        simplified_path.append(path_pixels[-1])

        return np.array(simplified_path)

    def plan_path(
        self, start_pixel, goal_pixel, full_res=True, use_spline=True, simplify=True
    ):
        if full_res:
            sx = start_pixel[0] - self.p1[0]
            sy = start_pixel[1] - self.p1[1]
            gx = goal_pixel[0] - self.p1[0]
            gy = goal_pixel[1] - self.p1[1]
        else:
            sx, sy = start_pixel
            gx, gy = goal_pixel

        a_star = AStarPlanner(self.ox, self.oy, self.resolution, self.robot_radius)
        rx, ry = a_star.planning(sx, sy, gx, gy)
        path_pixels = np.array([(int(x), int(y)) for x, y in zip(rx, ry)])

        if use_spline and len(path_pixels) > 3:
            tck, _ = si.splprep([path_pixels[:, 0], path_pixels[:, 1]], s=5.0)
            u_fine = np.linspace(0, 1, len(path_pixels) * 2)
            x_smooth, y_smooth = si.splev(u_fine, tck)
            path_pixels = np.array(
                [(int(x), int(y)) for x, y in zip(x_smooth, y_smooth)]
            )

        if simplify:
            path_pixels = self._simplify_path(path_pixels)

        if full_res:
            path_pixels = [[p[0] + self.p1[0], p[1] + self.p1[1]] for p in path_pixels]

        return np.array(path_pixels)[::-1]


def draw_planner_overlay(
    img, planner, start_pixel, goal_pixel, path_in_pixels, resolution=20
):
    # print(f"Planned Path (pixel coordinates): {path_in_pixels}")

    overlay = img.copy()

    x1, y1 = planner.p1
    x2, y2 = planner.p2

    for x in range(x1, x2, int(resolution)):
        cv2.line(overlay, (x, y1), (x, y2), (200, 200, 200), 1)

    for y in range(y1, y2, int(resolution)):
        cv2.line(overlay, (x1, y), (x2, y), (200, 200, 200), 1)

    for ox, oy in zip(planner.ox, planner.oy):
        ox += planner.p1[0]
        oy += planner.p1[1]
        cv2.circle(overlay, (ox, oy), 3, (255, 255, 255), -1)

    cv2.circle(overlay, tuple(start_pixel), 10, (255, 255, 255), -1)
    cv2.circle(overlay, tuple(goal_pixel), 10, (0, 255, 255), -1)

    cv2.putText(
        overlay,
        f"S {start_pixel}",
        (start_pixel[0] + 15, start_pixel[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"G {goal_pixel}",
        (goal_pixel[0] + 15, goal_pixel[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for i in range(1, len(path_in_pixels)):
        pt1 = tuple(map(int, path_in_pixels[i - 1]))
        pt2 = tuple(map(int, path_in_pixels[i]))
        cv2.line(overlay, pt1, pt2, (0, 255, 0), 2)

    for point in path_in_pixels:
        px, py = tuple(map(int, point))
        cv2.circle(overlay, (px, py), 5, (0, 0, 255), -1)  # Mark points in red

    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.imshow("Path in Image Space", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():

    planner = BoardPlanner(show_animation=False)
    img_copy = planner.full_img.copy()

    ########################################
    # comment if camera is not available
    # import rospy
    # from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber

    # rospy.init_node("testtest")
    # zed_cam = ZedCameraSubscriber()
    # while zed_cam.rgb_image is None or zed_cam.depth_image is None:
    #     rospy.sleep(0.1)
    #     rospy.loginfo("Waiting for images from ZED camera...")
    # frame = zed_cam.get_rgb()

    start_pixel = select_target_point(img_copy, rule="start")
    goal_pixel = select_target_point(img_copy, rule="end")
    #########################################

    path_in_pixels = planner.plan_path(start_pixel, goal_pixel)

    draw_planner_overlay(
        img_copy,
        planner,
        start_pixel,
        goal_pixel,
        path_in_pixels,
        planner.resolution,
    )


if __name__ == "__main__":
    main()
