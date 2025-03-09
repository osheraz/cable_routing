import numpy as np
import matplotlib.pyplot as plt
import cv2

from cable_routing.env.board.new_board import Board
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.algo.astar import AStarPlanner
from cable_routing.env.ext_camera.utils.img_utils import crop_img, select_target_point


class BoardPlanner:
    def __init__(
        self,
        config_path=None,
        resolution=30.0,
        robot_radius=30.0,
        inflation_radius=30.0,
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

    def plan_path(self, start_pixel, goal_pixel, full_res=True):
        """
        Plan path directly in pixel space (cropped image coordinates).
        """
        if full_res:
            sx = start_pixel[0] - self.p1[0]
            sy = start_pixel[1] - self.p1[1]
            gx = goal_pixel[0] - self.p1[0]
            gy = goal_pixel[1] - self.p1[1]
        else:
            sx, sy = start_pixel
            gx, gy = goal_pixel

        # A* directly in pixel space
        resolution = self.resolution

        a_star = AStarPlanner(self.ox, self.oy, resolution, self.robot_radius)

        rx, ry = a_star.planning(sx, sy, gx, gy)
        path_pixels = [(int(x), int(y)) for x, y in zip(rx, ry)]

        if self.show_animation:
            # === Matplotlib plot (planner view) ===
            plt.plot(self.ox, self.oy, ".k", label="Obstacles")
            plt.plot(sx, sy, "og", label="Start (Pixels)")
            plt.plot(gx, gy, "xb", label="Goal (Pixels)")
            plt.plot(rx, ry, "-r", label="Path (Pixels)")
            plt.grid(True)
            plt.axis("equal")
            plt.gca().invert_yaxis()  # Because image Y=0 is top, matplotlib Y=0 is bottom
            plt.legend()
            plt.show()

        if full_res:
            path_pixels = [[p[0] + self.p1[0], p[1] + self.p1[1]] for p in path_pixels]
        return np.array(path_pixels)[::-1]


def draw_planner_overlay(
    img, planner, start_pixel, goal_pixel, path_in_pixels, resolution=20
):
    print(f"Planned Path (pixel coordinates): {path_in_pixels}")

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
        cv2.line(overlay, path_in_pixels[i - 1], path_in_pixels[i], (0, 255, 0), 2)

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
