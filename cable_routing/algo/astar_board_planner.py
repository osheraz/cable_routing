import numpy as np
import matplotlib.pyplot as plt
import cv2

from cable_routing.env.board.new_board import Board
from cable_routing.configs.envconfig import ExperimentConfig
from cable_routing.algo.astar import AStarPlanner
from cable_routing.env.ext_camera.utils.img_utils import crop_img


class BoardPlanner:
    def __init__(self, config_path=None, show_animation=True):
        self.show_animation = show_animation

        self.cfg = ExperimentConfig
        self.board = Board(config_path=config_path or self.cfg.board_cfg_path)

        # Detect board corners and crop the image
        self.p1 = self.board.point1
        self.p2 = self.board.point2
        self.img = crop_img(cv2.imread(self.cfg.bg_img_path), self.p1, self.p2)

        # Pre-load obstacles directly in pixel space (cropped coordinates)
        self.ox, self.oy = self._extract_obstacles()

    def _extract_obstacles(self):
        ox, oy = [], []
        clips = self.board.get_clips()

        for clip in clips:
            # Convert to cropped space (offset from p1)
            clip_x = clip["x"] - self.p1[0]
            clip_y = clip["y"] - self.p1[1]
            ox.append(clip_x)
            oy.append(clip_y)

        # Add outer boundary of cropped image (board edges)
        height, width = self.img.shape[:2]
        for x in range(width):
            ox.extend([x, x])
            oy.extend([0, height - 1])

        for y in range(height):
            ox.extend([0, width - 1])
            oy.extend([y, y])

        return ox, oy

    def plan_path(self, start_pixel, goal_pixel, robot_radius=10):
        """
        Plan path directly in pixel space (cropped image coordinates).
        """
        sx, sy = start_pixel
        gx, gy = goal_pixel

        # A* directly in pixel space
        resolution = 30.0  # 1 pixel = 1 grid cell
        a_star = AStarPlanner(self.ox, self.oy, resolution, robot_radius)

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

        return path_pixels


def main():
    planner = BoardPlanner(show_animation=True)

    # Example start & goal directly in cropped pixel space
    start_pixel = [100, 500]  # Coordinates within cropped board image
    goal_pixel = [500, 100]

    path_in_pixels = planner.plan_path(start_pixel, goal_pixel)

    img_copy = planner.img.copy()

    # === OpenCV plot (image overlay) ===
    for ox, oy in zip(planner.ox, planner.oy):
        cv2.circle(img_copy, (ox, oy), 3, (255, 255, 255), -1)

    cv2.circle(img_copy, tuple(start_pixel), 8, (0, 255, 0), -1)  # Start - green
    cv2.circle(img_copy, tuple(goal_pixel), 8, (255, 0, 0), -1)  # Goal - blue

    for i in range(1, len(path_in_pixels)):
        cv2.line(img_copy, path_in_pixels[i - 1], path_in_pixels[i], (0, 0, 255), 2)

    cv2.imshow("Path in Image Space", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Planned Path (pixel coordinates): {path_in_pixels}")


if __name__ == "__main__":
    main()
