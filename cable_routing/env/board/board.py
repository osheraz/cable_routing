import cv2
import numpy as np
import json
import os


class Board:
    def __init__(self, config_path):

        self.config_path = config_path
        self.clip_positions = self.load_board_config()
        self.cable_positions = []
        self.clip_types = {1: "6Pin", 2: "2Pin", 3: "Clip", 4: "Retainer"}
        self.point1, self.point2 = (582, 5), (1391, 767)

    def load_board_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def find_nearest_clip(self, path):

        clips = self.get_clips()

        last_point = np.array(path[-1])

        nearest_clip = min(
            clips,
            key=lambda clip: np.linalg.norm(
                np.array([clip["x"], clip["y"]]) - last_point
            ),
        )

        return nearest_clip

    def set_cable_path(self, cable_positions):
        self.cable_positions = cable_positions

    def visualize_board(self, img):

        img_display = img.copy()
        for clip in self.clip_positions:
            self.draw_clip(
                img_display, clip["x"], clip["y"], clip["type"], clip["orientation"]
            )

        if self.cable_positions:
            for i in range(len(self.cable_positions) - 1):
                p1 = tuple(self.cable_positions[i])
                p2 = tuple(self.cable_positions[i + 1])
                cv2.line(img_display, p1, p2, (0, 255, 255), 2)

        cv2.imshow("Board Setup with Clips & Cable", img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_clip(self, img, x, y, clip_type, orientation):
        center = (x, y)
        cv2.circle(img, center, 10, (0, 0, 255), -1)
        arrow_length = 30
        angle_rad = np.deg2rad(orientation)

        if clip_type == 3:
            arrow_start = center
            arrow_end = (
                int(center[0] + arrow_length * np.cos(angle_rad)),
                int(center[1] + arrow_length * np.sin(angle_rad)),
            )
        else:
            arrow_start = (
                int(center[0] - (arrow_length / 2) * np.cos(angle_rad)),
                int(center[1] - (arrow_length / 2) * np.sin(angle_rad)),
            )
            arrow_end = (
                int(center[0] + (arrow_length / 2) * np.cos(angle_rad)),
                int(center[1] + (arrow_length / 2) * np.sin(angle_rad)),
            )

        cv2.arrowedLine(img, arrow_start, arrow_end, (255, 0, 0), 2)
        cv2.putText(
            img,
            self.clip_types[clip_type],
            (x + 15, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    def get_clips(self):
        return self.clip_positions

    def get_cable_path(self):
        return self.cable_positions


if __name__ == "__main__":

    board = Board(config_path="/home/osheraz/cable_routing/data/board_config.json")

    img = cv2.imread("/home/osheraz/cable_routing/data/board_setup.png")

    cable_path = [(100, 200), (150, 250), (200, 300), (200, 300)]
    board.set_cable_path(cable_path)

    annotated_img = board.visualize_board(img)
