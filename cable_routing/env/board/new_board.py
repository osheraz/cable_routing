import cv2
import numpy as np
import json
import os
import random

from cable_routing.env.board.cable import Cable
from cable_routing.algo.levenshtein import suggest_modifications
from cable_routing.configs.envconfig import ExperimentConfig


def pretty_matrix(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = "\t".join("{{:{}}}".format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print("\n" + "\n".join(table) + "\n")


class Board:
    def __init__(
        self,
        config_path="/home/osheraz/cable_routing/cable_routing/configs/board/board_config.json",
        width=1500,
        height=800,
        grid_size=(100, 100),
    ):
        """
        Initialize a default instance of our board environment.

                Parameter(s):
                        config_path : Path to the config JSON for our board
                        width (int): The width in our board in pixels
                        height (int): The height in tiles of our board in pixels
                        grid_size((int, int)): The size and shape of our grid cells, specified as (width, height)

                Returns:
                        None
        """

        self.config_path = config_path
        self.true_width = width  # Stores the width in pixels
        self.true_height = height  # Stores the height in pixels

        # Ceiling division implementation, per Stack Overflow
        self.width = -(width // -grid_size[0])
        self.height = -(height // -grid_size[1])
        self.grid_size = grid_size

        self.cables = {}
        self.board = [["." for _ in range(self.width)] for _ in range(self.height)]

        self.clip_positions = self.load_board_config()
        self.point1, self.point2 = (582, 5), (1391, 767)  # ROI
        self.cable_positions = []
        self.key_locations = set()
        self.key_features = {}

        clip_types = {1: "6Pin", 2: "2Pin", 3: "Clip", 4: "Retainer"}

        for clip in self.load_board_config():
            self.add_keypoint(
                BoardFeature(
                    clip["x"],
                    clip["y"],
                    clip_types[clip["type"]],
                    clip["orientation"],
                    grid_size=grid_size,
                )
            )

    def load_board_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def find_nearest_clip(self, path):

        # TODO: SOMETHING IS MODIFYING GET CLIPS
        clips = self.load_board_config()  # self.get_clips()

        # for clip in clips:
        #     clip["x"] -= self.point1[0]
        #     clip["y"] -= self.point1[1]

        last_point = np.array(path[-1])

        nearest_clip = min(
            clips,
            key=lambda clip: np.linalg.norm(
                np.array([clip["x"], clip["y"]]) - last_point
            ),
        )

        return nearest_clip

    def set_cable_path(self, cable_positions):
        """
        TODO: Deprecated, see add_cable
        """
        cable = Cable(
            coordinates=cable_positions,
            environment=self,
            grid_size=self.grid_size,
            id=np.random.randint(10),
        )
        self.add_cable(cable)
        self.cable_positions = cable_positions

    def add_cable(self, cable):
        """
        Adds a cable object to our environment representation.

                Parameter(s):
                        cable (cable): The cable to add

                Returns:
                        None
        """
        self.cables[cable.get_id()] = cable
        cable.update_keypoints(self)  # associate keylocation

        for coordinate in cable.get_quantized():
            if self.board[coordinate[1]][coordinate[0]] == ".":
                self.board[coordinate[1]][coordinate[0]] = str(cable.get_id())
            else:
                self.board[coordinate[1]][coordinate[0]] = (
                    self.board[coordinate[1]][coordinate[0]] + f", {cable.get_id()}"
                )

    def add_keypoint(self, feature):
        """
        Adds a feature of interest to our environment representation.
        This might include a bracket or a plug, for example

                Parameter(s):
                        feature (BoardFeature): The feature to object to be included on our board

                Returns:
                        None
        """

        self.key_locations.add(feature.get_position())
        self.key_features[feature.get_position()] = feature
        for cable in self.cables:
            self.cables[cable].update_keypoints(self)

    def visualize_board(self, img, quantized=True):

        img_display = img.copy()

        for key_location in self.key_locations:

            clip = self.key_features[key_location]

            self.draw_clip(
                img_display,
                clip.get_true_coordinate()[0],
                clip.get_true_coordinate()[1],
                clip.get_type(),
                clip.get_orientation(),
            )

        # for clip in self.clip_positions:
        #     self.draw_clip(
        #         img_display, clip["x"], clip["y"], clip["type"], clip["orientation"]
        #     )

        if quantized:
            for id in self.cables:
                cable = self.cables[id]
                cable_points = cable.get_quantized()

                color = (255, 0, 255) if id == 0 else (255, 255, 0)

                for i in range(len(cable_points) - 1):
                    p1 = tuple(
                        (
                            cable_points[i][0] * self.grid_size[0],
                            cable_points[i][1] * self.grid_size[1],
                        )
                    )
                    p2 = tuple(
                        (
                            cable_points[i + 1][0] * self.grid_size[0],
                            cable_points[i + 1][1] * self.grid_size[1],
                        )
                    )
                    cv2.line(img_display, p1, p2, color, 2)
        else:
            for id in self.cables:
                cable = self.cables[id]
                cable_points = cable.get_points()

                color = (255, 0, 0)

                for i in range(len(cable_points) - 1):
                    p1 = tuple(cable_points[i])
                    p2 = tuple(cable_points[i + 1])
                    cv2.line(img_display, p1, p2, color, 2)

        cv2.imshow("Board Setup with Clips & Cable", img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_clip(self, img, x, y, clip_type, orientation):

        center = (x, y)
        cv2.circle(img, center, 10, (0, 0, 255), -1)
        arrow_length = 30
        angle_rad = np.deg2rad(orientation)

        if clip_type == "Clip":
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
            clip_type,
            (x + 15, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    def get_clips(self):
        return self.clip_positions.copy()

    def get_cables(self):
        return self.cables.copy()

    def get_cable_path(self):
        """
        TODO: Deprecated
        """

        return self.cables[self.cables.keys()[0]].get_points()

    def return_board(self):
        """
        Returns a representation of the board as an occupancy map of cables
        """
        return self.board


class BoardFeature:
    """
    A class representing a key feature of the board environment, such as a plug,
    bracket, or spool, including relevant feature data.

    Can be subclassed to define more specific features with particular behavior.
    """

    def __init__(self, x, y, type="N/A", orientation=0, grid_size=None):
        """
        Instantiate an instance of our feature object.

        Parameter(s):
                x (int): The pixel x coordinate of our board feature
                y (int): The pixel y coordinate of our board feature
                type (str): A string indicating the type of feature that it is
                orientation (int?): A representation for the angle of our feature
                environment (Board:), optional: The environment that the feature is a part of,
                can be used to update the feature's coordinate scale

        Returns:
                None
        """
        self.true_coordinate = (x, y, orientation)
        self.type = type

        if grid_size is not None:
            self.coordinate = (
                int(round(x / grid_size[0])),
                int(round(y / grid_size[1])),
                orientation,
            )
        else:
            self.coordinate = self.true_coordinate

    # Getter functions
    def get_position(self):
        return (self.coordinate[0], self.coordinate[1])

    def get_true_coordinate(self):
        if self.type in ["Clip", "6Pin"]:
            return self.true_coordinate
        else:
            return (self.true_coordinate[0], self.true_coordinate[1])
    
    def get_type(self):
        return self.type

    def get_orientation(self):
        return self.coordinate[2]


if __name__ == "__main__":

    cfg = ExperimentConfig
    config_path = cfg.board_cfg_path
    img_path = cfg.bg_img_path

    board = Board(config_path=config_path, grid_size=(20,20))
    img = cv2.imread(img_path)

    goal_keypoints = [(657, 547), (825, 394), (886, 572), (1181, 240), (1309, 637)]
    cur_keypoints = [(657, 547), (827, 157), (974, 274), (1181, 240), (1313, 562)]

    goal_sequence = []
    num_steps = 10
    # Interpolate sequences:
    for k in range(len(goal_keypoints) - 1):
        goal_sequence.append(goal_keypoints[k])

        for l in range(1, num_steps+1):
            goal_sequence.append((goal_keypoints[k][0] + l*(goal_keypoints[k+1][0] - goal_keypoints[k][0])//num_steps, goal_keypoints[k][1] + l*(goal_keypoints[k+1][1] - goal_keypoints[k][1])//num_steps))

    goal_sequence.append(goal_keypoints[-1])

    cur_sequence = []
    # Interpolate sequences:
    for k in range(len(cur_keypoints) - 1):
        cur_sequence.append(cur_keypoints[k])
        for l in range(1, num_steps+1):
            cur_sequence.append((cur_keypoints[k][0] + l*(cur_keypoints[k+1][0] - cur_keypoints[k][0])//num_steps, cur_keypoints[k][1] + l*(cur_keypoints[k+1][1] - cur_keypoints[k][1])//num_steps))

    cur_sequence.append(cur_keypoints[-1])

    print(len(goal_sequence))

    goal_cable = Cable(
        coordinates=goal_sequence,
        environment=board,
        grid_size=board.grid_size,
        id=-1,
    )
    cur_cable = Cable(
        coordinates=cur_sequence,
        environment=board,
        grid_size=board.grid_size,
        id=0,
    )

    board.add_cable(goal_cable)
    board.add_cable(cur_cable)

    # cur_cable.get_keypoints()[0]
    print(cur_cable.intermediate_points(cur_cable.get_keypoints()[0], cur_cable.get_keypoints()[1], num_points=2))

    goal_config = {cur_cable.id: goal_cable}

    suggestion = suggest_modifications(
        board, goal_configuration=goal_config, human_readable=False
    )
    print(suggestion)

    annotated_img = board.visualize_board(img, quantized=False)
    #annotated_img = board.visualize_board(img, quantized=True)
