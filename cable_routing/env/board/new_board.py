import cv2
import numpy as np
import json
import os

# A pretty print for a matrix, thanks StackOverflow!
def pretty_matrix(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = "\t".join("{{:{}}}".format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print("\n" + "\n".join(table) + "\n")


class NewBoard:
    def __init__(self, config_path, width=1500, height=800, grid_size = (100,100)):
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
        self.true_width = width # Stores the width in pixels
        self.true_height = height # Stores the height in pixels

        self.width = -(width // -grid_size[0]) # Ceiling division implementation, per Stack Overflow
        self.height = -(height // -grid_size[1]) # Store the height in grid spaces
        self.grid_size = grid_size

        self.cables = {}
        self.board = [["." for i in range(self.width)] for j in range(self.height)]

        # self.clip_positions = self.load_board_config()
        # self.cable_positions = []
        self.point1, self.point2 = (582, 5), (1391, 767)

        self.key_locations = set()
        self.key_features = {}

        clip_types = {1: "6Pin", 2: "2Pin", 3: "Clip", 4: "Retainer"}
        for clip in self.load_board_config():
            self.add_keypoint(BoardFeature(clip["x"], clip["y"], clip_types[clip["type"]], clip["orientation"], self))


    def load_board_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def set_cable_path(self, cable_positions):
        '''
        TODO: Deprecated, see add_cable
        '''
        self.add_cable(Cable(cable_positions, environment=self))
        # self.cable_positions = cable_positions

    def add_cable(self, cable):
        """
        Adds a cable object to our environment representation.

                Parameter(s):
                        cable (cable): The cable to add

                Returns:
                        None
        """
        self.cables[cable.get_id()] = cable
        cable.update_keypoints(self)

        for coordinate in cable.get_quantized():
            if self.board[coordinate[1]][coordinate[0]] == ".":
                self.board[coordinate[1]][coordinate[0]] = str(cable.get_id())
            else:
                self.board[coordinate[1]][coordinate[0]] = (
                    self.board[coordinate[1]][coordinate[0]] + f", {cable.get_id()}"
                )

    def add_keypoint(self, feature):
        """
        Adds a feature of interest to our environment representation. This might include a bracket or a plug, for example

                Parameter(s):
                        feature (BoardFeature): The feature to object to be included on our board

                Returns:
                        None
        """

        self.key_locations.add(feature.get_position())
        self.key_features[feature.get_position()] = feature
        for cable in self.cables:
            self.cables[cable].update_keypoints(self)
    

    def visualize_board(self, img, quantized = False):

        img_display = img.copy()
        for key_location in self.key_locations:
            clip = self.key_features[key_location]
            self.draw_clip(
                img_display, clip.get_position()[0], clip.get_position()[1], clip.get_type(), clip.get_orientation()
            )

        # for clip in self.clip_positions:
        #     self.draw_clip(
        #         img_display, clip["x"], clip["y"], clip["type"], clip["orientation"]
        #     )

        if quantized:
            for id in self.cables:
                cable = self.cables[id]
                cable_points = cable.get_quantized()
                for i in range(len(cable_points)-1):
                    p1 = tuple((cable_points[i][0]*self.grid_size[0], cable_points[i][1]*self.grid_size[1]))
                    p2 = tuple((cable_points[i + 1][0]*self.grid_size[0], cable_points[i + 1][1]*self.grid_size[1]))
                    cv2.line(img_display, p1, p2, (0, 255, 255), 2)
        else:

            for id in self.cables:
                cable = self.cables[id]
                cable_points = cable.get_points()
                for i in range(len(cable_points)-1):
                    p1 = tuple(cable_points[i])
                    p2 = tuple(cable_points[i + 1])
                    cv2.line(img_display, p1, p2, (0, 255, 255), 2)


        # if self.cable_positions:
        #     for i in range(len(self.cable_positions) - 1):
        #         p1 = tuple(self.cable_positions[i])
        #         p2 = tuple(self.cable_positions[i + 1])
        #         cv2.line(img_display, p1, p2, (0, 255, 255), 2)

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
        return self.key_locations.copy()
    
    def get_cables(self):
        return self.cables.copy()

    def get_cable_path(self):
        '''
        TODO: Deprecated
        '''

        return self.cables[self.cables.keys()[0]].get_points()
    
    def return_board(self):
        """
        Returns a representation of the board as an occupancy map of cables
        """
        return self.board

class Cable:
    """
    A class for representing individual cables on the board, including estimates of their positions and key points.
    """

    num_cables_created = 0

    def __init__(self, coordinates, id=None, environment=None):
        """
        Instantiate an instance of our cable object.

                Parameter(s):
                        coordinates (list[(int,int)...]): A list of the coordinates of our cable board positions
                        id (int), optional: A unique integer identifier for our cable object. Note that multiple instances of a cable can have the same id, if they are intended to represent the same cable (e.g. a present state of the cable and the desired state of that cable)
                        environment (Board), optional: The environment that the cable is a part of, can be used to update the cable's relevant keypoints

                Returns:
                        None
        """
        self.true_coordinates = coordinates.copy()

        self.quantized = [] # Coordinates of positions within the environment's grid space
        self.keypoints = []
        self.environment = environment
        if environment != None:
            self.update_keypoints(environment)

        if id == None:
            self.id = self.num_cables_created
            self.num_cables_created += 1
        else:
            self.id = id

    def update_quantized(self, environment=None):
        """
        Update the scale of cable coordinates based on the grid size of a particular "environment" of choice, or using the environment that was assigned upon instantiation.

                Parameter(s):
                        environment (Board), optional: The environment we would like to use for updating the quantization scale, leave blank if we would like to use the instatiated environment

                Returns:
                        None
        """
        if environment == None and self.environment != None:
            environment = self.environment
            
        if environment != None:
            quantized_coords = [
                (int(round(point[0]/environment.grid_size[0])), int(round(point[1]/environment.grid_size[1])))
                for point in self.true_coordinates
            ]

            if len(quantized_coords) > 0:
                # Clean up proximal duplicates
                self.quantized = [quantized_coords[0]]
                most_recent_coord = quantized_coords[0]
                for i in range(1, len(quantized_coords)):
                    if quantized_coords[i] != most_recent_coord:
                        self.quantized.append(quantized_coords[i])
                        most_recent_coord = quantized_coords[i]


    
    def update_keypoints(self, environment=None):
        """
        Update the set of the keypoints for the cable, either using a particular "environment" of choice, or using the environment that was assigned upon instantiation.

                Parameter(s):
                        environment (Board:), optional: The environment we would like to use for updating the cable's keypoints, leave blank if we would like to use the instatiated environment

                Returns:
                        None
        """
        if environment == None and self.environment != None:
            environment = self.environment

        self.update_quantized(environment=environment)

        if environment != None:
            self.keypoints = [
                point
                for point in self.quantized
                if point in environment.key_locations
            ]

    # Appropriate getter functions
    def get_keypoints(self):
        return self.keypoints.copy()

    def get_quantized(self):
        return self.quantized.copy()

    def get_points(self):
        return self.true_coordinates.copy()

    def length(self):
        return len(self.true_coordinates)

    def get_id(self):
        return self.id


class BoardFeature:
    """
    A class representing a key feature of the board environment, such as a plug, bracket, or spool, including relevant feature data.

    Can be subclassed to define more specific features with particular behavior.
    """

    def __init__(self, x, y, type = "N/A", orientation = 0, environment = None):
        """
        Instantiate an instance of our feature object.

                Parameter(s):
                        x (int): The pixel x coordinate of our board feature
                        y (int): The pixel y coordinate of our board feature
                        type (str): A string indicating the type of feature that it is
                        orientation (int?): A representation for the angle of our feature
                        environment (Board:), optional: The environment that the feature is a part of, can be used to update the feature's coordinate scale

                Returns:
                        None
        """
        self.true_coordinate = (x, y, orientation)
        self.type = type

        if environment != None:
            self.coordinate = (int(round(x / environment.grid_size[0])), int(round(y / environment.grid_size[1])), orientation)
        else:
            self.coordinate = self.true_coordinate

    # Getter functions
    def get_position(self):
        return (self.coordinate[0], self.coordinate[1])
    
    def get_type(self):
        return self.type

    def get_orientation(self):
        return self.coordinate[2]
    

# Correction algorithm, unclean implementation
def levenshtein_algo(given, desired, environment):
    """
    Given two sequences of coordinates, "given" and "desired", provide the minimum-cost sequence of additions, removals and swaps to produce "desired" from "given".

            Parameter(s):
                    given (list[(int, int), ...]): The current sequence of coordinates (usually keypoints) for a given cable
                    desired (list[(int, int), ...]): The desired sequence of coordinates (usually keypoints) for the cable
                    environment (Board): Information on the environment involved, used purely for the cost functions
            Returns:
                    instructions (list[str, ...]): A list of verbal instructions to produce the "desired" sequence from the "given" sequence

    """

    x = given.copy()
    y = desired.copy()

    # Cost functions associated with the different moves, can involve the respective coordinates, any function of the 
    # environment, or simply a constant. Output should be integer-valued
    add_cost = lambda desired_coords, board: 1 #(desired_coords[0]**2 + desired_coords[1]**2)//400 + 1
    remove_cost = lambda given_coords, board: 1
    swap_cost = lambda given_coords, desired_coords, board: 1 # ((given_coords[0]-desired_coords[0])**2 + (given_coords[1]-desired_coords[1])**2)//100 + 1

    # Traverse the DP table to generate cost scores for all sequences of coordinates
    n = len(y)
    m = len(x)
    E = [[0 for b in range(n + 1)] for a in range(m + 1)]

    for a in range(1, m + 1):
        E[a][0] = E[a - 1][0] + remove_cost(x[a-1] ,environment)

    for b in range(1, n + 1):
        E[0][b] = E[0][b - 1] + add_cost(y[b-1] ,environment)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                E[i][j] = E[i - 1][j - 1]
            else:
                E[i][j] = min(
                    [
                        E[i][j - 1] + add_cost(y[j-1] ,environment),
                        E[i - 1][j] + remove_cost(x[i-1] ,environment),
                        E[i - 1][j - 1] + swap_cost(x[i-1], y[j-1] ,environment),
                    ]
                )

    # Generate a sequence of actions that corresponds to one of the minimum-cost solutions
    buildx = []
    buildy = []
    c = m
    d = n
    while c > 0 or d > 0:
        
        if c > 0 and E[c][d] == E[c - 1][d] + remove_cost(x[c-1],environment):
            buildx = [x[c - 1]] + buildx
            buildy = ["Remove"] + buildy
            c = c - 1
        elif d > 0 and E[c][d] == E[c][d - 1] + add_cost(y[d-1] ,environment):
            buildx = ["Add"] + buildx
            buildy = [y[d - 1]] + buildy
            d = d - 1
        elif E[c][d] == E[c-1][d - 1] or E[c][d] == E[c-1][d - 1] + swap_cost(x[c-1], y[d-1], environment):
            buildx = [x[c - 1]] + buildx
            buildy = [y[d - 1]] + buildy
            c = c - 1
            d = d - 1
        else:
            print("Uh oh.")
            # pretty_matrix(E)
            break
        

    # Generate the instructions list
    def safe_index(list, index):
        if index in range(1, len(list)-1):
            return list[index]
        else:
            return None
    instructions = []

    # print(buildx)
    # print(buildy)
    for i in range(len(buildx)):
        if buildx[i] == "Add":

            # Some code to output
            first_print = ""
            last_print = ""
            prev_index = i - 1
            next_index = i + 1

            while prev_index >= 0 and buildx[prev_index] == 'Add':
                prev_index -= 1

            while next_index < len(buildx) and buildx[next_index] == 'Add':
                next_index += 1

            if prev_index < 0:
                first_print = "the beginning"
            else:
                first_print = buildx[prev_index]
            
            if next_index >= len(buildx):
                last_print = "the end"
            else:
                last_print = buildx[next_index]
            
            instructions.append(f"Add {str(buildy[i])} between {first_print} and {last_print}")

            # instructions.append(f"Add {str(buildy[i])} at step {i}")
        elif buildy[i] == "Remove":
            instructions.append(f"Remove {str(buildx[i])} at step {i}")
        elif buildx[i] != buildy[i]:
            instructions.append(f"Swap {buildx[i]} to {buildy[i]}")

    return instructions


def suggest_modifications(environment, goal_configuration):
        """
        Given a goal cable configuration "goal_configuration", suggest the set of moves to produce that configuration given the board's current state.

                Parameter(s):
                        goal_configuration (dict{int:list[(int, int), ...], ...}): The desired configuration of cables for the board

                Returns:
                        instructions (list[str, ...]): A list of verbal instructions to produce the goal configuration from the given board configuration
        """

        instructions = []
        for cable_id in goal_configuration:
            goal_cable = goal_configuration[cable_id]
            if cable_id in environment.get_cables():
                # print(self.cables[cable_id].get_keypoints())
                # print(goal_cable.get_keypoints())
                instructions.append(
                    levenshtein_algo(
                        environment.get_cables()[cable_id].get_keypoints(),
                        goal_cable.get_keypoints(),
                        environment
                    )
                )
            else:
                instructions.append(f"Missing cable {cable_id}")

        # for instruction in instructions:
        #     print(instruction)

        return instructions

if __name__ == "__main__":
    import random

    board = NewBoard(config_path="../../../data/board_config.json", grid_size=(20,20))

    img = cv2.imread("../../../data/board_setup.png")

    test_cable = Cable([(100, 200), (150, 250), (200, 300), (200, 300)])
    test_cable = Cable([(600, 200), (900, 400), (1200, 300), (900, 350), (582, 5), (1391, 767)])
    # board.add_cable(test_cable)

    # ideal_cable = Cable([(653, 634), (701, 84), (890, 578), (829, 177), (1317, 559), (1182, 259), (655, 549), (974, 256), (1283, 88), (1319, 640)], id = -1)
    feature_coordinates = [(653, 634), (701, 84), (890, 578), (829, 177), (1317, 559), (1182, 259), (655, 549), (974, 256), (1283, 88), (1319, 640)]

    theoretical_cable = Cable([(701, 84),(829, 177), (974, 256), (890, 578),(1317, 559)], id=-2)
    # theoretical_cable = Cable([feature_coordinates[random.randint(0, len(feature_coordinates)-1)] for o in range(random.randint(4, 6))], id=-1)

    sequence = [(701, 84),(829, 177), (974, 256), (890, 578),(1317, 559)]
    random.shuffle(sequence)
    ideal_cable = Cable(sequence, id=-1)

    board.add_cable(theoretical_cable)
    # board.add_cable(ideal_cable)


    

    # Generate fake cables
    for k in range(1):
        test = []
        # first_pos = (random.randint(600, 1350), random.randint(0, 750))
        # rate = (random.randint(-20, 20), random.randint(-20, 20))

        first_pos = (701, 84)
        test.append(first_pos)
        rate = (20, 15)
        for i in range(random.randint(60, 100)):
            new_pos = (first_pos[0] + rate[0], first_pos[1] + rate[1])
            rate = (rate[0] + random.randint(-5, 5), rate[1]+ random.randint(-5, 5))
        
            if (
                new_pos[0] in range(600, 1350)
                and new_pos[1] in range(0, 750)
                # and new_pos not in test
            ):
                test.append(new_pos)
                first_pos = new_pos

        test_cable = Cable(test, id=k)
        board.add_cable(test_cable)

    #     print(test_cable.get_quantized())
    #     print(test_cable.get_keypoints())
    # print(board.key_locations)


    # Move frequency experiments
    counts = {"Add": 0, "Swap": 0, "Remove": 0}
    for w in range(0):
        random.shuffle(sequence)
        ideal_cable = Cable(sequence, id=-1)
        ideal_cable.update_keypoints(board)

        goal_config = {-2: ideal_cable}
        suggestion = suggest_modifications(board, goal_configuration=goal_config)
        print(suggestion[0])
        counts["Add"] += len([idea for idea in suggestion[0] if "Add" in idea])
        counts["Swap"] += len([idea for idea in suggestion[0] if "Swap" in idea])
        counts["Remove"] += len([idea for idea in suggestion[0] if "Remove" in idea])

    print(counts)

    goal_config = {0: theoretical_cable}
    suggestion = suggest_modifications(board, goal_configuration=goal_config)
    print(suggestion)
    # test2 = [
    #     (cable1.all_coordinates[random.randint(0, len(cable1.all_coordinates) - 1)])
    #     for i in range(random.randint(4, 7))
    # ]

    # pretty_matrix(board.return_board())

    # print(levenshtein_algo(board.get_cables()[0].get_keypoints(), ideal_cable.get_keypoints(), board))

    annotated_img = board.visualize_board(img, quantized=False)
