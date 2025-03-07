import numpy as np


class Cable:
    """
    A class for representing individual cables on the board,
    including estimates of their positions and key points.
    """

    def __init__(self, coordinates, environment, grid_size, id):
        """
        Instantiate an instance of our cable object.

        Parameter(s):
                coordinates (list[(int,int)...]): A list of the coordinates of
                our cable board positions
                id (int), optional: A unique integer identifier for our cable object.
                Note that multiple instances of a cable can have the same id,
                if they are intended to represent the same cable
                (e.g. a present state of the cable and the desired state of that cable)
                can be used to update the cable's relevant keypoints

        Returns:
                None
        """
        self.true_coordinates = coordinates.copy()
        self.environment = environment
        self.quantized = []
        self.keypoints = []
        self.id = id
        self.grid_size = grid_size
        self.update_quantized(grid_size=grid_size)
        self.update_keypoints(environment)

    def update_keypoints(self, environment=None):
        """
        Update the set of the keypoints for the cable,

        Parameter(s):
                env_keypoints, optional: The environment we would like
                to use for updating the cable's keypoints, leave blank
                if we would like to use the instatiated environment

        Returns:
                None
        """
        
        
        abs_value = lambda x: x if x > 0 else -x

        if environment != None:
            self.keypoints = []
            for point in self.quantized:
                if point in environment.key_locations:
                    feature = environment.key_features[point]
                    if feature.type in ["Clip", "6Pin"]:
                        true_point = self.true_point(point)
                        # true_point = find_first(self.true_coordinates, lambda x: (
                        #         int(round(x[0] / self.grid_size[0])),
                        #         int(round(x[1] / self.grid_size[1])),
                        #     ) == point
                        # )
                        
                        xdiff = true_point[0]-point[0]*self.grid_size[0]
                        ydiff = true_point[1]-point[1]*self.grid_size[1]
                        direction = 0
                        if abs_value(xdiff) > abs_value(ydiff):
                            if xdiff < 0:
                                direction = 2
                        else:
                            if ydiff > 0:
                                direction = 1
                            else:
                                direction = 3
                        
                        point = (point[0], point[1], direction)
                        
                    self.keypoints.append(point)

    def update_quantized(self, grid_size=None):
        """
        Update the scale of cable coordinates based on the grid size of a particular
        "environment" of choice, or using the environment that was assigned upon instantiation.

                Parameter(s):
                        environment (Board), optional: The environment we would like to
                        use for updating the quantization scale, leave blank if we
                        would like to use the instatiated environment

                Returns:
                        None
        """
        quantized_coords = [
            (
                int(round(point[0] / self.grid_size[0])),
                int(round(point[1] / self.grid_size[1])),
            )
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

    def true_point(self, coord):
        find_first = lambda list, condition: [item for item in list if condition(item)][0]
        true_point = find_first(self.true_coordinates, lambda x: (
                                int(round(x[0] / self.grid_size[0])),
                                int(round(x[1] / self.grid_size[1])),
                            ) == coord
                        )

        return true_point

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
