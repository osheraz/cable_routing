import numpy as np


class Cable:
    """
    A class for representing individual cables on the board,
    including estimates of their positions and key points.
    """

    def __init__(self, coordinates, env_keypoints, grid_size, id):
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
        self.quantized = []
        self.keypoints = []
        self.id = id
        self.grid_size = grid_size
        self.update_quantized(grid_size=grid_size)
        self.update_keypoints(env_keypoints=env_keypoints)

    def update_keypoints(self, env_keypoints=None):
        """
        Update the set of the keypoints for the cable,

        Parameter(s):
                env_keypoints, optional: The environment we would like
                to use for updating the cable's keypoints, leave blank
                if we would like to use the instatiated environment

        Returns:
                None
        """

        if env_keypoints != None:
            self.keypoints = [
                point for point in self.quantized if point in env_keypoints
            ]

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
