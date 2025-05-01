"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math

import matplotlib.pyplot as plt

show_animation = False


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return (
                str(self.x)
                + ","
                + str(self.y)
                + ","
                + str(self.cost)
                + ","
                + str(self.parent_index)
            )

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(
            self.calc_xy_index(sx, self.min_x),
            self.calc_xy_index(sy, self.min_y),
            0.0,
            -1,
        )
        goal_node = self.Node(
            self.calc_xy_index(gx, self.min_x),
            self.calc_xy_index(gy, self.min_y),
            0.0,
            -1,
        )

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost
                + self.calc_heuristic(goal_node, open_set[o]),
            )
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(
                    self.calc_grid_position(current.x, self.min_x),
                    self.calc_grid_position(current.y, self.min_y),
                    "xc",
                )
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    "key_release_event",
                    lambda event: [exit(0) if event.key == "escape" else None],
                )
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(
                    current.x + self.motion[i][0],
                    current.y + self.motion[i][1],
                    current.cost + self.motion[i][2],
                    c_id,
                )
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)
        ]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        # obstacle map generation
        self.obstacle_map = [
            [False for _ in range(self.y_width)] for _ in range(self.x_width)
        ]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)],
        ]

        return motion


def main():
    print(__file__ + " start!!")

    def random_coordinate(bottom_vertex, top_vertex):
        """
        Generate random (x, y) coordinate within the 20x20 grid boundary.
        """
        x = np.random.randint(bottom_vertex[0] + 1, top_vertex[0])
        y = np.random.randint(bottom_vertex[1] + 1, top_vertex[1])
        return [x, y]

    # simple test
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from cable_routing.env.board.new_board import Board
    from cable_routing.configs.envconfig import ExperimentConfig

    # === 1. Load Board and Clips ===
    cfg = ExperimentConfig

    top_vertex = [20, 20]  # Planner map size (grid units)
    bottom_vertex = [0, 0]

    board = Board(config_path=cfg.board_cfg_path, grid_size=(20, 20))
    img = cv2.imread(cfg.bg_img_path)

    # Detect the clips on the board (actual obstacles)
    clips = board.get_clips()

    # Determine board size in the image (used for scaling)
    board_coordiante = board.point1  # "Top-left corner" of board
    board_end = board.point2  # "Bottom-right corner" of board

    board_width = board_end[0] - board_coordiante[0]
    board_height = board_end[1] - board_coordiante[1]

    # Scale from image coordinates into 20x20 grid (planner coordinates)
    scale_x = (top_vertex[0] - bottom_vertex[0]) / board_width
    scale_y = (top_vertex[1] - bottom_vertex[1]) / board_height

    # === 2. Convert Clips to Scaled Obstacle Coordinates ===
    ox, oy = [], []
    for clip in clips:
        # Shift so (0,0) = top-left corner of the board
        # cuz everything in camera frame - TODO :fix
        clip["x"] -= board_coordiante[0]
        clip["y"] -= board_coordiante[1]

        # Scale to fit into grid
        clip["x"] *= scale_x
        clip["y"] *= scale_y

        ox.append(clip["x"])
        oy.append(clip["y"])

    # === 3. Add Outer Boundary (Board Edges) to Obstacles ===
    for x in range(top_vertex[0] + 1):  # Along x-axis (0 to 20)
        ox.append(x)
        oy.append(0)
        ox.append(x)
        oy.append(top_vertex[0])
    for y in range(top_vertex[1] + 1):  # Along y-axis (0 to 20)
        ox.append(0)
        oy.append(y)
        ox.append(top_vertex[1])
        oy.append(y)

    # === 4. Set Start & Goal (could be user-defined, or random within boundary) ===
    start = random_coordinate(bottom_vertex, top_vertex)
    end = random_coordinate(bottom_vertex, top_vertex)
    sx, sy = start[0], start[1]
    gx, gy = end[0], end[1]

    grid_size = 1.0  # Grid resolution (each grid cell = 1 unit in the 20x20 space)
    robot_radius = 1.0  # Robot size (buffer around obstacles)

    # === 5. Visualization Setup (optional) ===
    if show_animation:
        plt.plot(ox, oy, ".k", label="Obstacles")
        plt.plot(sx, sy, "og", label="Start")
        plt.plot(gx, gy, "xb", label="Goal")
        plt.grid(True)
        plt.axis("equal")

    # === 6. Run A* Planner ===
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    # === 7. Show Final Path (optional) ===
    if show_animation:
        plt.plot(rx, ry, "-r", label="Path")
        plt.legend()
        plt.pause(0.001)
        plt.gca().invert_yaxis()  # for visualization
        plt.show()

    print(f"Planned Path (grid coordinates): {list(zip(rx, ry))}")


if __name__ == "__main__":
    show_animation = True
    main()
