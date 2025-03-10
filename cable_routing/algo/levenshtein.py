import numpy as np


# Correction algorithm, unclean implementation
def levenshtein_algo(given, desired, environment, human_readable=True):
    """
    Given two sequences of coordinates, "given" and "desired", provide the minimum-cost sequence of additions, removals and swaps to produce "desired" from "given".

            Parameter(s):
                    given (list[(int, int), ...]): The current sequence of coordinates (usually keypoints) for a given cable
                    desired (list[(int, int), ...]): The desired sequence of coordinates (usually keypoints) for the cable
                    environment (Board): Information on the environment involved, used purely for the cost functions
                    human_readable (bool): Indicates whether the output should be in human-readable (string) or machine-readable (tuple) form
            Returns:
                    instructions (list[str, ...]): A list of verbal instructions to produce the "desired" sequence from the "given" sequence

    """

    x = given.copy()
    y = desired.copy()

    # Cost functions associated with the different moves, can involve the respective coordinates, any function of the
    # environment, or simply a constant. Output should be integer-valued
    add_cost = (
        lambda desired_coords, board: 1
    )  # (desired_coords[0]**2 + desired_coords[1]**2)//400 + 1
    remove_cost = lambda given_coords, board: 1
    swap_cost = (
        lambda given_coords, desired_coords, board: 100000
    )  # ((given_coords[0]-desired_coords[0])**2 + (given_coords[1]-desired_coords[1])**2)//100 + 1

    # Traverse the DP table to generate cost scores for all sequences of coordinates
    n = len(y)
    m = len(x)
    E = [[0 for b in range(n + 1)] for a in range(m + 1)]

    for a in range(1, m + 1):
        E[a][0] = E[a - 1][0] + remove_cost(x[a - 1], environment)

    for b in range(1, n + 1):
        E[0][b] = E[0][b - 1] + add_cost(y[b - 1], environment)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                E[i][j] = E[i - 1][j - 1]
            else:
                E[i][j] = min(
                    [
                        E[i][j - 1] + add_cost(y[j - 1], environment),
                        E[i - 1][j] + remove_cost(x[i - 1], environment),
                        E[i - 1][j - 1] + swap_cost(x[i - 1], y[j - 1], environment),
                    ]
                )

    # Generate a sequence of actions that corresponds to one of the minimum-cost solutions
    buildx = []
    buildy = []
    c = m
    d = n
    while c > 0 or d > 0:

        if c > 0 and E[c][d] == E[c - 1][d] + remove_cost(x[c - 1], environment):
            buildx = [x[c - 1]] + buildx
            buildy = ["Remove"] + buildy
            c = c - 1
        elif d > 0 and E[c][d] == E[c][d - 1] + add_cost(y[d - 1], environment):
            buildx = ["Add"] + buildx
            buildy = [y[d - 1]] + buildy
            d = d - 1
        elif E[c][d] == E[c - 1][d - 1] or E[c][d] == E[c - 1][d - 1] + swap_cost(
            x[c - 1], y[d - 1], environment
        ):
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
        if index in range(1, len(list) - 1):
            return list[index]
        else:
            return None

    instructions = []

    # print(buildx)
    # print(buildy)

    if human_readable:
        for i in range(len(buildx)):
            if buildx[i] == "Add":

                # Some code to output
                first_print = ""
                last_print = ""
                prev_index = i - 1
                next_index = i + 1

                while prev_index >= 0 and buildx[prev_index] == "Add":
                    prev_index -= 1

                while next_index < len(buildx) and buildx[next_index] == "Add":
                    next_index += 1

                if prev_index < 0:
                    first_print = "the beginning"
                else:
                    first_print = buildx[prev_index]

                if next_index >= len(buildx):
                    last_print = "the end"
                else:
                    last_print = buildx[next_index]

                instructions.append(
                    f"Add {str(buildy[i])} between {first_print} and {last_print}"
                )

                # instructions.append(f"Add {str(buildy[i])} at step {i}")
            elif buildy[i] == "Remove":
                instructions.append(f"Remove {str(buildx[i])} at step {i}")
            elif buildx[i] != buildy[i]:
                instructions.append(f"Swap {buildx[i]} to {buildy[i]}")
    else:
        to_pixel_space = lambda my_tuple: tuple([environment.grid_size[0] * my_tuple[0], environment.grid_size[0] * my_tuple[1]] + [my_tuple[n] for n in range(2, len(my_tuple))])
        for i in range(len(buildx)):
            if buildx[i] == "Add":

                # Some code to output
                first_print = ""
                last_print = ""
                prev_index = i - 1
                next_index = i + 1

                while prev_index >= 0 and buildx[prev_index] == "Add":
                    prev_index -= 1

                while next_index < len(buildx) and buildx[next_index] == "Add":
                    next_index += 1

                if prev_index < 0:
                    first_print = None
                else:
                    first_print = to_pixel_space(buildx[prev_index])

                if next_index >= len(buildx):
                    last_print = None
                else:
                    last_print = to_pixel_space(buildx[next_index])

                instructions.append(("Add", to_pixel_space(buildy[i]), first_print, last_print))

                # instructions.append(f"Add {str(buildy[i])} at step {i}")
            elif buildy[i] == "Remove":
                instructions.append(("Remove", to_pixel_space(buildx[i]), None, None))
            elif buildx[i] != buildy[i]:
                if i == 0:
                    prev_coord = None
                else:
                    prev_coord = buildx[i-1]

                instructions.append(("Swap", to_pixel_space(buildx[i]), to_pixel_space(buildy[i]), prev_coord))

    return instructions


def suggest_modifications(environment, goal_configuration, human_readable=False):
    """
    Given a goal cable configuration "goal_configuration", suggest the set of moves to produce that configuration given the board's current state.

            Parameter(s):
                    environment (Board): The board environment of cables that we would like to correct
                    goal_configuration (dict{int:list[(int, int), ...], ...}): The desired configuration of cables for the board
                    human_readable (bool): Indicates whether the output should be in human-readable (string) or machine-readable (tuple) form

            Returns:
                    instructions (list[str, ...]): A list of verbal instructions to produce the goal configuration from the given board configuration
    """

    cable_ids = []
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
                    environment,
                    human_readable=human_readable,
                )
            )
            cable_ids.append(cable_id)
        else:
            instructions.append(f"Missing cable {cable_id}")

    if human_readable:
        return instructions
    else :
        pick_place = []
        for cable_num in range(len(instructions)):
            cable = environment.get_cables()[cable_ids[cable_num]]
            add_dictionary = {}
            key = () # Keypoint for the last add operation we needed to do
            to_add = []
            for instruction in instructions[cable_num]:
                if instruction[0] == "Add":
                    keypoint1 = instruction[2] if instruction[2] != None else cable.true_coordinates[0]
                    keypoint2 = instruction[3] if instruction[3] != None else cable.true_coordinates[-1]
                    if (keypoint1, keypoint2) == key:
                        to_add.append(instruction[1])
                    else :
                        if key != ():
                            grasp_points = cable.intermediate_points((key[0][0]//cable.grid_size[0], key[0][1]//cable.grid_size[1]), (key[1][0]//cable.grid_size[0], key[1][1]//cable.grid_size[1]), num_points = len(to_add))
                        for goal_index in range(len(to_add)):
                            pick_place.append((grasp_points[goal_index], to_add[goal_index]))
                        key = (keypoint1, keypoint2)
                        to_add = [instruction[1]]


                    # if (keypoint1, keypoint2) in add_dictionary:
                    #     add_dictionary[(keypoint1, keypoint2)].append(instruction[1])
                    # else:
                    #     add_dictionary[(keypoint1, keypoint2)] = [instruction[1]]
                elif instruction[0] == "Remove":
                    # TODO: Can change output structure
                    pick_place.append(((instruction[1][0], instruction[1][1]), (instruction[1][0]+10, instruction[1][1]+10)))
                elif instruction[0] == "Swap":
                    pick_place.append(((instruction[1][0], instruction[1][1]), instruction[2]))
            
            # for key in add_dictionary:
            #     grasp_points = cable.intermediate_points((key[0][0]//cable.grid_size[0], key[0][1]//cable.grid_size[1]), (key[1][0]//cable.grid_size[0], key[1][1]//cable.grid_size[1]), num_points = len(add_dictionary[key]))
            #     for goal_index in range(len(add_dictionary[key])):
            #         pick_place.append((grasp_points[goal_index], add_dictionary[key][goal_index]))
        return pick_place

    # for instruction in instructions:
    #     print(instruction)
