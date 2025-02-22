import random
import pprint

class cable_environment:
    '''
    An overarching class that encapsulates the current features and state of a board configuration, as well as relevant algorithms with respect to cable correction.
    '''
    def __init__(self, width = 10, height = 10):
        '''
        Initialize a default instance of our board environment.

                Parameter(s):
                        width (int): The width in tiles of our board representation
                        height (int): The height in tiles of our board representation

                Returns:
                        None
        '''

        self.cables = {}
        self.width = width
        self.height = height
        self.board = [["." for i in range(self.width)] for j in range(self.height)]


        self.key_locations = set()
        self.key_features = {}
        self.goal_configuration = {}

    
    def add_cable(self, cable):
        '''
        Adds a cable object to our environment representation.

                Parameter(s):
                        cable (cable): The cable to add

                Returns:
                        None
        '''
        self.cables[cable.get_id()] = cable
        cable.update_keypoints(self)
 
        for coordinate in cable.get_points():
            if self.board[coordinate[0]][coordinate[1]] == ".":
                self.board[coordinate[0]][coordinate[1]] = str(cable.get_id())
            else:
                # print(self.board[coordinate[0]][coordinate[1]])
                self.board[coordinate[0]][coordinate[1]] = self.board[coordinate[0]][coordinate[1]] + f", {cable.get_id()}"

    def add_keypoint(self, feature):
        '''
        Adds a feature of interest to our environment representation. This might include a bracket or a plug, for example

                Parameter(s):
                        feature (board_feature): The feature to object to be included on our board

                Returns:
                        None
        '''

        self.key_locations.add(feature.get_position())
        self.key_features[feature.get_position()] = feature
        for cable in self.cables:
            self.cables[cable].update_keypoints(self)

    # Correction algorithm, unclean implementation
    def pairwise_correct(self, given, desired):
        '''
        Given two sequences of coordinates, "given" and "desired", provide the minimum-cost sequence of additions, removals and swaps to produce "desired" from "given".

                Parameter(s):
                        given (list[(int, int), ...]): The current sequence of coordinates (usually keypoints) for a given cable
                        desired (list[(int, int), ...]): The desired sequence of coordinates (usually keypoints) for the cable

                Returns:
                        instructions (list[str, ...]): A list of verbal instructions to produce the "desired" sequence from the "given" sequence
        
        TODO: Write support for coordinate-dependent costs, rather than simply action-specific constants
        '''

        x = given.copy()
        y = desired.copy()

        # Cost values of each action, currently hard-coded constants
        ADD_COST = 1
        REMOVE_COST = 1
        SWAP_COST = 1

        # Traverse the DP table to generate cost scores for all sequences of coordinates
        n = len(y)
        m = len(x)
        E = [[0 for b in range(n+1)] for a in range(m+1)]

        for a in range(m+1):
            E[a][0] = a
            
        for b in range(n+1):
            E[0][b] = b

        for i in range(1,m+1):
            for j in range(1,n+1):
                if x[i-1] == y[j-1]:
                    E[i][j] = E[i-1][j-1]
                else :
                    E[i][j] = min([E[i][j-1]+ADD_COST, E[i-1][j]+REMOVE_COST, E[i-1][j-1]+SWAP_COST])

        # Generate a sequence of actions that corresponds to one of the minimum-cost solutions
        buildx = []
        buildy = []
        c = m
        d = n
        while c > 0 or d > 0:
            if c>0 and E[c][d] == E[c-1][d] + REMOVE_COST:
                buildx = [x[c-1]] + buildx
                buildy = ["Remove"] + buildy
                c = c-1
            elif d > 0 and E[c][d] == E[c][d-1] + ADD_COST:
                buildx = ["Add"] + buildx
                buildy = [y[d-1]]+ buildy
                d = d-1
            else:
                buildx = [x[c - 1]] + buildx
                buildy = [y[d - 1]] + buildy
                c = c-1
                d = d-1

        # Generate the instructions list
        instructions = []
        for i in range(len(buildx)):
            if buildx[i] == "Add":
                instructions.append(f"Add {str(buildy[i])} at step {i}")
            elif buildy[i] == "Remove":
                instructions.append(f"Remove {str(buildx[i])} at step {i}")
            elif buildx[i] != buildy[i]:
                instructions.append(f"Swap {buildx[i]} to {buildy[i]}")

        return instructions
    
    def suggest_modifications(self, goal_configuration):
        '''
        Given a goal cable configuration "goal_configuration", suggest the set of moves to produce that configuration given the board's current state. 

                Parameter(s):
                        goal_configuration (dict{int:list[(int, int), ...], ...}): The desired configuration of cables for the board

                Returns:
                        instructions (list[str, ...]): A list of verbal instructions to produce the goal configuration from the given board configuration
        '''

        instructions = []
        for cable_id in goal_configuration:
            goal_cable = goal_configuration[cable_id]
            if cable_id in self.cables:
                # print(self.cables[cable_id].get_keypoints())
                # print(goal_cable.get_keypoints())
                instructions.append(self.pairwise_correct(self.cables[cable_id].get_keypoints(), goal_cable.get_keypoints()))
            else :
                instructions.append(f"Missing cable {cable_id}")

        # for instruction in instructions:
        #     print(instruction)

        return instructions
    
    def return_board(self):
        '''
        Returns a representation of the board as an occupancy map of cables
        '''
        return self.board
   
class cable:
    '''
    A class for representing individual cables on the board, including estimates of their positions and key points.
    '''
    num_cables_created = 0
    def __init__(self, coordinates, id=None, environment=None):
        '''
        Instantiate an instance of our cable object.

                Parameter(s):
                        coordinates (list[(int,int)...]): A list of the coordinates of our cable board positions
                        id (int), optional: A unique integer identifier for our cable object. Note that multiple instances of a cable can have the same id, if they are intended to represent the same cable (e.g. a present state of the cable and the desired state of that cable)
                        environment (cable_environment), optional: The environment that the cable is a part of, can be used to update the cable's relevant keypoints

                Returns:
                        None
        '''
        self.all_coordinates = coordinates.copy()

        self.keypoints = []
        self.environment = environment
        if environment != None:
            self.update_keypoints(environment)

        if id == None:
            self.id = cable.num_cables_created
            cable.num_cables_created += 1
        else:
            self.id = id
        

    def update_keypoints(self, environment = None):
        '''
        Update the set of the keypoints for the cable, either using a particular "environment" of choice, or using the environment that was assigned upon instantiation.

                Parameter(s):
                        environment (cable_environment), optional: The environment we would like to use for updating the cable's keypoints, leave blank if we would like to use the instatiated environment

                Returns:
                        None
        '''
        if environment != None:
            self.keypoints = [point for point in self.all_coordinates if point in environment.key_locations]
        elif self.environment != None:
            self.keypoints = [point for point in self.all_coordinates if point in self.environment.key_locations]


    # Appropriate getter functions
    def get_keypoints(self):
        return self.keypoints.copy()
    
    def get_points(self):
        return self.all_coordinates.copy()
    
    def length(self):
        return len(self.all_coordinates)
    
    def get_id(self):
        return self.id

class board_feature:
    '''
    A class representing a key feature of the board environment, such as a plug, bracket, or spool, including relevant feature data.

    Can be subclassed to define more specific features with particular behavior.
    '''
    def __init__(self, coordinate):
        self.coordinate = coordinate

    def get_position(self):
        return self.coordinate

    
# Testing code
if __name__ == "__main__":
    my_environment = cable_environment()
    goal_configuration = {}
    for k in range(3):

        # Generate fake cables
        test = []
        first_pos = (random.randint(1, 9), random.randint(1, 9))
        for i in range(random.randint(6, 10)):
            if random.randint(1, 4) == 1:
                new_pos = (first_pos[0]+1, first_pos[1])
            elif random.randint(1, 4) == 2:
                new_pos = (first_pos[0]-1, first_pos[1])
            elif random.randint(1, 4) == 3:
                new_pos = (first_pos[0], first_pos[1]+1)
            else:
                new_pos = (first_pos[0], first_pos[1]-1)
            
            if new_pos[0] in range(0, 10) and new_pos[0] in range(0, 10) and new_pos not in test:
                test.append(new_pos)
                first_pos = new_pos
        cable1 = cable(test, id=k, environment=my_environment)

        test2 = [(cable1.all_coordinates[random.randint(0, len(cable1.all_coordinates)-1)]) for i in range(random.randint(4, 7))]
        goal_configuration[k] = cable(test2, id=k, environment=my_environment)
        my_environment.add_cable(cable1)

        [my_environment.add_keypoint(board_feature(position)) for position in cable1.all_coordinates]

    for cable_id in goal_configuration:
        goal_configuration[cable_id].update_keypoints()

    # A pretty print for a matrix, thanks StackOverflow!
    def pretty_matrix(matrix):
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print ('\n' + '\n'.join(table) + '\n')

    # pprint.pprint(my_environment.return_board())

    pretty_matrix(my_environment.return_board())
    
    print(my_environment.suggest_modifications(goal_configuration))


