import random

# Levenshtein distance implementation
def error_correction(x, y):
    ADD_COST = 1
    REMOVE_COST = 3
    SWAP_COST = 3

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

    instructions = []
    for i in range(len(buildx)):
        if buildx[i] == "Add":
            instructions.append(f"Add {str(buildy[i])} at step {i}")
        elif buildy[i] == "Remove":
            instructions.append(f"Remove {str(buildx[i])} at step {i}")
        elif buildx[i] != buildy[i]:
            instructions.append(f"Swap {buildx[i]} to {buildy[i]}")
    return instructions


if __name__ == "__main__":
    # Example usage:
    cable1 = [(random.randint(1, 9), random.randint(1, 9)) for i in range(random.randint(4, 7))]
    cable2 = [(cable1[random.randint(0, len(cable1)-1)]) for i in range(random.randint(4, 7))]

    # s1 = [(1,2), (3,4), (5,6)]
    # s2 = [(2,2), (6,1), (3,4), (5,6)]

    cable1 = [(7,8), (6,4), (1,2)]
    cable2 = [(7,8), (3,6), (6,4), (7,1)]
    edits = error_correction(cable1, cable2)

    print(cable1)
    print(cable2)
    print("\nInstructions:")
    for edit in edits:
        print(edit)  # Output: List of edit operations