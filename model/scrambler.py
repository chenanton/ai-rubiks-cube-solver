# Functions related to scrambling the cube.

import random
import copy
import numpy as np

from cube import Cube  # pylint: disable=import-error
from twophase import solve

minScrambleLen = 1
maxScrambleLen = 25

fileExt = ".npy"

turns = ["D", "D2", "D'", "U", "U2", "U'", "F", "F2", "F'",
         "B", "B2", "B'", "L", "L2", "L'", "R", "R2", "R'"]


stickerToFace = {
    0: "D",
    1: "U",
    2: "F",
    3: "B",
    4: "L",
    5: "R"
}


def generateData(m, numFiles=1, filePathBase="data/trainingSets/"):
    for i in range(numFiles):
        data = getRandomScrambles(int(m / numFiles))
        print("Dataset for file #" + str(i) + " is generated.")
        np.save(filePathBase + str(i) + fileExt, data)
        print("Dataset for file #" + str(i) + " is saved.")


# Returns a list of random sticker and solution pairs
def getRandomScrambles(iterations):
    res = np.zeros((iterations, 55))  # 54 stickers + 1 solution move
    for i in range(iterations):
        lastTurn, stickers = randomScramble()
        res[i, :54] = stickers.flatten()
        res[i, 54] = lastTurn
    return res


# Generates a single random scramble and last move pair
def randomScramble():
    cube = Cube()

    # Randomly scrmble cube in range [minScrambleLen, maxScrambleLen]
    for _ in range(random.randint(minScrambleLen, maxScrambleLen)):
        index = random.randint(0, len(turns) - 1)
        cube(turns[index])

    # Get solution
    solution = solve(toStickerString(cube.stickers))
    lastMove = tokenizeSolution(solution)

    return lastMove, cube.stickers


# Converts a 6x3x3 sticker tensor into a 54 character string to pass into twophase.solve()
# Credit: tcbegley on GitHub: https://github.com/tcbegley/cube-solver
def toStickerString(stickers):
    stickerList = toTwoPhase(stickers).flatten()
    stickerList = map(indexToFace, stickerList)
    return "".join(stickerList)


# Shifts 6x3x3 tensor from default representation to twophase representation
# I.e.  1            0
#     4 2 5 3  =>  4 2 1 5
#       0            3
def toTwoPhase(stickers):
    return stickers[[1, 5, 2, 0, 4, 3], :, :]


# Converts index of sticker to face character
def indexToFace(index):
    return stickerToFace[index]


# Tokenizes solution obtained from solve() in twophase, returns first move only
def tokenizeSolution(solution):
    token = solution.split()[0]   # get only first move
    res = turns.index(token)
    return res


if __name__ == "__main__":
    solution, stickers = randomScramble()
    print(stickers)
    print(solution)
