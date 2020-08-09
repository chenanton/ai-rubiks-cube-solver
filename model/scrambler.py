# Functions related to scrambling the cube.

import random
import copy
import numpy as np

from cube import Cube  # pylint: disable=import-error
from twophase import solve

minScrambleLen = 1
maxScrambleLen = 25

fileExt = ".npy"

turns = ["D", "D'", "U", "U'", "F", "F'", "B", "B'", "L", "L'", "R", "R'"]
# turnsToIndex = dict(zip(turns, list(range(len(turns)))))


def generateData(m, numFiles=1, filePathBase="data/trainingSets/"):
    for i in range(numFiles):
        data = getRandomScrambles(int(m / numFiles))
        print("Dataset for file #" + str(i) + " is generated.")
        np.save(filePathBase + str(i) + fileExt, data)
        print("Dataset for file #" + str(i) + " is saved.")


# Returns a list of random sticker and solution pairs
def getRandomScrambles(iterations):
    res = np.zeros((iterations, 55)) # 54 stickers + 1 solution move
    for i in range(iterations):
        lastTurn, stickers = randomScramble()
        res[i, :54] = stickers.flatten()
        res[i, 54] = lastTurn
    return res


# Generates a single random scramble and last move pair
def randomScramble():
    cube = Cube()
    prevMove, prevMoveCnt = -1, 0

    # Randomly scrmble cube in range [minScrambleLen, maxScrambleLen]
    for _ in range(random.randint(minScrambleLen, maxScrambleLen)):
        index = random.randint(0, len(turns) - 1)
        while (index == prevMove + (1 if prevMove % 2 == 0 else -1)) or prevMoveCnt == 2:
            index = random.randint(0, len(turns) - 1)
            if index != prevMove:
                prevMoveCnt = 1

        cube(turns[index])
        prevMoveCnt = 1 if prevMove != index else prevMoveCnt + 1
        prevMove = index

    # Get solution
    solution = solve(toStickerString(cube.stickers))

    return solution, cube.stickers


# Converts a 6x3x3 sticker tensor into a 54 character string to pass into twophase.solve()
# Credit: tcbegley on GitHub: https://github.com/tcbegley/cube-solver
def toStickerString(stickers):
    print("BEFORE:")
    print(stickers)
    stickerList = toTwoPhase(stickers)
    print("\nAFTER:")
    print(stickerList)
    print(flattenStickers(stickerList))
    stickerList = map(indexToFace, flattenStickers(stickerList))
    print("\nFINAL:")
    print(flattenStickers(stickers))
    return "".join(stickerList)


# Converts index of sticker to face character
def indexToFace(index):
    stickerToFace = {
        0: "D",
        1: "U",
        2: "F", 
        3: "B",
        4: "L",
        5: "R"
    }
    return stickerToFace[index]


# Flattens sticker patterns; changes stickers to np array
# Output dimensions: (number of sticker layouts, 54)
def flattenStickers(stickers):
    stickers = stickers.flatten()
    return stickers


# Shifts 6x3x3 tensor from default representation to twophase representation
# I.e.  1            0
#     4 2 5 3  =>  4 2 1 5
#       0            3
def toTwoPhase(stickers):
    return stickers[[1, 5, 2, 0, 4, 3], :, :]


if __name__ == "__main__":
    solution, stickers = randomScramble()
    print(stickers)
    print(solution)