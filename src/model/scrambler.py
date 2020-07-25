# Functions related to scrambling the cube.

import random
import copy

from cube import Cube  # pylint: disable=import-error

minScrambleLen = 15
maxScrambleLen = 25

turns = ["D", "D'", "U", "U'", "F", "F'", "B", "B'", "L", "L'", "R", "R'"]
# turnsToIndex = dict(zip(turns, list(range(len(turns)))))


def getRandomScrambles(iterations):
    scrambles = []
    stickerList = []
    for _ in range(iterations):
        scramble, stickers = randomScramble()
        scrambles.append(scramble)
        stickerList.append(stickers)
    return scrambles, stickerList


def randomScramble():
    cube = Cube()
    scramble = []
    prevMove, prevMoveCnt = -1, 0

    for _ in range(random.randint(minScrambleLen, maxScrambleLen)):
        index = random.randint(0, len(turns) - 1)
        while (index == prevMove + (1 if prevMove % 2 == 0 else -1)) or prevMoveCnt == 2:
            index = random.randint(0, len(turns) - 1)
            if index != prevMove:
                prevMoveCnt = 1

        cube(turns[index])
        scramble.append(index)
        prevMoveCnt = 1 if prevMove != index else prevMoveCnt + 1
        prevMove = index
    # cube.plotCube(title="Solution: " + " ".join([turns[x] for x in scramble]))
    return scramble, cube.stickers


def getSolution(scramble):
    cp = copy.deepcopy(scramble)
    cp.reverse()
    return [x + (1 if x % 2 == 0 else -1) for x in cp]


if __name__ == "__main__":
    scramble, _ = randomScramble()
    solution = getSolution(scramble)

    # scramble = [turns[x] for x in scramble]
    # solution = [turns[x] for x in solution]

    print(scramble)
    print(solution)