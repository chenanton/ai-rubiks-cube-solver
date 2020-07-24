# This file represents the model for the cube. The center white piece is always facing the bottom, and the center green piece is always facing forward.

import numpy as np
import copy

stickers = ["white", "yellow", "green", "blue", "red", "orange"]

class Cube:
    side_len = 3
    sides = np.empty([6, side_len, side_len])

    # constructor
    def __init__(self):
        for i, _ in enumerate(stickers):
            self.sides[i, :, :] = np.full([self.side_len, self.side_len], i)


    # MOVES

    # rotate front face clockwise
    def rotateF(self):
        self.sides[2] = np.rot90(self.sides[2])

        tmp = copy.deepcopy(self.sides[4, :, 2])
        self.sides[4, :, 2] = copy.deepcopy(self.sides[0, 0, :])
        self.sides[0, 0, :] = copy.deepcopy(self.sides[5, :, 0])
        self.sides[5, :, 0] = copy.deepcopy(self.sides[1, 2, :])
        self.sides[1, 2, :] = tmp

    # rotate front face counter-clockwise
    def rotateFprime(self):
        for _ in range(3):
            self.rotateF()


    


    


    
    
cube = Cube()
print(cube.sides)
print("\n")
cube.rotateF()
print(cube.sides)


