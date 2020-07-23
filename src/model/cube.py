# This file represents the model for the cube. The center white piece is always facing the bottom, and the center green piece is always facing forward.

import numpy as np

stickers = ["white", "yellow", "green", "blue", "red", "orange"]

class Cube:
    side_len = 3
    sides = np.zeros([6, side_len, side_len])

    # initializes a solved cube
    def __init__(self):
        for i in range(self.sides.shape[0]):
            self.sides[i, :, :] = np.full([self.side_len, self.side_len], i)

    
cube = Cube()
print(cube.sides)


