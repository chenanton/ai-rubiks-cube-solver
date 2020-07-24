# This file represents the model for the cube. The center white piece is always facing the bottom, and the center green piece is always facing forward.

import numpy as np
import copy

stickers = ["white", "yellow", "green", "blue", "red", "orange"]
# moves = [rotateD, rotateDprime, rotateF, rotateFprime, rotateU, rotateUprime, rotateB, rotateBprime, rotateL, rotateLprime, rotateR, rotateRprime]


class Cube:
    sideLen = 3
    sides = np.empty([6, sideLen, sideLen])

    # constructor
    def __init__(self):
        for i, _ in enumerate(stickers):
            self.sides[i, :, :] = np.full([self.sideLen, self.sideLen], i)

    # print cube

    def printCube(self):
        mt = np.full((3, 3), 8)
        print(np.concatenate((mt, self.sides[1], mt, mt), axis=1))
        print(np.concatenate(
            (self.sides[4], self.sides[2], self.sides[5], self.sides[3]), axis=1))
        print(np.concatenate((mt, self.sides[0], mt, mt), axis=1))

    # ROTATIONS

    # rotate bottom face clockwise

    def rotateD(self):
        self.sides[0] = np.rot90(self.sides[0])

        tmp = copy.copy(self.sides[4, 2, :])
        self.sides[4, 2, :] = copy.copy(self.sides[3, 2, :])
        self.sides[3, 2, :] = copy.copy(self.sides[5, 2, :])
        self.sides[5, 2, :] = copy.copy(self.sides[2, 2, :])
        self.sides[2, 2, :] = tmp

    # rotate bottom face counter-clockwise

    def rotateDprime(self):
        for _ in range(3):
            self.rotateD()

    # rotate top face clockwise
    def rotateU(self):
        self.sides[1] = np.rot90(self.sides[1])

        tmp = copy.copy(self.sides[4, 0, :])
        self.sides[4, 0, :] = copy.copy(self.sides[2, 0, :])
        self.sides[2, 0, :] = copy.copy(self.sides[5, 0, :])
        self.sides[5, 0, :] = copy.copy(self.sides[3, 0, :])
        self.sides[3, 0, :] = tmp

    # rotate top face counter-clockwise

    def rotateUprime(self):
        for _ in range(3):
            self.rotateU()

    # rotate front face clockwise
    def rotateF(self):
        self.sides[2] = np.rot90(self.sides[2])

        tmp = copy.copy(self.sides[4, :, 2])
        self.sides[4, :, 2] = copy.copy(self.sides[0, 0, :])
        self.sides[0, 0, :] = copy.copy(self.sides[5, :, 0])
        self.sides[5, :, 0] = copy.copy(self.sides[1, 2, :])
        self.sides[1, 2, :] = tmp

    # rotate front face counter-clockwise
    def rotateFprime(self):
        for _ in range(3):
            self.rotateF()

    # rotate back face clockwise

    def rotateB(self):
        self.sides[3] = np.rot90(self.sides[3])

        tmp = copy.copy(self.sides[5, :, 2])
        self.sides[5, :, 2] = copy.copy(self.sides[0, 2, :])
        self.sides[0, 2, :] = copy.copy(self.sides[4, :, 0])
        self.sides[4, :, 0] = copy.copy(self.sides[1, 0, :])
        self.sides[1, 0, :] = tmp

    # rotate back face counter-clockwise
    def rotateBprime(self):
        for _ in range(3):
            self.rotateB()

    # rotate left face clockwise
    def rotateL(self):
        self.sides[4] = np.rot90(self.sides[4])

        tmp = copy.copy(self.sides[3, :, 2])
        self.sides[3, :, 2] = copy.copy(np.flip(self.sides[0, :, 0]))
        self.sides[0, :, 0] = copy.copy(self.sides[2, :, 0])
        self.sides[2, :, 0] = copy.copy(self.sides[1, :, 0])
        self.sides[1, :, 0] = np.flip(tmp)

    # rotate left face counter-clockwise
    def rotateLprime(self):
        for _ in range(3):
            self.rotateL()

    # rotate right face clockwise

    def rotateR(self):
        self.sides[5] = np.rot90(self.sides[5])

        tmp = copy.copy(self.sides[2, :, 2])
        self.sides[2, :, 2] = copy.copy(self.sides[0, :, 2])
        self.sides[0, :, 2] = copy.copy(np.flip(self.sides[3, :, 0]))
        self.sides[3, :, 0] = copy.copy(np.flip(self.sides[1, :, 2]))
        self.sides[1, :, 2] = tmp

    # rotate right face counter-clockwise
    def rotateRprime(self):
        for _ in range(3):
            self.rotateR()


cube = Cube()
# cube.printCube()
# print("\n")

cube.rotateD()
cube.printCube()
print("\n")
cube.rotateU()
cube.printCube()
print("\n")
cube.rotateR()
cube.printCube()
