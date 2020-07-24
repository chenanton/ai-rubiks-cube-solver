# This file represents the model for the cube. The center white piece is always facing the bottom, and the center green piece is always facing forward.

import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import colors

stickers = ["white", "yellow", "green", "blue", "red", "orange"]
cmap = colors.ListedColormap(stickers + ["black"])
# moves = [rotateD, rotateDprime, rotateF, rotateFprime, rotateU, rotateUprime, rotateB, rotateBprime, rotateL, rotateLprime, rotateR, rotateRprime]


class Cube:
    sideLen = 3
    sides = np.empty([6, sideLen, sideLen])

    # constructor
    def __init__(self):
        for i, _ in enumerate(stickers):
            self.sides[i, :, :] = np.full([self.sideLen, self.sideLen], i)

    # print cube

    def plotCube(self, title="Sticker Mapping"):
        plt.close("all")

        mt = np.full((3, 3), 7)
        top = np.concatenate((mt, self.sides[1], mt, mt), axis=1)
        mid = np.concatenate(
            (self.sides[4], self.sides[2], self.sides[5], self.sides[3]), axis=1)
        bot = np.concatenate((mt, self.sides[0], mt, mt), axis=1)
        printRepr = np.concatenate((top, mid, bot), axis=0)
        
        plt.imshow(printRepr, interpolation='nearest', cmap=cmap)
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_xticks(np.arange(-0.5, 11.5, 1))
        ax.set_yticks(np.arange(-0.5, 8.5, 1))
        plt.title(title)
        plt.tight_layout()
        plt.grid(b=True, which='major', color='#000000', linestyle='-', linewidth=4)
        plt.show()

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
cube.rotateR()
cube.rotateR()
cube.rotateUprime()
cube.rotateR()
cube.rotateUprime()
cube.rotateR()
cube.rotateU()
cube.rotateRprime()
cube.rotateU()
cube.rotateR()
cube.rotateR()
cube.rotateU()
cube.rotateDprime()
cube.rotateR()
cube.rotateUprime()
cube.rotateRprime()
cube.rotateD()
cube.plotCube()
