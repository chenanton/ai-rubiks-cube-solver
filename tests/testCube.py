import unittest
import numpy as np
import sys

from model.cube import Cube


# run test: py.test src/tests/testCube.py
class TestCube(unittest.TestCase):

    def setUp(self):
        self.cube1 = Cube()

    def testInit(self):
        for i in range(6):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))

    def testRotateDSolved(self):
        self.cube1.rotateD()
        for i in range(2):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))
        for i in range(2, 4):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i, :2, :], np.full((2, 3), i)))
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i, 2, :], np.full((3), i + 2)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, :2, :], np.full((2, 3), 4)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, 2, :], np.full((3), 3)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, :2, :], np.full((2, 3), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, 2, :], np.full((3), 2)))

    def testRotateD2Solved(self):
        self.cube1.rotateD2()
        for i in range(2):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, :2, :], np.full((2, 3), 2)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, 2, :], np.full((3), 3)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, :2, :], np.full((2, 3), 3)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, 2, :], np.full((3), 2)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, :2, :], np.full((2, 3), 4)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, 2, :], np.full((3), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, :2, :], np.full((2, 3), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, 2, :], np.full((3), 4)))

    def testRotateDPrimeSolved(self):
        self.cube1.rotateDprime()
        for i in range(2):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, :2, :], np.full((2, 3), 2)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, 2, :], np.full((3), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, :2, :], np.full((2, 3), 3)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, 2, :], np.full((3), 4)))
        for i in range(4, 6):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i, :2, :], np.full((2, 3), i)))
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i, 2, :], np.full((3), i - 2)))

    def testRotateUSolved(self):
        self.cube1.rotateU()
        for i in range(2):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, 1:, :], np.full((2, 3), 2)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, 0, :], np.full((3), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, 1:, :], np.full((2, 3), 3)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, 0, :], np.full((3), 4)))
        for i in range(4, 6):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i, 1:, :], np.full((2, 3), i)))
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i, 0, :], np.full((3), i - 2)))

    def testRotateUPrimeSolved(self):
        self.cube1.rotateUprime()
        for i in range(2):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))
        for i in range(2, 4):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i, 1:, :], np.full((2, 3), i)))
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i, 0, :], np.full((3), i + 2)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, 1:, :], np.full((2, 3), 4)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, 0, :], np.full((3), 3)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, 1:, :], np.full((2, 3), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, 0, :], np.full((3), 2)))

    def testRotateFSolved(self):
        self.cube1.rotateF()
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, 1:, :], np.full((2, 3), 0)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, 0, :], np.full((3), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, :2, :], np.full((2, 3), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, 2, :], np.full((3), 4)))
        for i in range(2, 4):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, :, :2], np.full((3, 2), 4)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, :, 2], np.full((3), 0)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, :, 1:], np.full((3, 2), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, :, 0], np.full((3), 1)))

    def testRotateFPrimeSolved(self):
        self.cube1.rotateFprime()
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, 1:, :], np.full((2, 3), 0)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, 0, :], np.full((3), 4)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, :2, :], np.full((2, 3), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, 2, :], np.full((3), 5)))
        for i in range(2, 4):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, :, :2], np.full((3, 2), 4)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, :, 2], np.full((3), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, :, 1:], np.full((3, 2), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, :, 0], np.full((3), 0)))

    def testRotateBSolved(self):
        self.cube1.rotateB()
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, :2, :], np.full((2, 3), 0)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, 2, :], np.full((3), 4)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, 1:, :], np.full((2, 3), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, 0, :], np.full((3), 5)))
        for i in range(2, 4):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, :, 1:], np.full((3, 2), 4)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, :, 0], np.full((3), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, :, :2], np.full((3, 2), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, :, 2], np.full((3), 0)))

    def testRotateBprimeSolved(self):
        self.cube1.rotateBprime()
        print(self.cube1.stickers[0, :2, :])
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, :2, :], np.full((2, 3), 0)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, 2, :], np.full((3), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, 1:, :], np.full((2, 3), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, 0, :], np.full((3), 4)))
        for i in range(2, 4):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, :, 1:], np.full((3, 2), 4)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, :, 0], np.full((3), 0)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, :, :2], np.full((3, 2), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, :, 2], np.full((3), 1)))

    def testRotateLSolved(self):
        self.cube1.rotateL()
        for i in range(0, 2):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i, :, 1:], np.full((3, 2), i)))
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i, :, 0], np.full((3), i + 2)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, :, 1:], np.full((3, 2), 2)))
        # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, :, 0], np.full((3), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, :, :2], np.full((3, 2), 3)))
        # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, :, 2], np.full((3), 0)))
        for i in range(4, 6):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))

    def testRotateLPrimeSolved(self):
        self.cube1.rotateLprime()
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, :, 1:], np.full((3, 2), 0)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, :, 0], np.full((3), 3)))    # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, :, 1:], np.full((3, 2), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, :, 0], np.full((3), 2)))    # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, :, 1:], np.full((3, 2), 2)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, :, 0], np.full((3), 0)))    # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, :, :2], np.full((3, 2), 3)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, :, 2], np.full((3), 1)))    # inverted
        for i in range(4, 6):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))

    def testRotateRSolved(self):
        self.cube1.rotateR()
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, :, :2], np.full((3, 2), 0)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, :, 2], np.full((3), 3)))    # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, :, :2], np.full((3, 2), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, :, 2], np.full((3), 2)))    # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, :, :2], np.full((3, 2), 2)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, :, 2], np.full((3), 0)))    # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, :, 1:], np.full((3, 2), 3)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, :, 0], np.full((3), 1)))    # inverted
        for i in range(4, 6):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))

    def testRotateRPrimeSolved(self):
        self.cube1.rotateRprime()
        for i in range(0, 2):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i, :, :2], np.full((3, 2), i)))
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i, :, 2], np.full((3), i + 2)))    # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, :, :2], np.full((3, 2), 2)))
        # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, :, 2], np.full((3), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, :, 1:], np.full((3, 2), 3)))
        # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, :, 0], np.full((3), 0)))
        for i in range(4, 6):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))


if __name__ == "__main__":
    unittest.main()
