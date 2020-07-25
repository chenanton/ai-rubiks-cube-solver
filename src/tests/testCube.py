import pytest
import numpy as np
import sys

from src.model.cube import Cube


# run test: py.test src/tests/testCube.py
class TestCube:
    cube1 = Cube()  # solved
    cube2 = Cube()  # scrambled

    @pytest.fixture(autouse=True)
    def setup(self):
        self.cube1 = Cube()

    def testInit(self):
        for i in range(6):
            assert(np.array_equal(self.cube1.stickers[i], np.full((3, 3), i)))

    def testRotateDSolved(self):
        self.cube1.rotateD()
        for i in range(2):
            assert(np.array_equal(self.cube1.stickers[i], np.full((3, 3), i)))
        for i in range(2, 4):
            assert(np.array_equal(
                self.cube1.stickers[i, :2, :], np.full((2, 3), i)))
            assert(np.array_equal(
                self.cube1.stickers[i, 2, :], np.full((3), i + 2)))
        assert(np.array_equal(self.cube1.stickers[4, :2, :], np.full((2, 3), 4)))
        assert(np.array_equal(self.cube1.stickers[4, 2, :], np.full((3), 3)))
        assert(np.array_equal(self.cube1.stickers[5, :2, :], np.full((2, 3), 5)))
        assert(np.array_equal(self.cube1.stickers[5, 2, :], np.full((3), 2)))

    def testRotateDPrimeSolved(self):
        self.cube1.rotateDprime()
        for i in range(2):
            assert(np.array_equal(self.cube1.stickers[i], np.full((3, 3), i)))
        assert(np.array_equal(self.cube1.stickers[2, :2, :], np.full((2, 3), 2)))
        assert(np.array_equal(self.cube1.stickers[2, 2, :], np.full((3), 5)))
        assert(np.array_equal(self.cube1.stickers[3, :2, :], np.full((2, 3), 3)))
        assert(np.array_equal(self.cube1.stickers[3, 2, :], np.full((3), 4)))
        for i in range(4, 6):
            assert(np.array_equal(
                self.cube1.stickers[i, :2, :], np.full((2, 3), i)))
            assert(np.array_equal(
                self.cube1.stickers[i, 2, :], np.full((3), i - 2)))

    def testRotateUSolved(self):
        self.cube1.rotateU()
        for i in range(2):
            assert(np.array_equal(self.cube1.stickers[i], np.full((3, 3), i)))
        assert(np.array_equal(self.cube1.stickers[2, 1:, :], np.full((2, 3), 2)))
        assert(np.array_equal(self.cube1.stickers[2, 0, :], np.full((3), 5)))
        assert(np.array_equal(self.cube1.stickers[3, 1:, :], np.full((2, 3), 3)))
        assert(np.array_equal(self.cube1.stickers[3, 0, :], np.full((3), 4)))
        for i in range(4, 6):
            assert(np.array_equal(
                self.cube1.stickers[i, 1:, :], np.full((2, 3), i)))
            assert(np.array_equal(
                self.cube1.stickers[i, 0, :], np.full((3), i - 2)))

    def testRotateUPrimeSolved(self):
        self.cube1.rotateUprime()
        for i in range(2):
            assert(np.array_equal(self.cube1.stickers[i], np.full((3, 3), i)))
        for i in range(2, 4):
            assert(np.array_equal(
                self.cube1.stickers[i, 1:, :], np.full((2, 3), i)))
            assert(np.array_equal(
                self.cube1.stickers[i, 0, :], np.full((3), i + 2)))
        assert(np.array_equal(self.cube1.stickers[4, 1:, :], np.full((2, 3), 4)))
        assert(np.array_equal(self.cube1.stickers[4, 0, :], np.full((3), 3)))
        assert(np.array_equal(self.cube1.stickers[5, 1:, :], np.full((2, 3), 5)))
        assert(np.array_equal(self.cube1.stickers[5, 0, :], np.full((3), 2)))

    def testRotateFSolved(self):
        self.cube1.rotateF()
        assert(np.array_equal(self.cube1.stickers[0, 1:, :], np.full((2, 3), 0)))
        assert(np.array_equal(self.cube1.stickers[0, 0, :], np.full((3), 5)))
        assert(np.array_equal(self.cube1.stickers[1, :2, :], np.full((2, 3), 1)))
        assert(np.array_equal(self.cube1.stickers[1, 2, :], np.full((3), 4)))
        for i in range(2, 4):
            assert(np.array_equal(self.cube1.stickers[i], np.full((3, 3), i)))
        assert(np.array_equal(self.cube1.stickers[4, :, :2], np.full((3, 2), 4)))
        assert(np.array_equal(self.cube1.stickers[4, :, 2], np.full((3), 0)))
        assert(np.array_equal(self.cube1.stickers[5, :, 1:], np.full((3, 2), 5)))
        assert(np.array_equal(self.cube1.stickers[5, :, 0], np.full((3), 1)))

    def testRotateFPrimeSolved(self):
        self.cube1.rotateFprime()
        assert(np.array_equal(self.cube1.stickers[0, 1:, :], np.full((2, 3), 0)))
        assert(np.array_equal(self.cube1.stickers[0, 0, :], np.full((3), 4)))
        assert(np.array_equal(self.cube1.stickers[1, :2, :], np.full((2, 3), 1)))
        assert(np.array_equal(self.cube1.stickers[1, 2, :], np.full((3), 5)))
        for i in range(2, 4):
            assert(np.array_equal(self.cube1.stickers[i], np.full((3, 3), i)))
        assert(np.array_equal(self.cube1.stickers[4, :, :2], np.full((3, 2), 4)))
        assert(np.array_equal(self.cube1.stickers[4, :, 2], np.full((3), 1)))
        assert(np.array_equal(self.cube1.stickers[5, :, 1:], np.full((3, 2), 5)))
        assert(np.array_equal(self.cube1.stickers[5, :, 0], np.full((3), 0)))

    def testRotateBSolved(self):
        self.cube1.rotateB()
        assert(np.array_equal(self.cube1.stickers[0, :2, :], np.full((2, 3), 0)))
        assert(np.array_equal(self.cube1.stickers[0, 2, :], np.full((3), 4)))
        assert(np.array_equal(self.cube1.stickers[1, 1:, :], np.full((2, 3), 1)))
        assert(np.array_equal(self.cube1.stickers[1, 0, :], np.full((3), 5)))
        for i in range(2, 4):
            assert(np.array_equal(self.cube1.stickers[i], np.full((3, 3), i)))
        assert(np.array_equal(self.cube1.stickers[4, :, 1:], np.full((3, 2), 4)))
        assert(np.array_equal(self.cube1.stickers[4, :, 0], np.full((3), 1)))
        assert(np.array_equal(self.cube1.stickers[5, :, :2], np.full((3, 2), 5)))
        assert(np.array_equal(self.cube1.stickers[5, :, 2], np.full((3), 0)))

    def testRotateBprimeSolved(self):
        self.cube1.rotateBprime()
        print(self.cube1.stickers[0, :2, :])
        assert(np.array_equal(self.cube1.stickers[0, :2, :], np.full((2, 3), 0)))
        assert(np.array_equal(self.cube1.stickers[0, 2, :], np.full((3), 5)))
        assert(np.array_equal(self.cube1.stickers[1, 1:, :], np.full((2, 3), 1)))
        assert(np.array_equal(self.cube1.stickers[1, 0, :], np.full((3), 4)))
        for i in range(2, 4):
            assert(np.array_equal(self.cube1.stickers[i], np.full((3, 3), i)))
        assert(np.array_equal(self.cube1.stickers[4, :, 1:], np.full((3, 2), 4)))
        assert(np.array_equal(self.cube1.stickers[4, :, 0], np.full((3), 0)))
        assert(np.array_equal(self.cube1.stickers[5, :, :2], np.full((3, 2), 5)))
        assert(np.array_equal(self.cube1.stickers[5, :, 2], np.full((3), 1)))

    def testRotateLSolved(self):
        self.cube1.rotateL()
        for i in range(0, 2):
            assert(np.array_equal(
                self.cube1.stickers[i, :, 1:], np.full((3, 2), i)))
            assert(np.array_equal(
                self.cube1.stickers[i, :, 0], np.full((3), i + 2)))
        assert(np.array_equal(self.cube1.stickers[2, :, 1:], np.full((3, 2), 2)))
        # inverted
        assert(np.array_equal(self.cube1.stickers[2, :, 0], np.full((3), 1)))
        assert(np.array_equal(self.cube1.stickers[3, :, :2], np.full((3, 2), 3)))
        # inverted
        assert(np.array_equal(self.cube1.stickers[3, :, 2], np.full((3), 0)))
        for i in range(4, 6):
            assert(np.array_equal(self.cube1.stickers[i], np.full((3, 3), i)))

    def testRotateLPrimeSolved(self):
        self.cube1.rotateLprime()
        assert(np.array_equal(
            self.cube1.stickers[0, :, 1:], np.full((3, 2), 0)))
        assert(np.array_equal(
            self.cube1.stickers[0, :, 0], np.full((3), 3)))    # inverted
        assert(np.array_equal(
            self.cube1.stickers[1, :, 1:], np.full((3, 2), 1)))
        assert(np.array_equal(
            self.cube1.stickers[1, :, 0], np.full((3), 2)))    # inverted
        assert(np.array_equal(
            self.cube1.stickers[2, :, 1:], np.full((3, 2), 2)))
        assert(np.array_equal(
            self.cube1.stickers[2, :, 0], np.full((3), 0)))    # inverted
        assert(np.array_equal(
            self.cube1.stickers[3, :, :2], np.full((3, 2), 3)))
        assert(np.array_equal(
            self.cube1.stickers[3, :, 2], np.full((3), 1)))    # inverted
        for i in range(4, 6):
            assert(np.array_equal(self.cube1.stickers[i], np.full((3, 3), i)))

    def testRotateRSolved(self):
        self.cube1.rotateR()
        assert(np.array_equal(
            self.cube1.stickers[0, :, :2], np.full((3, 2), 0)))
        assert(np.array_equal(
            self.cube1.stickers[0, :, 2], np.full((3), 3)))    # inverted
        assert(np.array_equal(
            self.cube1.stickers[1, :, :2], np.full((3, 2), 1)))
        assert(np.array_equal(
            self.cube1.stickers[1, :, 2], np.full((3), 2)))    # inverted
        assert(np.array_equal(
            self.cube1.stickers[2, :, :2], np.full((3, 2), 2)))
        assert(np.array_equal(
            self.cube1.stickers[2, :, 2], np.full((3), 0)))    # inverted
        assert(np.array_equal(
            self.cube1.stickers[3, :, 1:], np.full((3, 2), 3)))
        assert(np.array_equal(
            self.cube1.stickers[3, :, 0], np.full((3), 1)))    # inverted
        for i in range(4, 6):
            assert(np.array_equal(self.cube1.stickers[i], np.full((3, 3), i)))

    def testRotateRPrimeSolved(self):
        self.cube1.rotateRprime()
        for i in range(0, 2):
            assert(np.array_equal(
                self.cube1.stickers[i, :, :2], np.full((3, 2), i)))
            assert(np.array_equal(
                self.cube1.stickers[i, :, 2], np.full((3), i + 2)))    # inverted
        assert(np.array_equal(self.cube1.stickers[2, :, :2], np.full((3, 2), 2)))
        # inverted
        assert(np.array_equal(self.cube1.stickers[2, :, 2], np.full((3), 1)))
        assert(np.array_equal(self.cube1.stickers[3, :, 1:], np.full((3, 2), 3)))
        # inverted
        assert(np.array_equal(self.cube1.stickers[3, :, 0], np.full((3), 0)))
        for i in range(4, 6):
            assert(np.array_equal(self.cube1.stickers[i], np.full((3, 3), i)))
