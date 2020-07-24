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
        self.cube = Cube()

    def testInit(self):
        for i in range(6):
            assert(np.array_equal(self.cube1.sides[i], np.full((3, 3), i)))

    def testRotateDSolved(self):
        self.cube1.rotateD()
        for i in range(2):
            assert(np.array_equal(self.cube1.sides[i], np.full((3, 3), i)))
        for i in range(2, 4):
            assert(np.array_equal(
                self.cube1.sides[i, :2, :], np.full((2, 3), i)))
            assert(np.array_equal(
                self.cube1.sides[i, 2, :], np.full((3), i + 2)))
        assert(np.array_equal(self.cube1.sides[4, :2, :], np.full((2, 3), 4)))
        assert(np.array_equal(self.cube1.sides[4, 2, :], np.full((3), 3)))
        assert(np.array_equal(self.cube1.sides[5, :2, :], np.full((2, 3), 5)))
        assert(np.array_equal(self.cube1.sides[5, 2, :], np.full((3), 2)))

    def testRotateDPrimeSolved(self):
        self.cube1.rotateDprime()
        for i in range(2):
            assert(np.array_equal(self.cube1.sides[i], np.full((3, 3), i)))
        assert(np.array_equal(self.cube1.sides[2, :2, :], np.full((2, 3), 2)))
        assert(np.array_equal(self.cube1.sides[2, 2, :], np.full((3), 5)))
        assert(np.array_equal(self.cube1.sides[3, :2, :], np.full((2, 3), 3)))
        assert(np.array_equal(self.cube1.sides[3, 2, :], np.full((3), 4)))
        for i in range(4, 6):
            assert(np.array_equal(
                self.cube1.sides[i, :2, :], np.full((2, 3), i)))
            assert(np.array_equal(
                self.cube1.sides[i, 2, :], np.full((3), i - 2)))

    def testRotateUSolved(self):
        self.cube1.rotateU()
        for i in range(2):
            assert(np.array_equal(self.cube1.sides[i], np.full((3, 3), i)))
        assert(np.array_equal(self.cube1.sides[2, 1:3, :], np.full((2, 3), 2)))
        assert(np.array_equal(self.cube1.sides[2, 0, :], np.full((3), 5)))
        assert(np.array_equal(self.cube1.sides[3, 1:3, :], np.full((2, 3), 3)))
        assert(np.array_equal(self.cube1.sides[3, 0, :], np.full((3), 4)))
        for i in range(4, 6):
            assert(np.array_equal(
                self.cube1.sides[i, 1:3, :], np.full((2, 3), i)))
            assert(np.array_equal(
                self.cube1.sides[i, 0, :], np.full((3), i - 2)))

    def testRotateUPrimeSolved(self):
        self.cube1.rotateUprime()
        for i in range(2):
            assert(np.array_equal(self.cube1.sides[i], np.full((3, 3), i)))
        for i in range(2, 4):
            assert(np.array_equal(
                self.cube1.sides[i, 1:3, :], np.full((2, 3), i)))
            assert(np.array_equal(
                self.cube1.sides[i, 0, :], np.full((3), i + 2)))
        assert(np.array_equal(self.cube1.sides[4, 1:3, :], np.full((2, 3), 4)))
        assert(np.array_equal(self.cube1.sides[4, 0, :], np.full((3), 3)))
        assert(np.array_equal(self.cube1.sides[5, 1:3, :], np.full((2, 3), 5)))
        assert(np.array_equal(self.cube1.sides[5, 0, :], np.full((3), 2)))

    # def testRotateFSolved(self):
    #     self.cube1.rotateF()
        

    #     for i in range(2, 4):
    #         assert(np.array_equal(self.cube1.sides[i], np.full((3, 3), i)))
            