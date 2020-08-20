import unittest
import numpy as np
import sys

from model.cube import Cube

# Scramble pattern for cube2 test case,
#   each piece's adjacent edges are different colour
superflip = "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"


class TestCube(unittest.TestCase):

    def setUp(self):
        self.cube1 = Cube()
        self.cube2 = Cube()
        for move in superflip.split():
            self.cube2(move)

    def testInitSolved(self):
        for i in range(6):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))

    def testIsSolved(self):
        self.assertTrue(self.cube1.isSolved())

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

    def testRotateU2Solved(self):
        self.cube1.rotateU2()
        for i in range(2):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, 1:, :], np.full((2, 3), 2)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, 0, :], np.full((3), 3)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, 1:, :], np.full((2, 3), 3)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, 0, :], np.full((3), 2)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, 1:, :], np.full((2, 3), 4)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, 0, :], np.full((3), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, 1:, :], np.full((2, 3), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, 0, :], np.full((3), 4)))

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

    def testRotateF2Solved(self):
        self.cube1.rotateF2()
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, 1:, :], np.full((2, 3), 0)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, 0, :], np.full((3), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, :2, :], np.full((2, 3), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, 2, :], np.full((3), 0)))
        for i in range(2, 4):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, :, :2], np.full((3, 2), 4)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, :, 2], np.full((3), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, :, 1:], np.full((3, 2), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, :, 0], np.full((3), 4)))

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

    def testRotateB2Solved(self):
        self.cube1.rotateB2()
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, :2, :], np.full((2, 3), 0)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, 2, :], np.full((3), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, 1:, :], np.full((2, 3), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, 0, :], np.full((3), 0)))
        for i in range(2, 4):
            self.assertTrue(np.array_equal(
                self.cube1.stickers[i], np.full((3, 3), i)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, :, 1:], np.full((3, 2), 4)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[4, :, 0], np.full((3), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, :, :2], np.full((3, 2), 5)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[5, :, 2], np.full((3), 4)))

    def testRotateBprimeSolved(self):
        self.cube1.rotateBprime()
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

    def testRotateL2Solved(self):
        self.cube1.rotateL2()
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, :, 1:], np.full((3, 2), 0)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, :, 0], np.full((3), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, :, 1:], np.full((3, 2), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, :, 0], np.full((3), 0)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, :, 1:], np.full((3, 2), 2)))
        # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, :, 0], np.full((3), 3)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, :, :2], np.full((3, 2), 3)))
        # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, :, 2], np.full((3), 2)))
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

    def testRotateR2Solved(self):
        self.cube1.rotateR2()
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, :, :2], np.full((3, 2), 0)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[0, :, 2], np.full((3), 1)))    # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, :, :2], np.full((3, 2), 1)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[1, :, 2], np.full((3), 0)))    # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, :, :2], np.full((3, 2), 2)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[2, :, 2], np.full((3), 3)))    # inverted
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, :, 1:], np.full((3, 2), 3)))
        self.assertTrue(np.array_equal(
            self.cube1.stickers[3, :, 0], np.full((3), 2)))    # inverted
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

    def testInitUnsolved(self):
        pattern = np.array([
            [[0, 2, 0], [4, 0, 5], [0, 3, 0]],
            [[1, 3, 1], [4, 1, 5], [1, 2, 1]],
            [[2, 1, 2], [4, 2, 5], [2, 0, 2]],
            [[3, 1, 3], [5, 3, 4], [3, 0, 3]],
            [[4, 1, 4], [3, 4, 2], [4, 0, 4]],
            [[5, 1, 5], [2, 5, 3], [5, 0, 5]]
        ])
        self.assertEqual(pattern.tolist(), self.cube2.stickers.tolist())

    def testIsSolvedNotSolved(self):
        self.assertFalse(self.cube2.isSolved())

    def testRotateDUnsolved(self):
        self.cube2("D")
        expected = np.array([
            [[0, 4, 0], [3, 0, 2], [0, 5, 0]],
            [[1, 3, 1], [4, 1, 5], [1, 2, 1]],
            [[2, 1, 2], [4, 2, 5], [4, 0, 4]],
            [[3, 1, 3], [5, 3, 4], [5, 0, 5]],
            [[4, 1, 4], [3, 4, 2], [3, 0, 3]],
            [[5, 1, 5], [2, 5, 3], [2, 0, 2]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist())

    def testRotateD2Unsolved(self):
        self.cube2("D2")
        expected = np.array([
            [[0, 3, 0], [5, 0, 4], [0, 2, 0]],
            [[1, 3, 1], [4, 1, 5], [1, 2, 1]],
            [[2, 1, 2], [4, 2, 5], [3, 0, 3]],
            [[3, 1, 3], [5, 3, 4], [2, 0, 2]],
            [[4, 1, 4], [3, 4, 2], [5, 0, 5]],
            [[5, 1, 5], [2, 5, 3], [4, 0, 4]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist())

    def testRotateDPrimeUnsolved(self):
        self.cube2("D'")
        expected = np.array([
            [[0, 5, 0], [2, 0, 3], [0, 4, 0]],
            [[1, 3, 1], [4, 1, 5], [1, 2, 1]],
            [[2, 1, 2], [4, 2, 5], [5, 0, 5]],
            [[3, 1, 3], [5, 3, 4], [4, 0, 4]],
            [[4, 1, 4], [3, 4, 2], [2, 0, 2]],
            [[5, 1, 5], [2, 5, 3], [3, 0, 3]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist())

    def testRotateUUnsolved(self):
        self.cube2("U")
        expected = np.array([
            [[0, 2, 0], [4, 0, 5], [0, 3, 0]],
            [[1, 4, 1], [2, 1, 3], [1, 5, 1]],
            [[5, 1, 5], [4, 2, 5], [2, 0, 2]],
            [[4, 1, 4], [5, 3, 4], [3, 0, 3]],
            [[2, 1, 2], [3, 4, 2], [4, 0, 4]],
            [[3, 1, 3], [2, 5, 3], [5, 0, 5]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist())

    def testRotateU2Unsolved(self):
        self.cube2("U2")
        expected = np.array([
            [[0, 2, 0], [4, 0, 5], [0, 3, 0]],
            [[1, 2, 1], [5, 1, 4], [1, 3, 1]],
            [[3, 1, 3], [4, 2, 5], [2, 0, 2]],
            [[2, 1, 2], [5, 3, 4], [3, 0, 3]],
            [[5, 1, 5], [3, 4, 2], [4, 0, 4]],
            [[4, 1, 4], [2, 5, 3], [5, 0, 5]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist())

    def testRotateUPrimeUnsolved(self):
        self.cube2("U'")
        expected = np.array([
            [[0, 2, 0], [4, 0, 5], [0, 3, 0]],
            [[1, 5, 1], [3, 1, 2], [1, 4, 1]],
            [[4, 1, 4], [4, 2, 5], [2, 0, 2]],
            [[5, 1, 5], [5, 3, 4], [3, 0, 3]],
            [[3, 1, 3], [3, 4, 2], [4, 0, 4]],
            [[2, 1, 2], [2, 5, 3], [5, 0, 5]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist())

    def testRotateFUnsolved(self):
        self.cube2("F")
        expected = np.array([
            [[5, 2, 5], [4, 0, 5], [0, 3, 0]],
            [[1, 3, 1], [4, 1, 5], [4, 2, 4]],
            [[2, 4, 2], [0, 2, 1], [2, 5, 2]],
            [[3, 1, 3], [5, 3, 4], [3, 0, 3]],
            [[4, 1, 0], [3, 4, 2], [4, 0, 0]],
            [[1, 1, 5], [2, 5, 3], [1, 0, 5]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist()) 

    def testRotateF2Unsolved(self):
        self.cube2("F2")
        expected = np.array([
            [[1, 2, 1], [4, 0, 5], [0, 3, 0]],
            [[1, 3, 1], [4, 1, 5], [0, 2, 0]],
            [[2, 0, 2], [5, 2, 4], [2, 1, 2]],
            [[3, 1, 3], [5, 3, 4], [3, 0, 3]],
            [[4, 1, 5], [3, 4, 2], [4, 0, 5]],
            [[4, 1, 5], [2, 5, 3], [4, 0, 5]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist())   

    def testRotateFPrimeUnsolved(self):
        self.cube2("F'")
        expected = np.array([
            [[4, 2, 4], [4, 0, 5], [0, 3, 0]],
            [[1, 3, 1], [4, 1, 5], [5, 2, 5]],
            [[2, 5, 2], [1, 2, 0], [2, 4, 2]],
            [[3, 1, 3], [5, 3, 4], [3, 0, 3]],
            [[4, 1, 1], [3, 4, 2], [4, 0, 1]],
            [[0, 1, 5], [2, 5, 3], [0, 0, 5]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist()) 

    def testRotateBUnsolved(self):
        self.cube2("B")
        expected = np.array([
            [[0, 2, 0], [4, 0, 5], [4, 3, 4]],
            [[5, 3, 5], [4, 1, 5], [1, 2, 1]],
            [[2, 1, 2], [4, 2, 5], [2, 0, 2]],
            [[3, 5, 3], [0, 3, 1], [3, 4, 3]],
            [[1, 1, 4], [3, 4, 2], [1, 0, 4]],
            [[5, 1, 0], [2, 5, 3], [5, 0, 0]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist()) 

    def testRotateB2Unsolved(self):
        self.cube2("B2")
        expected = np.array([
            [[0, 2, 0], [4, 0, 5], [1, 3, 1]],
            [[0, 3, 0], [4, 1, 5], [1, 2, 1]],
            [[2, 1, 2], [4, 2, 5], [2, 0, 2]],
            [[3, 0, 3], [4, 3, 5], [3, 1, 3]],
            [[5, 1, 4], [3, 4, 2], [5, 0, 4]],
            [[5, 1, 4], [2, 5, 3], [5, 0, 4]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist()) 

    def testRotateBPrimeUnsolved(self):
        self.cube2("B'")
        expected = np.array([
            [[0, 2, 0], [4, 0, 5], [5, 3, 5]],
            [[4, 3, 4], [4, 1, 5], [1, 2, 1]],
            [[2, 1, 2], [4, 2, 5], [2, 0, 2]],
            [[3, 4, 3], [1, 3, 0], [3, 5, 3]],
            [[0, 1, 4], [3, 4, 2], [0, 0, 4]],
            [[5, 1, 1], [2, 5, 3], [5, 0, 1]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist()) 

    def testRotateLUnsolved(self):
        self.cube2("L")
        expected = np.array([
            [[2, 2, 0], [4, 0, 5], [2, 3, 0]],
            [[3, 3, 1], [4, 1, 5], [3, 2, 1]],
            [[1, 1, 2], [4, 2, 5], [1, 0, 2]],
            [[3, 1, 0], [5, 3, 4], [3, 0, 0]],
            [[4, 3, 4], [0, 4, 1], [4, 2, 4]],
            [[5, 1, 5], [2, 5, 3], [5, 0, 5]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist()) 

    def testRotateL2Unsolved(self):
        self.cube2("L2")
        expected = np.array([
            [[1, 2, 0], [4, 0, 5], [1, 3, 0]],
            [[0, 3, 1], [4, 1, 5], [0, 2, 1]],
            [[3, 1, 2], [4, 2, 5], [3, 0, 2]],
            [[3, 1, 2], [5, 3, 4], [3, 0, 2]],
            [[4, 0, 4], [2, 4, 3], [4, 1, 4]],
            [[5, 1, 5], [2, 5, 3], [5, 0, 5]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist())

    def testRotateLPrimeUnsolved(self):
        self.cube2("L'")
        expected = np.array([
            [[3, 2, 0], [4, 0, 5], [3, 3, 0]],
            [[2, 3, 1], [4, 1, 5], [2, 2, 1]],
            [[0, 1, 2], [4, 2, 5], [0, 0, 2]],
            [[3, 1, 1], [5, 3, 4], [3, 0, 1]],
            [[4, 2, 4], [1, 4, 0], [4, 3, 4]],
            [[5, 1, 5], [2, 5, 3], [5, 0, 5]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist()) 

    def testRotateRUnsolved(self):
        self.cube2("R")
        expected = np.array([
            [[0, 2, 3], [4, 0, 5], [0, 3, 3]],
            [[1, 3, 2], [4, 1, 5], [1, 2, 2]],
            [[2, 1, 0], [4, 2, 5], [2, 0, 0]],
            [[1, 1, 3], [5, 3, 4], [1, 0, 3]],
            [[4, 1, 4], [3, 4, 2], [4, 0, 4]],
            [[5, 2, 5], [0, 5, 1], [5, 3, 5]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist()) 

    def testRotateR2Unsolved(self):
        self.cube2("R2")
        expected = np.array([
            [[0, 2, 1], [4, 0, 5], [0, 3, 1]],
            [[1, 3, 0], [4, 1, 5], [1, 2, 0]],
            [[2, 1, 3], [4, 2, 5], [2, 0, 3]],
            [[2, 1, 3], [5, 3, 4], [2, 0, 3]],
            [[4, 1, 4], [3, 4, 2], [4, 0, 4]],
            [[5, 0, 5], [3, 5, 2], [5, 1, 5]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist()) 

    def testRotateRPrimeUnsolved(self):
        self.cube2("R'")
        expected = np.array([
            [[0, 2, 2], [4, 0, 5], [0, 3, 2]],
            [[1, 3, 3], [4, 1, 5], [1, 2, 3]],
            [[2, 1, 1], [4, 2, 5], [2, 0, 1]],
            [[0, 1, 3], [5, 3, 4], [0, 0, 3]],
            [[4, 1, 4], [3, 4, 2], [4, 0, 4]],
            [[5, 3, 5], [1, 5, 0], [5, 2, 5]]
        ])
        self.assertEqual(expected.tolist(),
                         self.cube2.stickers.astype(int).tolist()) 

if __name__ == "__main__":
    unittest.main()
