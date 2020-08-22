import unittest
import numpy as np
import sys
import os

sys.path.append("model")

from model.cube import Cube
import model.scrambler as scrambler

superflip = "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"

class TestScrambler(unittest.TestCase):

    def testGenerateData(self):
        numFiles = 4
        filePathBase = "tests/testData/"
        scrambler.generateData(200, numFiles=numFiles, filePathBase=filePathBase)
        for i in range(numFiles):
            data = np.load(filePathBase + str(i) + ".npy")
            self.assertEqual((50, 55), data.shape)

    def testGetRandomScrambles(self):
        scrambles = scrambler.getRandomScrambles(100)
        self.assertEqual(100, scrambles.shape[0])
        self.assertEqual(55, scrambles.shape[1])

    def testRandomScrambles(self):
        randomScrambles = scrambler.randomScrambles()
        self.assertEqual(55, randomScrambles.shape[1])

    def testGetSolution(self):
        for _ in range(10):
            cube = Cube()
            cube = scrambler._randomlyScrambleCube(cube)
            solution = scrambler._getSolution(cube)
            for move in solution.split():
                cube(move)
            self.assertTrue(cube.isSolved())

    def testRandomlyScrambleCube(self):
        cube = Cube()
        cube = scrambler._randomlyScrambleCube(cube)
        self.assertFalse(cube.isSolved())

    def testToStickerStringSolved(self):
        cube = Cube()
        stickers = scrambler._toStickerString(cube.stickers)
        self.assertEqual(stickers, "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB")

    def testToStickerStringUnsolved(self):
        cube = Cube()
        for move in superflip.split():
            cube(move)
        expected = "UBULURUFURURFRBRDRFUFLFRFDFDFDLDRDBDLULBLFLDLBUBRBLBDB"
        stickers = scrambler._toStickerString(cube.stickers)
        self.assertEqual(stickers, expected)


if __name__ == "__main__":
    unittest.main()