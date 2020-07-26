import pytest
import numpy as np

import sys
from model.cube import Cube

# from model.generateData import padScrambles, toSparse, flattenStickers

# run test: py.test src/tests/testGenerateData.py


class TestGenerateData:

    singleScramble = None
    multipleScramble = None
    singleStickers = None
    multipleStickers = None


    @pytest.fixture(autouse=True)
    def setup(self):
        pass
        # singleScramble, singleStickers = scrambler.getRandomScrambles(1)
        # multipleScramble, multipleStickers = scrambler.getRandomScrambles(5)


    def testPadScramblesSingle(self):
        pass


    def testPadScramblesMultiple(self):
        pass


    def testToSparseSingle(self):
        pass


    def testToSparseMultiple(self):
        pass


    def testFlattenStickersSingle(self):
        pass


    def testFlattenStickersMultiple(self):
        pass

