import pytest
import numpy as np
import sys

from src.model.cube import Cube
from src.model import scrambler, generateData

# run test: py.test src/tests/testGenerateData.py

class TestGenerateData:

    singleScramble, singleStickers = scrambler.getRandomScrambles(5) 