import pytest
import numpy as np
import sys

from src.model.cube import Cube


class TestCube:

    # @pytest.mark.one
    def testInit(self):
        cube = Cube()
        for i in range(6):
            assert(np.array_equal(cube.sides[i], np.full((3, 3), i)))
