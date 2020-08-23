# Continuously generates random scrambles, which are fed into the RNN

import sys
import os

sys.path.append("model")
sys.path.append("lib/MagicCube/code")

# Disable tensorflow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt

from model.cube import Cube
from model.train import predict
from model.scrambler import turns

from cube_interactive import Cube as UICube # pylint: disable=import-error

### Load model

# c = UICube(N=3)
# c.draw_interactive()
# plt.show()

c = Cube()

### Scramble cube
c("R")
c("U")
c("R'")
c("U'")
# c("R")
# c("B2")


soln = []

count = 0
while not c.isSolved():
    count += 1
    if count > 50:
        break
    pred = predict(c.stickers)
    soln.append(pred)
    c(pred)

print("Prediction: ")
print(soln)
print(len(soln))