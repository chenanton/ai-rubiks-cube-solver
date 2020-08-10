# from src.model.cube import Cube

# Continuously generates random scrambles, which are fed into the RNN

# cube = Cube()
# cube.plotCube()

# cube("R")
# cube.plotCube()

import sys
from model.cube import Cube
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

from model.train import createModel, checkpointPath

historyLoad = json.load(open("data/histories/attempt1/history0.json", 'r'))
for i in range(1, 50):
    history_i = json.load(
        open("data/histories/attempt1/history" + str(i) + ".json", 'r'))
    for key, val in history_i.items():
        historyLoad[key] += val

# print(historyLoad)

# print(historyLoad["loss"])

plt.plot(historyLoad["val_accuracy"])
plt.plot(historyLoad["accuracy"])
# plt.figaspect(3)

plt.legend(["val_accuracy", "accuracy"])
plt.title("Training and Validation Prediction Accuracy Over Time")
plt.ylabel("Prediction Accuracy")
plt.xlabel("Epochs (200k training examples per epoch)")
# plt.show()

model = createModel()
model.load_weights(checkpointPath)

X = np.load("data/features/X0.npy")
Y = np.load("data/labels/Y0.npy")

preds = model.predict(X[0:20])
for i in range(20):
    print("Prediction: " + str(preds[i]))
    print("Actual: " + str(Y[i]))

