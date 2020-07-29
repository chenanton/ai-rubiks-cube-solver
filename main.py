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

historyLoad = json.load(open("data/histories/history0.json", 'r'))
for i in range(1, 50):
    history_i = json.load(open("data/histories/history" + str(i) + ".json", 'r'))
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
plt.show()