# from src.model.cube import Cube

# Continuously generates random scrambles, which are fed into the RNN

import sys
from model.cube import Cube
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

from model.train import createModel, checkpointPath


