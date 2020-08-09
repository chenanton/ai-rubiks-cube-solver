import json
import os

# Disable tensorflow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import time

from tensorflow import keras
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, TimeDistributed, Bidirectional, Attention, Input, RepeatVector

from scrambler import generateData, fileExt


# Hyperparameters
trainingSize = 30000000
batchSize = 512
epochs = 10
numFiles = 20

XBOS = 7
XEOS = 6
YBOS = 13
YEOS = 12

modelName = "rubiks-cube-lstm-{}".format(int(time.time()))
checkpointPath = "logs/checkpoints/checkpoint.keras"

maxInputLen = 54
hiddenSize = 128


# Loads data from specified input and output files, returns features and labels
def loadData(numFiles=0, filePathBase="data/trainingSets/"):
    data = np.load(filePathBase + str(numFiles) + fileExt).astype("float32")
    
    X = data[:, :54]
    Y = data[:, 54]

    return X, Y


# Defines model layers, compiles model
def createModel():
    model = keras.Sequential([
        keras.layers.Input(shape=54),
        keras.layers.Dense(units=1024, activation="relu", name="dense0"),
        keras.layers.Dense(units=2048, activation="relu", name="dense1"),
        keras.layers.Dense(units=1024, activation="relu", name="dense2"),
        keras.layers.Dense(units=512, activation="relu", name="dense3"),
        keras.layers.Dense(units=256, activation="relu", name="dense4"),
        keras.layers.Dense(units=128, activation="relu", name="dense5"),
        keras.layers.Dense(units=64, activation="relu", name="dense6"),
        keras.layers.Dense(units=32, activation="relu", name="dense7"),
        keras.layers.Dense(units=12, activation="softmax", name="dense8"),
    ])

    # Compile model
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    return model


# Trains model
def trainModel(loadPrev=True):
    model = createModel()

    if loadPrev:
        model.load_weights(checkpointPath)

    for i in range(numFiles):
        X, Y = loadData(i)

        callbacks = getCallbacks()

        model.fit(
            x=X, y=Y, epochs=epochs, batch_size=batchSize, validation_split=0.01, callbacks=callbacks)

    model.summary()
    

# Get callbacks for model.fit()
def getCallbacks():
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpointPath, monitor="val_loss", verbose=1, save_weights_only=True, save_best_only=True)
    earlyStopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(modelName))

    return [checkpoint, earlyStopping, tensorboard]


# Predicts solution from single sticker mapping
def predict(stickers, model):
    pred = model.predict(stickers)
    return pred


if __name__ == "__main__":
    # generateData(trainingSize, numFiles=15)
    # trainModel(loadPrev=False)

    model = createModel()
    model.load_weights(checkpointPath)
    # encoderModel.load_weights(checkpointPath, by_name=True)
    # decoderModel.load_weights(checkpointPath, by_name=True)


    X, Y = loadData(10)
    X = X[:20]
    Y = Y[:20]

    print("Input: ")
    print(X)
    print("Prediction: ")
    print(np.argmax(predict(X, model), axis=-1))
    print("Actual: ")
    print(Y)

    # X, Y = loadData()
    # preds = np.argmax(model.predict(X), axis=-1)
    # print("Predictions: ")
    # print(preds)
    # print("Actual: ")
    # print(Y["decDense"].astype(int))
