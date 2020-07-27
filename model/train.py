import numpy as np
import tensorflow as tf
from tensorflow import keras

from generateData import generateData, generateDataMulti, inputFileBase, outputFileBase, fileExt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Loads data from specified input and output files, returns features and labels
def loadData(numFiles=0):
    X = np.load(inputFileBase + str(numFiles) + fileExt)
    Y = np.load(outputFileBase + str(numFiles) + fileExt)
    return X, Y


# Partitions data into train/dev/test sets
# Specified partition weights determine how large each set is
#       E.g. trainWeight=98, devWeight=1, testWeight=1
#           --> (XTrain, YTrain) will have 98% of input's examples,
#               (XDev, YDev) and (XTest, YTest) will have 1% respectively
def partitionData(X, Y, trainWeight=3, devWeight=1, testWeight=1):

    singleWeight = int(X.shape[0] / (trainWeight + devWeight + testWeight))

    partition1 = trainWeight * singleWeight
    partition2 = partition1 + devWeight * singleWeight

    XTrain = X[:partition1]
    YTrain = Y[:partition1]
    XDev = X[partition1:partition2]
    YDev = Y[partition1:partition2]
    XTest = X[partition2:]
    YTest = Y[partition2:]

    return (XTrain, YTrain), (XDev, YDev), (XTest, YTest)


# Hyperparameters
trainingSize = 1000
batchSize = 512
epochs = 5
numFiles = 10

maxLen = 54
hiddenSize = 128


# Defines model layers, compiles model
def createModel(Tx, Ty, inputSize, outputSize, na=128, ns=128):
    model = keras.Sequential([
        keras.layers.LSTM(units=na),
        keras.layers.RepeatVector(n=Ty),
        keras.layers.LSTM(units=ns, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(
            units=outputSize, activation="softmax"))
    ])
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return model


# Trains model
def trainModel():
    generateDataMulti(trainingSize, numFiles)

    model = createModel(Tx=54, Ty=25, inputSize=6, outputSize=12)

    for i in range(numFiles):
        Xi, Yi = loadData(i)
        (XTrain, YTrain), (XDev, YDev), (XTest,
                                         YTest) = partitionData(Xi, Yi, 98, 1, 1)
        model.fit(XTrain, YTrain, epochs=epochs,
                  batch_size=batchSize, validation_data=(XDev, YDev))
        model.evaluate(XTest, YTest)

    model.summary()
    model.save(filepath="data/model.hdf5", save_format="h5")


if __name__ == "__main__":
    trainModel()
