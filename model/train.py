import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, TimeDistributed, Bidirectional, Attention, Input, RepeatVector
from generateData import generateData, toSparse, generateDataMulti, inputFileBase, outputFileBase, fileExt




# Loads data from specified input and output files, returns features and labels
def loadData(numFiles=0):
    encInput = np.load(inputFileBase + str(numFiles) + fileExt)
    decInput = np.load(outputFileBase + str(numFiles) + fileExt)

    decOutput = toSparse(decInput, 13)

    X = {
        "encInput": encInput,
        "decInput": decInput
    }

    Y = {
        "time_distributed": decOutput
    }

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
trainingSize = 10000000
batchSize = 512
epochs = 10
numFiles = 50

maxLen = 54
hiddenSize = 128

historyPath = "data/histories/history"


# Defines model layers, compiles model
def createModel(Tx, Ty, inputVocabLen, outputVocabLen, embedDim=128, hiddenDim=512):

    # Encoder
    encInput = Input(shape=(Tx, ), name="encInput")
    encEmbedding = Embedding(input_dim=inputVocabLen,
                             output_dim=embedDim, input_length=Tx)(encInput)

    _, h, c = LSTM(units=hiddenDim, return_state=True)(encEmbedding)

    # Decoder
    decInput = Input(shape=(Ty, ), name="decInput")
    decEmbedding = Embedding(input_dim=outputVocabLen,
                             output_dim=embedDim, input_length=Ty)(decInput)

    decLSTM, _, _ = LSTM(units=hiddenDim, return_state=True, return_sequences=True)(
        decEmbedding, initial_state=[h, c])

    decOutput = TimeDistributed(
        Dense(outputVocabLen, activation="softmax"))(decLSTM)

    model = keras.Model(inputs=[encInput, decInput], outputs=[decOutput])
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    return model


# Trains model
def trainModel():
    # generateDataMulti(trainingSize, numFiles)

    model = createModel(Tx=54, Ty=25, inputVocabLen=6, outputVocabLen=12 + 1)
    # model = keras.models.load_model("data/model.hdf5")

    for i in range(numFiles):
        X, Y = loadData(i)

        # (XTrain, YTrain), (XDev, YDev), (XTest,
        #  YTest) = partitionData(Xi, Yi, 98, 1, 1)

        # , validation_data=(XDev, YDev)
        historyModel = model.fit(
            x=X, y=Y, epochs=epochs, batch_size=batchSize, validation_split=0.02)
        # model.evaluate(XTest, YTest)

        historyDict = historyModel.history
        json.dump(historyDict, open(historyPath + str(i) + ".json", 'w'))

    model.summary()
    model.save(filepath="data/model.hdf5", save_format="h5")


if __name__ == "__main__":
    # generateDataMulti(trainingSize, totalFiles=numFiles)
    trainModel()
