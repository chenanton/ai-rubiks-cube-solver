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

from generateData import generateData, toSparse, generateDataMulti, inputFileBase, outputFileBase, fileExt, fillInt, stickerLen, turnLen
from scrambler import maxScrambleLen


# Hyperparameters
trainingSize = 10000000
batchSize = 512
epochs = 1
numFiles = 5

XBOS = 7
XEOS = 6
YBOS = 13
YEOS = 12

modelName = "rubiks-cube-lstm-{}".format(int(time.time()))
checkpointPath = "logs/checkpoints/checkpoint.keras"

maxInputLen = 54
hiddenSize = 128


# Loads data from specified input and output files, returns features and labels
def loadData(numFiles=0):
    encInput = np.load(inputFileBase + str(numFiles) + fileExt)
    decInput = np.load(outputFileBase + str(numFiles) + fileExt)

    encInput, decInput = addStartEnd(encInput, decInput)
    truncateInputPadding = np.full((decInput.shape[0], 1), fill_value=YEOS)
    decTruncatedInput = np.concatenate((decInput[:, 1:], truncateInputPadding), axis=1) 
    X = {
        "encInput": encInput,
        "decInput": decInput
    }

    Y = {
        "decDense": decTruncatedInput
    }

    return X, Y


# Adds start and end characters to data:
#   Features start: 7
#   Features end: 6
#   Labels start: 13
#   Labels end: 12
def addStartEnd(features, labels):
    featuresStart = np.full((features.shape[0], 1), fill_value=XBOS)
    featuresEnd = np.full((features.shape[0], 1), fill_value=XEOS)
    features = np.concatenate((featuresStart, features, featuresEnd), axis=1)

    labelsStart = np.full((labels.shape[0], 1), fill_value=YBOS)
    labelsEnd = np.full((labels.shape[0], 1), fill_value=YEOS)
    labels = np.concatenate((labelsStart, labels, labelsEnd), axis=1)

    return features, labels


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


# Creates encoder layers
def createEncoderLayers(Tx, inputVocabLen, embedDim=128, hiddenDim=512):
    layers = {}

    encInput = Input(shape=(Tx, ), name="encInput")
    encEmbedding = Embedding(input_dim=inputVocabLen,
                             output_dim=embedDim, input_length=Tx, name="encEmbedding")
    encLSTM0 = LSTM(units=hiddenDim, return_sequences=True, name="encLSTM0")
    encLSTM1 = LSTM(units=hiddenDim, return_sequences=True, name="encLSTM1")
    encLSTM2 = LSTM(units=hiddenDim, return_state=True, name="encLSTM2")

    layers["encInput"] = encInput
    layers["encEmbedding"] = encEmbedding
    layers["encLSTM0"] = encLSTM0
    layers["encLSTM1"] = encLSTM1
    layers["encLSTM2"] = encLSTM2

    return layers


# Connects encoder layers together
def connectEncoder(layers):
    net = layers["encInput"]
    net = layers["encEmbedding"](net)
    net = layers["encLSTM0"](net)
    net = layers["encLSTM1"](net)
    _, h, c = layers["encLSTM2"](net)

    encOutput = [h, c]
    return encOutput


# Creates decoder layers
def createDecoderLayers(Ty, outputVocabLen, embedDim=128, hiddenDim=512):
    layers = {}

    decInput = Input(shape=(Ty, ), name="decInput")
    decInitialStateH = Input(shape=(hiddenDim, ), name="decInitialStateH")
    decInitialStateC = Input(shape=(hiddenDim, ), name="decInitialStateC")

    decEmbedding = Embedding(input_dim=outputVocabLen,
                             output_dim=embedDim, input_length=Ty, name="decEmbedding")
    decLSTM0 = LSTM(units=hiddenDim, return_sequences=True, name="decLSTM0")
    decLSTM1 = LSTM(units=hiddenDim, return_sequences=True, name="decLSTM1")
    decLSTM2 = LSTM(units=hiddenDim, return_sequences=True, name="decLSTM2")
    decDense = TimeDistributed(Dense(outputVocabLen, activation="softmax"), name="decDense")

    layers["decInput"] = decInput
    layers["decInitialStateH"] = decInitialStateH
    layers["decInitialStateC"] = decInitialStateC
    layers["decEmbedding"] = decEmbedding
    layers["decLSTM0"] = decLSTM0
    layers["decLSTM1"] = decLSTM1
    layers["decLSTM2"] = decLSTM2
    layers["decDense"] = decDense

    return layers


# Connects decoder layers together
def connectDecoder(layers, initialState):
    net = layers["decInput"]
    net = layers["decEmbedding"](net)
    net = layers["decLSTM0"](net, initial_state=initialState)
    net = layers["decLSTM1"](net, initial_state=initialState)
    net = layers["decLSTM2"](net, initial_state=initialState)
    net = layers["decDense"](net)

    decOutput = net
    return decOutput


# Defines model layers, compiles model
def createModel(Tx, Ty, inputVocabLen, outputVocabLen, embedDim=128, hiddenDim=512):
    # Create  layers
    encLayers = createEncoderLayers(Tx, inputVocabLen)
    decLayers = createDecoderLayers(Ty, outputVocabLen)

    # Connect layers for training model
    encOutput = connectEncoder(encLayers)
    decOutput = connectDecoder(decLayers, initialState=encOutput)

    # Create training model
    encInput = encLayers["encInput"]
    decInput = decLayers["decInput"]
    model = keras.Model(inputs=[encInput, decInput], outputs=[decOutput], name="trainingModel")

    encModel = keras.Model(inputs=encLayers["encInput"], outputs=encOutput)

    decInitialState = [decLayers["decInitialStateH"], decLayers["decInitialStateC"]]
    decOutput = connectDecoder(decLayers, initialState=decInitialState)    # re-assign decOutput for decModel
    decModel = keras.Model(inputs=[decInput] + decInitialState, outputs=decOutput)

    # Compile model
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    return model, encModel, decModel


# Trains model
def trainModel(loadPrev=True):
    model, encoderModel, decoderModel = createModel(Tx=maxInputLen + 2, Ty=maxScrambleLen + 2, inputVocabLen=stickerLen + 2, outputVocabLen=turnLen + 2)

    if loadPrev:
        model.load_weights(checkpointPath)

    for i in range(numFiles):
        X, Y = loadData(i)

        callbacks = getCallbacks()

        print(X["encInput"].shape)
        print(X["decInput"].shape)
        model.fit(
            x=X, y=Y, epochs=epochs, batch_size=batchSize, validation_split=0.02, callbacks=callbacks)

        encoderModel.save_weights(filepath="data/models/encoderModel.hdf5", save_format="h5")
        decoderModel.save_weights(filepath="data/models/decoderModel.hdf5", save_format="h5")
        model.save(filepath="data/models/model.hdf5", save_format="h5")

    model.summary()
    

# Get callbacks for model.fit()
def getCallbacks():
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpointPath, monitor="val_loss", verbose=1, save_weights_only=True, save_best_only=True)
    earlyStopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(modelName))

    return [checkpoint, earlyStopping, tensorboard]


# Predicts solution from single sticker mapping
def predict(stickers, encoderModel, decoderModel):
    stickersPadded, _ = addStartEnd(stickers, np.zeros((3, 3)))
    h, c = encoderModel.predict(stickersPadded)

    targetSeq = np.zeros((stickers.shape[0], maxScrambleLen + 2))
    targetSeq[:, 0] = np.full((stickers.shape[0], ), fill_value=YBOS)
    
    prevMoves = np.zeros((stickers.shape[0], YBOS + 1))
    for i in range(maxScrambleLen):

        X = {
            "decInput": targetSeq,
            "decInitialStateH": h,
            "decInitialStateC": c
        }

        outputs = decoderModel.predict(X)
        prevMoves = outputs[:, i, :]
        targetSeq[:, i] = np.argmax(prevMoves, axis=1)

    return targetSeq


if __name__ == "__main__":
    # generateDataMulti(trainingSize, totalFiles=numFiles)
    # trainModel(loadPrev=True)

    model, encoderModel, decoderModel = createModel(56, 27, 8, 14)
    model.load_weights(checkpointPath)
    # encoderModel.load_weights(checkpointPath, by_name=True)
    # decoderModel.load_weights(checkpointPath, by_name=True)

    X = np.load("data/features/X0.npy")[:20]
    Y = np.load("data/labels/Y0.npy")[:20]

    print("Input: ")
    print(X)
    print("Prediction: ")
    print(predict(X, encoderModel, decoderModel))
    print("Actual: ")
    _, yPad = addStartEnd(np.zeros((2, 2)), Y)
    print(yPad)

    # X, Y = loadData()
    # preds = np.argmax(model.predict(X), axis=-1)
    # print("Predictions: ")
    # print(preds)
    # print("Actual: ")
    # print(Y["decDense"].astype(int))
