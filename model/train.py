import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import time

from tensorflow import keras
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, TimeDistributed, Bidirectional, Attention, Input, RepeatVector

from generateData import generateData, toSparse, generateDataMulti, inputFileBase, outputFileBase, fileExt, fillInt
from scrambler import maxScrambleLen


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
epochs = 3
numFiles = 50

modelName = "rubiks-cube-lstm-{}".format(int(time.time()))
checkpointPath = "logs/checkpoints/checkpoint.keras"

maxInputLen = 54
hiddenSize = 128

historyPath = "data/histories/history"


# Defines model layers, compiles model
def createModel(Tx, Ty, inputVocabLen, outputVocabLen, embedDim=128, hiddenDim=512):
    # Training Model
    # Encoder
    encInput = Input(shape=(Tx, ), name="encInput")
    encEmbedding = Embedding(input_dim=inputVocabLen,
                             output_dim=embedDim, input_length=Tx)(encInput)

    encLSTM = LSTM(units=hiddenDim, return_state=True)
    _, h, c = encLSTM(encEmbedding)

    # Decoder
    decInput = Input(shape=(Ty, ), name="decInput")
    decEmbedding = Embedding(input_dim=outputVocabLen,
                             output_dim=embedDim, input_length=Ty)(decInput)

    decLSTM = LSTM(units=hiddenDim, return_state=True, return_sequences=True)
    decLSTMOutput, _, _ = decLSTM(decEmbedding, initial_state=[h, c])

    decDense = TimeDistributed(Dense(outputVocabLen, activation="softmax"))
    decOutput = decDense(decLSTMOutput)

    model = keras.Model(inputs=[encInput, decInput], outputs=[decOutput])
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])


    # Encoder model
    encoderModel = keras.Model(inputs=encInput, outputs=[h, c])

    # Decoder model
    decModelStateH = Input(shape=(hiddenDim, ), name="decInitialStateH")
    decModelStateC = Input(shape=(hiddenDim, ), name="decInitialStateC")
    decModelOutputs, decModelH, decModelC = decLSTM(decEmbedding, initial_state=[decModelStateH, decModelStateC])
    decModelOutputs = decDense(decModelOutputs)

    decoderModel = keras.Model(inputs=[decInput, decModelStateH ,decModelStateC], outputs=[decModelOutputs, decModelH, decModelC])

    return model, encoderModel, decoderModel


# Trains model
def trainModel():
    model, _, _ = createModel(Tx=maxInputLen, Ty=maxScrambleLen, inputVocabLen=6, outputVocabLen=12 + 1)
    model.load_weights(checkpointPath)

    for i in range(numFiles):
        X, Y = loadData(i)

        checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpointPath, monitor="val_loss", verbose=1, save_weights_only=True, save_best_only=True)
        earlyStopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1)
        tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(modelName))

        callbacks=[checkpoint, earlyStopping, tensorboard]

        model.fit(
            x=X, y=Y, epochs=epochs, batch_size=batchSize, validation_split=0.02, callbacks=callbacks)

    model.summary()
    model.save(filepath="data/model.hdf5", save_format="h5")


# Predicts solution from single sticker mapping
def predict(stickers, encoderModel, decoderModel):
    h, c = encoderModel.predict(stickers)

    targetSeq = np.zeros((stickers.shape[0], maxScrambleLen))
    
    prevMoves = np.zeros((fillInt + 1, ))
    for i in range(maxScrambleLen):

        X = {
            "decInput": targetSeq,
            "decInitialStateH": h,
            "decInitialStateC": c
        }

        outputs, _, _ = decoderModel.predict(X)
        prevMoves = outputs[:, i, :]
        targetSeq[:, i] = np.argmax(prevMoves, axis=-1)

    return targetSeq


if __name__ == "__main__":
    # generateDataMulti(trainingSize, totalFiles=numFiles)
    # trainModel()

    model, encoderModel, decoderModel = createModel(54, 25, 6, 13)
    model.load_weights(checkpointPath)
    encoderModel.load_weights(checkpointPath)
    decoderModel.load_weights(checkpointPath)

    X = np.load("data/features/X0.npy")[:20]
    Y = np.load("data/labels/Y0.npy")[:20]

    print("Prediction: ")
    print(predict(X, encoderModel, decoderModel))
    print("Actual: ")
    print(Y)

