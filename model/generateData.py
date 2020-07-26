from scrambler import getRandomScrambles, getSolutions, maxScrambleLen, turns  # pylint: disable=import-error
from cube import stickerColors # pylint: disable=import-error 
import numpy as np

# Functions related to generating, formatting, and saving scrambles and corresponding solutions

turnLen = len(turns)
stickerLen = len(stickerColors)

inputFile = "data/features.npy"
outputFile = "data/labels.npy"

# Generates specified amount of training examples and saves it to specified files
def generateData(m, inputFile="data/features.npy", outputFile="data/labels.npy"):
    scrambles, stickers = getRandomScrambles(m)
    solutions = getSolutions(scrambles)

    solutionsPadded = padScrambles(solutions)
    stickersFlat = flattenStickers(stickers)

    solutionsOH = toSparse(solutionsPadded, turnLen)
    stickersOH = toSparse(stickersFlat, stickerLen)

    np.save(inputFile, stickersOH)
    np.save(outputFile, solutionsOH)


# Pads each scramble in scrambles to maximum scramble length; returns np array
# Output dimensions: (number of scrambles, maximum scramble length)
def padScrambles(scrambles, maxScrambleLen=25):
    fillInt = -1   # empty character to be ignored once turned into sparse tensor
    res = np.full((len(scrambles), maxScrambleLen), fill_value=fillInt)
    for i in range(len(scrambles)):
        res[i, :len(scrambles[i])] = np.array(scrambles[i], dtype="int32")
    return res


# Converts matrix to sparse one-hot tensor
# Output dimensions: (mat.shape[0], mat.shape[1], numClasses)
#   numClasses = turnLen if mat == scrambles
#   numClasses = stickerLen if mat == stickers
def toSparse(mat, numClasses):
    return (np.arange(numClasses) == mat[..., None]).astype(int)


# Flattens sticker patterns; changes stickers to np array
# Output dimensions: (number of sticker layouts, 54)
def flattenStickers(stickers):
    stickers = np.array(stickers)
    return stickers.reshape(stickers.shape[0], -1)


if __name__ == "__main__":
    generateData(1000)

    XTrain = np.load(inputFile)
    YTrain = np.load(outputFile)

    print("X shape: ")
    print(XTrain.shape)

    print("Y shape: ")
    print(YTrain.shape)