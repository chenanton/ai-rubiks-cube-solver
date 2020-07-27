from scrambler import getRandomScrambles, getSolutions, maxScrambleLen, turns  # pylint: disable=import-error
from cube import stickerColors  # pylint: disable=import-error
import numpy as np

# Functions related to generating, formatting, and saving scrambles and corresponding solutions

turnLen = len(turns)
stickerLen = len(stickerColors)

inputFileBase = "data/features/X"
outputFileBase = "data/labels/Y"
fileExt = ".npy"


# Generates specified amount of training examples across specified amount of files
def generateDataMulti(totalExamples, totalFiles=1):
    for i in range(totalFiles):
        generateData(int(totalExamples / totalFiles), i)


# Generates specified amount of training examples and saves it to specified files
def generateData(m, numFiles=0):
    scrambles, stickers = getRandomScrambles(m)
    solutions = getSolutions(scrambles)

    solutionsPadded = padScrambles(solutions)
    stickersFlat = flattenStickers(stickers)

    solutionsOH = toSparse(solutionsPadded, turnLen)
    stickersOH = toSparse(stickersFlat, stickerLen)

    np.save(inputFileBase + str(numFiles) + fileExt, stickersOH)
    np.save(outputFileBase + str(numFiles) + fileExt, solutionsOH)

    print("Saved data to files no. " + str(numFiles) + ".")



# Pads each scramble in scrambles to maximum scramble length; returns np array
# Output dimensions: (number of scrambles, maximum scramble length)
def padScrambles(scrambles, maxScrambleLen=25):
    fillInt = -1   # empty character to be ignored once turned into sparse tensor
    res = np.full((len(scrambles), maxScrambleLen),
                  fill_value=fillInt, dtype="float32")
    for i in range(len(scrambles)):
        res[i, :len(scrambles[i])] = np.array(scrambles[i], dtype="float32")
    return res


# Converts matrix to sparse one-hot tensor
# Output dimensions: (mat.shape[0], mat.shape[1], numClasses)
#   numClasses = turnLen if mat == scrambles
#   numClasses = stickerLen if mat == stickers
def toSparse(mat, numClasses):
    return (np.arange(numClasses) == mat[..., None]).astype("float32")


# Flattens sticker patterns; changes stickers to np array
# Output dimensions: (number of sticker layouts, 54)
def flattenStickers(stickers):
    stickers = np.array(stickers)
    return stickers.reshape(stickers.shape[0], -1)


if __name__ == "__main__":
    # generateData(1000)

    # XTrain = np.load(inputFile)
    # YTrain = np.load(outputFile)

    # print("X shape: ")
    # print(XTrain.shape)

    # print("Y shape: ")
    # print(YTrain.shape)

    test = np.array([[0, 1, 2], [3, 4, 5], [-1, 0, 8]])
    print(toSparse(test, numClasses=6))
