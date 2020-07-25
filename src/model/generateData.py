from scrambler import getRandomScrambles, getSolution, maxScrambleLen, turns  # pylint: disable=import-error
from cube import stickerColors # pylint: disable=import-error
import numpy as np

# Functions related to generating, formatting, and saving scrambles and corresponding solutions

turnLen = len(turns)
stickerLen = len(stickerColors)


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



scrambles, stickers = getRandomScrambles(2)
# for s in scrambles:
#     print(s)
# padded = padScrambles(scrambles)
# sparse = toSparseScrambles(padded)
# print(sparse)

print(stickers)
print(toSparse(flattenStickers(stickers), numClasses=len(turns)))