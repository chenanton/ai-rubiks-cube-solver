from scrambler import getRandomScrambles, getSolution, maxScrambleLen, turns  # pylint: disable=import-error
import numpy as np

# Functions related to generating, formatting, and saving scrambles and corresponding solutions


# Pads each scramble in scrambles to maximum scramble length; returns np array
# Output dimensions: (number of scrambles * maximum scramble length)
def padScrambles(scrambles, maxScrambleLen=25):
    fillInt = -1   # empty character to be ignored once turned into sparse tensor
    res = np.full((len(scrambles), maxScrambleLen), fill_value=fillInt)
    for i in range(len(scrambles)):
        res[i, :len(scrambles[i])] = np.array(scrambles[i], dtype="int32")
    return res


# Converts padded scrambles matrix to sparse one-hot tensor
# Output dimensions: (number of scrambles * max scramble length * number of turn types)
def toSparseScrambles(scrambles, turns=len(turns)):
    return (np.arange(turns) == scrambles[..., None]).astype(int)


print("working!")


scrambles, _ = getRandomScrambles(10)
for s in scrambles:
    print(s)
padded = padScrambles(scrambles)
sparse = toSparseScrambles(padded)
print(sparse)


