import tensorflow as tf
import numpy as np


# Loads data from specified input and output files, returns features and labels
def loadData(inputFile="data/features.npy", outputFile="data/labels.npy"):
    X = np.load(inputFile)
    Y = np.load(outputFile)
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


X, Y = loadData()
(XTrain, YTrain), (XDev, YDev), (XTest, YTest) = partitionData(X, Y, 98, 1, 1)

print(XTrain.shape)
print(XDev.shape)
print(XTest.shape)

# model = tf.keras.Sequential()