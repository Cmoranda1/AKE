import numpy as np

def getFeature(pushPeak):
    feature = [[0 for x in range(100)] for y in range(441)]
    for i in range(100):
        temp = np.fft.fft((pushPeak[i]))
        temp = abs(temp)
        feature[i] = temp
    return feature
