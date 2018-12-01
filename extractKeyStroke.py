import numpy as np
import scipy.io.wavfile as wav
import os
import math
import soundfile as sf

def extractKeyStroke(filename, mode, threshold):
    y, fs = sf.read(filename)
    maxClicks = 100
    rawSound = np.transpose(y[:,0])
   
    #disregard first second
    rawSound = rawSound[44100-1:np.size(rawSound)]
    winSize = 40
    winNum = math.floor(np.size(rawSound)/winSize)
    clickSize = 44100*0.08
    binSums = np.zeros(winNum)
    for i in range(winNum):
        currentWindow = np.fft.fft(rawSound[(winSize*i):(winSize*(i+1))])
        for j in range(np.size(currentWindow)):
            binSums[i] = binSums[i] + abs(currentWindow[j])
    #problem is possibly in binSums
    clickPositions = np.zeros(maxClicks)
    j = 0
    h = 0
    offsetToNextClick = math.ceil(clickSize/winSize)
    while(h < np.size(binSums)-1 and j < maxClicks):
        if(binSums[h] > threshold):
            clickPositions[j] = (h+1)*winSize
            j+=1
            h = h+offsetToNextClick
        else:
            h+=1
    k = 0
    clicksRecognized = 0
    while(k < np.size(clickPositions)):
        if(clickPositions[k] != 0):
            clicksRecognized = clicksRecognized + 1
        k += 1

    numOfClicks = clicksRecognized
    #keys = [[0 for x in range(numOfClicks)]for y in range(int(clickSize))]
    keys = [[0 for x in range(int(clickSize))]for y in range(numOfClicks)]
    for i in range(numOfClicks):
        if(clickPositions[i] != 0):
            startIndex = clickPositions[i] - 101
            endIndex = startIndex+int(clickSize) - 1
        if(startIndex >=0 and endIndex < np.size(rawSound)):
            keys[i] = rawSound[int(startIndex):int(endIndex)]

    #pushPeak = [[0 for x in range(numOfClicks)]for y in range(441)]
    pushPeak = [[0 for x in range(441)]for y in range(numOfClicks)]
    x = 0
    for i in range(numOfClicks):
        #pushPeak[x:x+441] = keys[i][0:441]
        #x+=441
        pushPeak[i][0:441] = keys[i][0:441]
    print(filename)
    return pushPeak, clicksRecognized, keys
