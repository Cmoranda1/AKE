import extractKeyStroke as extract
import os
import config
import numpy as np
import getFeature as gf

def prepareInputTarget():
    os.chdir('/Users/chrismoranda/Desktop/Python_AKE/recording/training/dell/')
    
    keyDict = {key: None for key in config.chosenKeys}
    featureDict = {key: None for key in config.chosenKeys}
    for i in range(config.numOfCharacters):
        currKey = config.chosenKeys[i]
        if(currKey == 't' or currKey == 'w' or currKey == 'z'):
            pushPeak, clicksRecognized, keys = extract.extractKeyStroke(currKey + '.wav', 1, 24)
        else:
            pushPeak, clicksRecognized, keys = extract.extractKeyStroke(currKey + '.wav', 1, config.trainingThreshold)
        print("Recognized " ,clicksRecognized)
        keyDict[currKey] = pushPeak
        featureDict[currKey] = gf.getFeature(pushPeak)
    input = [None]*26
    for i in range(config.numOfCharacters):
        input[i] = featureDict[config.chosenKeys[i]]
    print((input[0]))
    target = np.zeros((26,2600))
    ones = [1]*100
    for i in range(config.numOfCharacters):
        target[i][((i)*config.training_keys):((i)*config.training_keys)+config.training_keys] = ones
    return input, target
