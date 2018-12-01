import extractKeyStroke as extract
import prepareInputTarget as prepare
import neuralnet as nn
import randomForest as rf

my_input, target = prepare.prepareInputTarget()

rf.randomForest(my_input)

#hiddenLayerSize = 100
#net = nn.neuralnet(my_input, target, hiddenLayerSize)

