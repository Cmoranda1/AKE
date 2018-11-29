import config
import matlab.engine

def neuralnet(my_input, target, hiddenLayerSize):
	x = my_input
	t = target
	eng = matlab.engine.start_matlab()
	net = eng.patternnet(hiddenLayerSize)
	net.eng.divideParam.trainRatio = 70/100;
	net.eng.divideParam.valRatio = 15/100;
	net.eng.divideParam.testRatio = 15/100;
	net, tr = eng.train(net,x,t)
	return net