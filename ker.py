from keras.models import Sequential
from keras.layers import Dense
import os
import numpy

os.chdir('/Users/chrismoranda/Desktop/Python_AKE')
numpy.random.seed(7)

#load pima indians dataset

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

#split into imput (X) and output (Y) variables

X = dataset[:,0:8]
Y = dataset[:,8]
print(X)
print(Y)
#create model

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit model

model.fit(X, Y, epochs=150, batch_size=10, verbose = 2)

#evaluate the model

#calculate predictions
predictions = model.predict(X)

rounded = [round(x[0]) for x in predictions]
print(rounded)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
