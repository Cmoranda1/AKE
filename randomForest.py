from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np

def randomForest(input):
    x = -1
    #put input in workable array
    new_input = [[0 for x in range(441)]for y in range(2600)]
    new_target = [None] * 2600
    
    for i in range(26):
        for j in range(100):
            x += 1
            for k in range(441):
                new_input[x][k] = input[i][j][k]
    new_input = np.array(new_input)
    #put test in workable array [0,0,0...
    #                           [26, 26, ...
    curr = -1
    for i in range(2600):
        if((i % 100) == 0):
            curr += 1
        new_target[i] = curr

    data = pd.DataFrame({
                        '1':new_input[:,0],
                        '2':new_input[:,1],
                        '3':new_input[:,2],
                        '4':new_input[:,3],
                        '5':new_input[:,4],
                        '6':new_input[:,5],
                        '7':new_input[:,6],
                        '8':new_input[:,7],
                        '9':new_input[:,8],
                        '10':new_input[:,9],
                        '11':new_input[:,10],
                        '12':new_input[:,11],
                        '13':new_input[:,12],
                        '14':new_input[:,13],
                        '15':new_input[:,14],
                        '16':new_input[:,15],
                        'letter':new_target
                        })

    data.head()

    X = data[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']]
    Y = data['letter']

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)

    clf = RandomForestClassifier(n_estimators = 100)

    clf.fit(X_train, Y_train)

    Y_pred=clf.predict(X_test)
    print(clf.predict([[17.525543212890625, 15.875853489523283, 16.516690189144512, 17.06837589628611, 21.290453413552488, 7.716098066275512, 2.720956366940635, 5.482293485721773, 8.042865555602257, 9.82747886696717, 9.583944631808775, 11.807086725039062, 12.570187104824875, 18.40145327220292, 18.332389547865795, 15.397757194336817]]))

    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
