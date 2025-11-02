#-------------------------------------------------------------------------
# AUTHOR: Benjamin Tran
# FILENAME: perceptron.py
# SPECIFICATION: build a single-layer and multi-layer perceptron for digits written by hand and procced using sliding window
# FOR: CS 4210- Assignment #3
# TIME SPENT: 1 day
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

#trackers
best_single = 0.0
best_params_single= None

best_multi = 0.0
best_params_multi = None

runs = 0

for rate in n: #iterates over n

    for shuffled in r: #iterates over r

        #two diff algs
        for algo in ['Perceptron', 'MLP']:
            
            runs += 1

            if algo == 'Perceptron':
                #single layer, with given hyperparameters
                clf = Perceptron(eta0=rate, shuffle=shuffled, max_iter=1000, random_state=0)
            else:
                #mluti layer, with given hyperparametrs
                clf = MLPClassifier(activation='logistic', learning_rate_init=rate, hidden_layer_sizes=(25,), shuffle=shuffled, max_iter=1000, random_state=0)


            clf.fit(X_training, y_training)

            # calc accuracy on test 
            correct = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                pred = clf.predict([x_testSample])[0]
                if pred == y_testSample:
                    correct += 1
            accuracy = correct / len(y_test)

            # check/update bests
            if algo == 'Perceptron':
                if accuracy > best_single:
                    best_single = accuracy
                    best_params_single = (rate, shuffled)
                    print(f"Highest Perceptron accuracy so far: {best_single:.4f}, "
                          f"Parameters: learning rate={rate}, shuffle={shuffled}")
            else:
                if accuracy > best_multi:
                    best_multi = accuracy
                    best_params_multi = (rate, shuffled)
                    print(f"Highest MLP accuracy so far: {best_multi:.4f}, "
                          f"Parameters: learning rate={rate}, shuffle={shuffled}")

# summary
print("\n\nModels Finished!")
print(f"Total runs: {runs}")

print(f"Best Perceptron: accuracy={best_single:.4f}, "
      f"learning_rate={best_params_single[0]}, shuffle={best_params_single[1]}")

print(f"Best MLP: accuracy={best_multi:.4f}, "
      f"learning_rate={best_params_multi[0]}, shuffle={best_params_multi[1]}")












