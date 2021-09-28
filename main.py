#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:34:41 2021

@author: youness
"""
import numpy as np
from load_data import Dataset
from utils import predictor, plot_result
from sklearn.model_selection import RepeatedKFold

# Load Dataset
var = 'pressure'
X, y = Dataset(var=var)



cv = RepeatedKFold(n_splits=4, n_repeats=1, random_state=1)

acc_list = list()           # accuracy list
pred_list = list()          # Prediction list
fbeta_list = list()         # F-Beta list


# Params
BS = 16
EPOCHS = 8
step = 1
for train_ix, test_ix in cv.split(X):
    # prepare data
    X_train, X_test = X[train_ix], X[test_ix]
    y_train, y_test = y[train_ix], y[test_ix]
    
    # define decoder
    
    Decodeur = predictor()
    
    print(">>> Step %d" %step)
    # fitting the model 
    history = Decodeur.fit(X_train, y_train,
                           validation_split=0.1,
                           epochs=EPOCHS, shuffle=True,
                           #validation_data=(X_test, y_test),
                           batch_size=BS, verbose= 1)
    
    # make a prediction on the test set
    _, acc, = Decodeur.evaluate(X_test, y_test, verbose=0)
    
    acc_list.append(acc)
    #pred_list.append(pred)
    #fbeta_list.append(fbeta_acc)
    step+=1
   
# summarize performance
print('\n> Accuracy: %.3f (%.3f)' % (np.mean(acc_list), np.std(acc_list)))
# print('> Precision: %.3f (%.3f)' % (np.mean(pred_list), np.std(pred_list)))
# print('> Fbeta : %.3f (%.3f)' % (np.mean(fbeta_list), np.std(fbeta_list)))


# plot curves of metrics
path = 'results/'
plot_result(history, metric="loss", path=path)
plot_result(history, metric="accuracy", path=path)