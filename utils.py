#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:17:04 2021
TODO
@author: youness
"""

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from metrics import precision, fbeta
import matplotlib.pyplot as plt

def predictor(in_shape=(64,64,3), out_shape=50):
   
    """
    Model :
         - 3 conv blocs 
         - N denses layers
    Goal : predict 
    """
    
   
    # Features Extractor
    k = 5 # kernel size
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(k, k), activation="relu",
                     input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(5, 5)))

    
    # model.add(Conv2D(filters=32, kernel_size=(k, k), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())

    
    # model.add(Conv2D(filters=64, kernel_size=(k, k), activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())

    model.add(Flatten())

    # Classifier parameters
    epsilon = 1e-6
    momentum = 0.9
    
    lenght_layers = 1
    neuron_layers = 50
    
    out_shape = 1
    kernel_ini='uniform'
    lr = 0.001
    
    for _ in range(lenght_layers):
        model.add(Dense(neuron_layers,kernel_initializer=kernel_ini,
                        activation='relu'))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))

    model.add(Dense(out_shape,kernel_initializer=kernel_ini,
                    activation='sigmoid'))
    optimisateur = Adam(learning_rate=lr)
    model.compile(optimizer=optimisateur, loss='binary_crossentropy', 
                         metrics=['accuracy'])
    return model


def plot_result(history, metric, path = None):
    """
    Plotting a metric value from a history training.   
    
    Parameters
    ----------
    history : Dictionnary
        historic of Model training.
    metric : string
        name of the metric.
    path : string, optional
        name of the path to save the plot.

    Returns
    -------
    None.

    """
    plt.style.use("ggplot")
    plt.figure()
    acc = history.history[metric]
    val_acc = history.history['val_'+metric]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'y', label='Training '+ metric)
    plt.plot(epochs, val_acc, 'r', label='Validation '+ metric)
    plt.title('Training and validation '+ metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    
    if path != None:
        file_name = path+metric+'.png'
        plt.savefig(file_name)
    