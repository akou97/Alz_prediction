#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:10:55 2021

@author: youness
"""

import numpy as np
from load_data import Dataset
from utils import predictor, plot_result
from sklearn.model_selection import RepeatedKFold
import keras
from keras import layers
from keras import regularizers
from sklearn.model_selection import train_test_split


# Load Dataset
var = 'pressure'
X, y = Dataset(var=var)

def AE(encoding_dim = 32, opt='adam'):
        # This is our input image
    input_img = keras.Input(shape=(12288,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # encoded = layers.Dense(encoding_dim, activation='relu',
    #             activity_regularizer=regularizers.l1(10e-5))(input_img)
    decoded = layers.Dense(12288, activation='sigmoid')(encoded)
    autoencoder = keras.Model(input_img, decoded)
    
    encoder = keras.Model(input_img, encoded)
    encoded_input = keras.Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy',
                        metrics = ['accuracy'])
    return autoencoder


x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    shuffle=True)

encoding_dim = 6000
autoencoder = AE(encoding_dim=encoding_dim)
    
# training model
history = autoencoder.fit(x_train, x_train,
                    epochs=10,
                    batch_size=16,
                    shuffle=True,
                    verbose = 1,
                    validation_data=(x_test, x_test))

print(">> Encoding Dim %d : loss = %.2f val_loss = %.2f " %(encoding_dim,
                                                            history.history['loss'][-1],
                                                            history.history['val_loss'][-1]))
path = 'results/'
plot_result(history, metric="loss", path=path)