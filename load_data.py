#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:15:20 2021

@author: youness
"""

import os
import numpy as np
from zipfile import ZipFile
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import load_img, img_to_array

# Choose a your variable
#var = 'raw'
var = "pressure"
#var = "penups_raw"
def Dataset(var = var):
    # Load data
    file_name = var + "_augmented.zip"
    zipo = ZipFile(file_name)
    zipo.extractall()
    
    path_alz =  'augmented/alz/'
    path_control = 'augmented/control/'
    
    # n_alz = len(os.listdir(path_alz))
    # n_control = len(os.listdir(path_alz))
    # labels=['control','alz']
    
    photos, targets = list(), list()
    # enumerate files in the directory
    for filename in os.listdir(path_alz):
        # load image
        photo = load_img(path_alz + filename, target_size=(64,64))
        # convert to numpy array
        photo = img_to_array(photo, dtype='uint8')
        # expand dimensions so that it represents a single 'sample'
        #photo = np.expand_dims(photo, axis=0)

        # store
        photos.append(photo.reshape(-1))
        targets.append(1)
        
    # enumerate files in the directory
    for filename in os.listdir(path_control):
        # load image
        photo = load_img(path_control + filename, target_size=(64,64))
        # convert to numpy array
        photo = img_to_array(photo, dtype='uint8')
        # store
        photos.append(photo.reshape(-1))
        targets.append(0)
    
    X = np.array(photos, dtype='uint8')
    y = np.array(targets, dtype='uint8')
    
    print("Dimension of X is ", X.shape)
    print("Dimension of y is ", y.shape)
    
    return X, y






