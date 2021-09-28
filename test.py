#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:22:04 2021

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