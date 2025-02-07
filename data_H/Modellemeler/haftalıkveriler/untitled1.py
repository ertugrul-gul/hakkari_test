#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 23:34:59 2025

@author: ertugrulgul
"""

import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, TimeDistributed, Bidirectional
from tensorflow.keras.models import model_from_json

df = pd.read_csv("/home/ertugrulgul/Belgeler/GitHub/hakkari_test/data_H/Modellemeler/haftalıkveriler/hakkari.csv")
df.head()
df.info()
df.describe()