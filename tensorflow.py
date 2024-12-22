import os
import platform
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.impute import SimpleImputer
import tensorflow as tf

# GPU Kullanımı
physical_devices = tf.config.list_physical_devices('GPU')
print("GPU:", len(physical_devices))