import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Dropout

from sklearn.preprocessing import MinMaxScaler

# NetCDF dosyalarını yükleme
data_path_0 = 'data_H/data_0.nc'
data_path_1 = 'data_H/data_1.nc'

# Veriyi xarray ile açma
data_0 = xr.open_dataset(data_path_0)
data_1 = xr.open_dataset(data_path_1)

# Veriyi inceleme
print(data_0)
print(data_1)
