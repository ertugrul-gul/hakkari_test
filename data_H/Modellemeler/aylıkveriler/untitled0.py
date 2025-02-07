#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:31:20 2025

@author: ertugrulgul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# Aylık Ortalama Veriyi Hazırlama
# =============================================================================

data_dir = "/home/ertugrulgul/Belgeler/GitHub/hakkari_test/data_H/Modellemeler/Aylık veriler/hakkari.csv"
df = pd.read_csv(data_dir)

df['valid_time'] = pd.to_datetime(df['valid_time'])
df.set_index('valid_time', inplace=True)

# Aylık ortalama hesaplama
df_monthly = df.resample('M').mean()

# =============================================================================
# LSTM Modeli için Veri Hazırlama
# =============================================================================

dataset = df_monthly["t2m"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
print("Train Size:", train_size, "Test Size:", test_size)

train_data = scaled_data[:train_size, :]

# Zaman adımı belirleme
time_steps = 3  # 3 aylık geçmiş veriyi kullanıyoruz
test_data = scaled_data[train_size - time_steps:, :]

x_train, y_train = [], []
for i in range(time_steps, len(train_data)):
    x_train.append(train_data[i-time_steps:i, :])
    y_train.append(train_data[i, :])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

# =============================================================================
# Model Eğitimi veya Yükleme
# =============================================================================

model_path = "monthly_trained_model.keras"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = Sequential([
        Input(shape=(x_train.shape[1], x_train.shape[2])),
        LSTM(50, return_sequences=True),
        LSTM(64, return_sequences=False),
        Dense(32),
        Dense(16),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=["mean_absolute_error"])
    model.fit(x_train, y_train, epochs=100, batch_size=16)
    model.save(model_path)

# =============================================================================
# Tahmin ve Görselleştirme
# =============================================================================

x_test, y_test = [], []
for i in range(time_steps, len(test_data)):
    x_test.append(test_data[i-time_steps:i, :])
    y_test.append(test_data[i, :])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

plt.figure(figsize=(12, 6))
plt.plot(df_monthly.index[train_size:], y_test, label="Gerçek Değerler")
plt.plot(df_monthly.index[train_size:], predictions, label="Tahminler")
plt.xlabel("Tarih")
plt.ylabel("Sıcaklık (°C)")
plt.title("Aylık Ortalama ile LSTM Tahminleri")
plt.legend()
plt.show()
