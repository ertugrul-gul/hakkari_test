import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import os
import json

# 📌 Eğer history2 tanımlı değilse, eski eğitim geçmişini kullan
if 'history2' in locals():
    plt.figure(figsize=(12, 8))
    plt.plot(history2.history["loss"])
    plt.plot(history2.history["mean_absolute_error"])
    plt.legend(['Mean Squared Error', 'Mean Absolute Error'])
    plt.title("Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
else:
    # 📌 Eğer history2 yoksa, daha önce kaydedilmiş eğitimi yükleyelim
    with open("training_history2.json", "r") as f:
        history_data = json.load(f)

    plt.figure(figsize=(12, 8))
    plt.plot(history_data["loss"])
    plt.plot(history_data["mean_absolute_error"])
    plt.legend(['Mean Squared Error', 'Mean Absolute'])
