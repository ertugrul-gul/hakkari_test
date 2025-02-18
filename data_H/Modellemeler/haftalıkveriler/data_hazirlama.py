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
from datetime import datetime
from scipy.stats import zscore


"""""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, TimeDistributed, Bidirectional
from tensorflow.keras.models import model_from_json
"""""


df = pd.read_csv("hakkari.csv")
# Kolon indeksleri 0'dan başlar. 3. ve 4. kolonlar sırasıyla index 2 ve 3'e denk gelir.
df = df.drop(df.columns[[3, 4]], axis=1)
# Sonuçları göster
print(df.head())
df.info()

# valid_time'ı datetime formatına çevir
df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")

# Sayısal kolonları belirle
numeric_cols = ["lat", "lon", "sp", "u10", "v10", "t2m", "tp"]

# Z-score hesapla ve aykırı değerleri belirle
z_scores = df[numeric_cols].apply(zscore)  # Her sütun için Z-score hesaplar
threshold = 3  # Aykırılık eşiği (genellikle 3 kullanılır)
outliers_mask = (np.abs(z_scores) > threshold).any(axis=1)  # En az bir sütunda aykırı olanları bul

# Aykırı değerleri temizlenmiş yeni DataFrame
df_cleaned = df[~outliers_mask]  # Aykırı satırları çıkar

# Temizlenmiş veriyi yeni CSV olarak kaydet, datetime formatı bozulmasın!
df_cleaned.to_csv("hakkari_end.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
df_cleaned.info()