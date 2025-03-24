#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 23:34:59 2025

@author: ertugrulgul

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot
from scipy.stats import zscore


df = pd.read_csv("../base_data/hakkari_0.csv")
"""
df['t2m'] = df['t2m'] - 273.15
df['tp'] = df['tp']*1000
df['sp'] = df['sp']/100
df["ws"] = np.sqrt(df["u10"]**2 + df["v10"]**2)


"""

# Sonuçları göster
print(df.head())
df.info()

# valid_time'ı datetime formatına çevir
df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")

# Sayısal kolonları belirle
numeric_cols = ["lat", "lon", "sp", "u10", "v10", "t2m", "tp", "ws"]

# Z-score hesapla ve aykırı değerleri belirle
z_scores = df[numeric_cols].apply(zscore)  # Her sütun için Z-score hesaplar
threshold = 3  # Aykırılık eşiği (genellikle 3 kullanılır)
outliers_mask = (np.abs(z_scores) > threshold).any(axis=1)  # En az bir sütunda aykırı olanları bul

# Aykırı değerleri temizlenmiş yeni DataFrame
df_cleaned = df[~outliers_mask]  # Aykırı satırları çıkar

# Temizlenmiş veriyi yeni CSV olarak kaydet, datetime formatı bozulmasın!
df_cleaned.to_csv("hakkari_0.csv", index=False, date_format="%Y-%m-%d")
df_cleaned.info()

df_cleaned["valid_time"] = pd.to_datetime(df_cleaned["valid_time"])
plt.plot(df_cleaned.index, df_cleaned["t2m"])
plt.show()

<