#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 23:34:59 2025

@author: ertugrulgul

"""
import numpy as np
import pandas as pd
from scipy.stats import zscore
import xarray as xr

# NC dosyalarını açma
df1 = xr.open_dataset("data_0_m.nc")
df2 = xr.open_dataset("data_1_m.nc")

# Verileri değişkenler açısından birleştirme (aynı koordinat ve zaman bilgisi üzerinden)
ds_combined = xr.merge([df1, df2])

# Birleştirilmiş veriyi yeni bir dosyaya kaydetme
ds_combined.to_netcdf("combined_data.nc")

# Veriyi DataFrame olarak CSV'ye dönüştürme
ds_combined_df = ds_combined.to_dataframe().reset_index()
ds_combined_df.to_csv("combined_data.csv", index=False)

"""
df['t2m'] = df['t2m'] - 273.15
df['tp'] = df['tp']*1000
df['sp'] = df['sp']/100
df["ws"] = np.sqrt(df["u10"]**2 + df["v10"]**2)



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
df_cleaned.to_csv("hakkari_0_1.csv", index=False, date_format="%Y-%m-%d")
df_cleaned.info()

"""
