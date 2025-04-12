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

"""
# valid_time saat bilgilerini kaldır, sadece tarih kalsın
df1['valid_time'] = df1['valid_time'].dt.floor('D')
df2['valid_time'] = df2['valid_time'].dt.floor('D')
"""

# Verileri değişkenler açısından birleştirme (aynı koordinat ve zaman bilgisi üzerinden)
df_combined = xr.merge([df1, df2])

# Birleştirilmiş veriyi yeni bir dosyaya kaydetme
df_combined.to_netcdf("combined_data.nc")

# Veriyi DataFrame olarak CSV'ye dönüştürme
df_combined_df = df_combined.to_dataframe().reset_index()

# Gereksiz sütunları kaldır
df_combined_df.drop(df_combined_df.columns[3:4], axis=1, inplace=True)

"""
# Dönüşümlerin gerçekleştirilmesi
df_combined_df['t2m'] = df_combined_df['t2m'] - 273.15
df_combined_df['tp'] = df_combined_df['tp']*1000
df_combined_df['sp'] = df_combined_df['sp']/100
df_combined_df["ws"] = np.sqrt(df_combined_df["u10"]**2 + df_combined_df["v10"]**2)
"""

# CSV olarak kaydet
df_combined_df.to_csv("combined_data.csv", index=False)



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
df_cleaned.to_csv("hakkari_0_1.csv", index=False, date_format="%Y-%m-%d")
df_cleaned.info()

"""
