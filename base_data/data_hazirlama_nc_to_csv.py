#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 23:34:59 2025

@author: ertugrulgul

"""
import numpy as np
import pandas as pd
import xarray as xr

# NC dosyalarını açma
df1 = xr.open_dataset("data_0.nc")
df2 = xr.open_dataset("data_1.nc")


# valid_time saat bilgilerini kaldır, sadece tarih kalsın
df1['valid_time'] = df1['valid_time'].dt.floor('D')
df2['valid_time'] = df2['valid_time'].dt.floor('D')


# Verileri değişkenler açısından birleştirme (aynı koordinat ve zaman bilgisi üzerinden)
df_combined = xr.merge([df1, df2])

# Birleştirilmiş veriyi yeni bir dosyaya kaydetme
df_combined.to_netcdf("combined_data.nc")

# Veriyi DataFrame olarak CSV'ye dönüştürme
df_combined_df = df_combined.to_dataframe().reset_index()

# Gereksiz sütunları kaldır
df_combined_df.drop(df_combined_df.columns[[3,4]], axis=1, inplace=True)

# Dönüşümlerin gerçekleştirilmesi
df_combined_df['t2m'] = df_combined_df['t2m'] - 273.15 # kelvin to Celcius
df_combined_df['tp'] = df_combined_df['tp']*1000 # m to mm
df_combined_df['sp'] = df_combined_df['sp']/100 # Pascal to hPa
df_combined_df['d2m'] = df_combined_df['d2m'] - 273.15 # çiğ sıcaklık kelvin to Celcius

#Bağıl nem (RH) hesapla (t2m ve d2m'den)
d2m_C = df_combined_df["d2m"]
t_C = df_combined_df["t2m"]
e_t = 6.112 * np.exp((17.67 * t_C) / (t_C + 243.5))
e_d = 6.112 * np.exp((17.67 * d2m_C) / (d2m_C + 243.5))
RH = 100 * (e_d / e_t)
df_combined_df["RH"] = RH.clip(upper=100)

#eksik verileri kontrol etme
missing_data = df_combined_df.isnull().sum()
rows_before = len(df_combined_df)
df_combined_df.dropna(inplace=True)
rows_after = len(df_combined_df)
print(f"{rows_before - rows_after} satır silindi Toplam {missing_data} eksik hücre vardı.")

# IQR yöntemiyle aykırı değer analizi
def clean_by_iqr(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    iteration = 0
    while True:
        iteration += 1
        outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        num_outliers = outlier_mask.any(axis=1).sum()
        if num_outliers == 0:
            print(f"Aykırı değer kalmadı. Toplam {iteration} adımda temizlendi.")
            break
        print(f"Iterasyon {iteration}: {num_outliers} satır silindi.")
        df = df[~outlier_mask.any(axis=1)]
    return df

# veriyi temizle
df_cleaned = clean_by_iqr(df_combined_df)

# valid_time sütununu datetime formatına çevir (emin olmak için)
df_cleaned["valid_time"] = pd.to_datetime(df_cleaned["valid_time"])

# Ay bilgisini ekle
df_cleaned["month"] = df_cleaned["valid_time"].dt.month

# Mevsim bilgisi ((0: Kış, 1: İlkbahar, 2: Yaz, 3: Sonbahar))
df_cleaned["season"] = df_cleaned["month"] % 12 // 3
def get_season(month):
    if month in [12, 1, 2]:
        return 0
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    else:
        return 3

df_cleaned["season"] = df_cleaned["month"].apply(get_season)

# Dairesel mevsimsellik dönüşümleri
df_cleaned["month_sin"] = np.sin(2 * np.pi * df_cleaned["month"] / 12)
df_cleaned["month_cos"] = np.cos(2 * np.pi * df_cleaned["month"] / 12)

#Temizlenmiş veriyi CSV olarak kaydet
df_cleaned.to_csv("combined_data_cleaned_final.csv", index=False)
print("Mevsimsellik eklendi ve tamamen temizlenmiş veri combined_data_cleaned_final.csv dosyasına kaydedildi.")

# ... önceki kodlarının tamamı burada aynen korunuyor ...

# Temizlenmiş veri CSV'ye kaydedildikten sonra devam:
df_cleaned.to_csv("combined_data_cleaned_final.csv", index=False)
print("Mevsimsellik eklendi ve tamamen temizlenmiş veri combined_data_cleaned_final.csv dosyasına kaydedildi.")

# Sonuçları göster
print(df_combined_df.head())
df_combined_df.info()

# -------------------------------------------
# STL Ayrıştırma: Her koordinat için t2m ve tp
# -------------------------------------------
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import os

# Grafik çıktıları için klasör oluştur
output_dir = "decomposition_plots"
os.makedirs(output_dir, exist_ok=True)

# Benzersiz koordinat çiftlerini al
coords = df_cleaned[["latitude", "longitude"]].drop_duplicates().values

# Her koordinat çifti için ayrıştırma yap
for lat, lon in coords:
    sub_df = df_cleaned[(df_cleaned["latitude"] == lat) & (df_cleaned["longitude"] == lon)].sort_values("valid_time")

    ts_t2m = sub_df.set_index("valid_time")["t2m"]
    ts_tp = sub_df.set_index("valid_time")["tp"]

    try:
        # STL ile ayrıştır
        stl_t2m = STL(ts_t2m, period=12, robust=True).fit()
        stl_tp = STL(ts_tp, period=12, robust=True).fit()

        # t2m için grafik
        fig1 = stl_t2m.plot()
        fig1.suptitle(f"STL Decomposition - Temperature (t2m)\n({lat}, {lon})")
        fig1.tight_layout()
        fig1.savefig(f"{output_dir}/decompose_{lat}_{lon}_t2m.png", dpi=300)
        plt.close(fig1)

        # tp için grafik
        fig2 = stl_tp.plot()
        fig2.suptitle(f"STL Decomposition - Precipitation (tp)\n({lat}, {lon})")
        fig2.tight_layout()
        fig2.savefig(f"{output_dir}/decompose_{lat}_{lon}_tp.png", dpi=300)
        plt.close(fig2)

        print(f"✓ STL ayrıştırma tamamlandı: ({lat}, {lon})")

    except Exception as e:
        print(f"⚠ STL hatası: ({lat}, {lon}) -> {e}")

# -------------------------------------------
# Haritalandırma
# -------------------------------------------
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

plt.show()

