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

# Veri yükleme
data_path_0 = 'data_H/data_0.nc'
data_path_1 = 'data_H/data_1.nc'
data_0 = xr.open_dataset(data_path_0)
data_1 = xr.open_dataset(data_path_1)
data = xr.concat([data_0, data_1], dim='time')

# Hedef ve giriş değişkenleri
temperature = data['t2m']
features = data[['sp', 'u10', 'v10', 'tp', 'valid_time', 'latitude', 'longitude']]

# Tüm veri setini Pandas DataFrame'e dönüştürme
df_features = features.to_dataframe().reset_index()
df_temperature = temperature.to_dataframe().reset_index()

# tp değişkeninde 0 olan verileri kaldırma
df_features = df_features[df_features['tp'] != 0]  

# Mevsim bilgisi ekleme
def add_season_column(df):
    def assign_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'

    df['month'] = pd.to_datetime(df['valid_time']).dt.month
    df['season'] = df['month'].apply(assign_season)
    df['year'] = pd.to_datetime(df['valid_time']).dt.year
    return df

# Tüm verilere mevsim bilgisi ekleme
df_features = add_season_column(df_features)
df_temperature = add_season_column(df_temperature)

# Verilerin ilk birkaç satırını inceleme
print(df_features.head())
print(df_temperature.head())

# Tüm yıllar için mevsimsel yağış değişiklik grafiği oluşturma
def plot_precipitation_trends(df_features):
    # Yıl ve mevsime göre toplam yağış miktarını hesaplama
    df_features['tp'] = df_features['tp'] * 1000  # Yağış birimi metre -> milimetre dönüştürüldü
    precipitation_sum = df_features.groupby(['year', 'season'])['tp'].sum().unstack()

    # Çubuk grafik oluşturma
    precipitation_sum = precipitation_sum.reindex(index=range(1940, 2024))  # Yıllar 1940'tan itibaren yeniden düzenlendi
    precipitation_sum.plot(kind='bar', figsize=(14, 8), width=0.8, edgecolor='black')
    plt.title("1940'tan Günümüze Mevsimlik Yağış Miktarı")
    plt.ylabel('Toplam Yağış (mm)')
    plt.xlabel('Yıl')
    plt.xticks(rotation=90, fontsize=8)
    plt.legend(title='Mevsim')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Yağış grafiği oluşturma
plot_precipitation_trends(df_features)

# Tüm yıllar için mevsimsel çubuk grafik oluşturma
def plot_seasonal_trends_all_years(df_temperature):
    # Sıcaklıkları Kelvin'den Celsius'a dönüştürme
    df_temperature['t2m'] = df_temperature['t2m'] - 273.15  

    # Yıl ve mevsime göre ortalamaları hesaplama
    seasonal_avg = df_temperature.groupby(['year', 'season'])['t2m'].mean().unstack()

    # Çubuk grafik oluşturma
    seasonal_avg = seasonal_avg.reindex(index=range(1940, 2024)) 
    seasonal_avg.plot(kind='bar', figsize=(14, 8), width=0.8, edgecolor='black')
    plt.title("1940'tan Günümüze Mevsimlik Sıcaklık Ortalamaları")
    plt.ylabel('Ortalama Sıcaklık (°C)')
    plt.xlabel('Yıl')
    plt.xticks(rotation=90, fontsize=8)
    plt.legend(title='Mevsim')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Grafik oluşturma
plot_seasonal_trends_all_years(df_temperature)

# 1991-2021 yılları arasında her ay için tablo oluşturma
def create_monthly_statistics(df_temperature, df_features):
    # Sıcaklık birimini Celsius'a dönüştürme
    df_temperature['t2m'] = df_temperature['t2m']

    # 1991-2021 yıllarını filtreleme
    filtered_temp = df_temperature[(df_temperature['year'] >= 1991) & (df_temperature['year'] <= 2021)]
    filtered_features = df_features[(df_features['year'] >= 1991) & (df_features['year'] <= 2021)]

    # Yağış miktarını aylık toplam olarak hesaplama
    monthly_precipitation = filtered_features.groupby(['year', 'month'])['tp'].sum().reset_index()

    # Her ay için toplam yağış ortalamasını hesaplama
    monthly_precipitation_avg = monthly_precipitation.groupby('month')['tp'].mean()

    # Sıcaklık istatistiklerini aylık bazda hesaplama
    monthly_stats = filtered_temp.groupby('month')['t2m'].agg(['mean', 'min', 'max']).rename(
        columns={'mean': 'Ortalama Sıcaklık (°C)', 'min': 'Minimum Sıcaklık (°C)', 'max': 'Maksimum Sıcaklık (°C)'}
    )

    # Aylık toplam yağışları tabloya ekleme
    monthly_stats['Yağış (mm)'] = monthly_precipitation_avg.values

    # Ay isimlerini ekleme
    month_names = {
        1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan', 5: 'Mayıs', 6: 'Haziran',
        7: 'Temmuz', 8: 'Ağustos', 9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
    }
    monthly_stats.index = monthly_stats.index.map(month_names)

    return monthly_stats

# Tabloyu görselleştirme
def plot_monthly_statistics_table(monthly_statistics):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    table = plt.table(cellText=monthly_statistics.values,
                      colLabels=monthly_statistics.columns,
                      rowLabels=monthly_statistics.index,
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(monthly_statistics.columns))))
    plt.title("1991-2021 Aylık İstatistikler", fontsize=14, pad=20)
    plt.show()

# Güncellenmiş tabloyu oluşturma ve görselleştirme
monthly_statistics = create_monthly_statistics(df_temperature, df_features)
plot_monthly_statistics_table(monthly_statistics)
