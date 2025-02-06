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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata

# Veri yükleme
data_path_0 = 'data_H/data_0_m.nc'
data_path_1 = 'data_H/data_1_m.nc'
data_0 = xr.open_dataset(data_path_0)
data_1 = xr.open_dataset(data_path_1)
data = xr.concat([data_0, data_1], dim='time')

# Hedef ve giriş değişkenleri
temperature = data['t2m'] - 273.15  # Kelvin -> Celsius
features = data[['u10', 'v10', 'tp', 'valid_time', 'latitude', 'longitude']]

# tp değişkeninde 0 olan verileri kaldırma
df_features = features.to_dataframe().reset_index()
df_features = df_features[df_features['tp'] != 0]

# t2m sütununu df_features'e ekleme
temp_df = temperature.to_dataframe().reset_index()
df_features = pd.merge(df_features, temp_df[['valid_time', 'latitude', 'longitude', 't2m']], on=['valid_time', 'latitude', 'longitude'], how='left')

# Tarih sütununu datetime formatına çevirme
df_features['valid_time'] = pd.to_datetime(df_features['valid_time'])
df_features['year'] = df_features['valid_time'].dt.year
df_features['season'] = df_features['valid_time'].dt.month.map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
})

# Harita üzerindeki gösterim için ortalama sıcaklık
temp_mean = data_0['t2m'].mean(dim='valid_time').values - 273.15  # Kelvin -> Celsius
lons = data_0['longitude'].values
lats = data_0['latitude'].values

# Grid verilerini interpolasyon ile genişletme
lon2d, lat2d = np.meshgrid(lons, lats)
lon_new = np.linspace(lons.min(), lons.max(), 400)  # Daha yüksek çözünürlük
lat_new = np.linspace(lats.min(), lats.max(), 400)
lon2d_new, lat2d_new = np.meshgrid(lon_new, lat_new)
temp_mean_interp = griddata((lon2d.flatten(), lat2d.flatten()), temp_mean.flatten(), (lon2d_new, lat2d_new), method='cubic')

# Türkiye sınırlarını tanımlama (tüm Türkiye haritası)
turkey_extent = [25, 45, 35, 45]  # [min_lon, max_lon, min_lat, max_lat]

# Şehir merkezlerinin koordinatları ve isimleri
city_coordinates = {
    "Ankara": (32.8597, 39.9334),
    "Istanbul": (28.9784, 41.0082),
    "Izmir": (27.1428, 38.4192),
    "Hakkari": (43.7408, 37.5744)
}

# Harita oluşturma
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(turkey_extent, crs=ccrs.PlateCarree())

# Türkiye'yi idari harita olarak renklendirme
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')  # Sınır çizgileri
ax.add_feature(cfeature.COASTLINE, edgecolor='black')  # Sahil çizgileri
ax.add_feature(cfeature.LAND, facecolor='lightyellow')  # Karayı açık sarı yap
ax.add_feature(cfeature.LAKES, facecolor='lightblue')  # Gölleri mavi yap

# Şehir isimlerini ekleme
for city, coord in city_coordinates.items():
    ax.plot(coord[0], coord[1], marker='o', color='red', markersize=5, transform=ccrs.PlateCarree())
    ax.text(coord[0] + 0.2, coord[1], city, transform=ccrs.PlateCarree(), fontsize=10, color='black')

# Hakkâri bölgesinde sıcaklık verisini gösterme
temp_plot = ax.contourf(lon2d_new, lat2d_new, temp_mean_interp, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=50, alpha=0.9)
plt.colorbar(temp_plot, ax=ax, orientation='vertical', label='Sıcaklık (°C)')

# Başlık ve açıklamalar
plt.title("Türkiye Haritası Üzerinde Hakkâri Bölgesi Sıcaklık Dağılımı", fontsize=14)
plt.xlabel("Boylam")
plt.ylabel("Enlem")

# Haritayı gösterme
plt.show()

# Yağış grafiği oluşturma
def plot_precipitation_trends(df_features):
    df_features['tp'] = df_features['tp'] * 1000  # Yağış birimi metre -> milimetre dönüştürüldü
    precipitation_sum = df_features.groupby(['year', 'season'])['tp'].sum().unstack()

    # Çubuk grafik oluşturma
    precipitation_sum.plot(kind='bar', figsize=(14, 8), width=0.8, edgecolor='black')
    plt.title("1940'tan Günümüze Mevsimlik Yağış Miktarı")
    plt.ylabel('Toplam Yağış (mm)')
    plt.xlabel('Yıl')
    plt.xticks(rotation=90, fontsize=8)
    plt.legend(title='Mevsim')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

plot_precipitation_trends(df_features)

# Sıcaklık grafiği oluşturma
def plot_seasonal_trends_all_years(df_features):
    seasonal_avg = df_features.groupby(['year', 'season'])['t2m'].mean().unstack()

    # Çubuk grafik oluşturma
    seasonal_avg.plot(kind='bar', figsize=(14, 8), width=0.8, edgecolor='black')
    plt.title("1940'tan Günümüze Mevsimlik Sıcaklık Ortalamaları")
    plt.ylabel('Ortalama Sıcaklık (°C)')
    plt.xlabel('Yıl')
    plt.xticks(rotation=90, fontsize=8)
    plt.legend(title='Mevsim')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

plot_seasonal_trends_all_years(df_features)
