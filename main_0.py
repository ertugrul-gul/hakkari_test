# Gerekli kütüphaneleri yükle
import xarray as xr
import os

# NetCDF dosya yolu
file_path = "C:/Users/ertu_/Desktop/test/data_H/Data_0.nc"

# Dosyanın varlığını kontrol et
if os.path.exists(file_path):
    print("Dosya bulundu. Yükleniyor...")
    ds = xr.open_dataset(file_path)
    
    # Dosya içeriğini inceleme
    print("Değişkenler:", list(ds.data_vars.keys()))
    print("Boyutlar:", list(ds.dims.keys()))
else:
    print("Dosya bulunamadı. Dosya yolunu kontrol edin.")

# Değişkenlerin temel özelliklerini inceleyin
print(ds['u10'])
print(ds['v10'])
print(ds['t2m'])


import matplotlib.pyplot as plt

# İlk zaman dilimindeki sıcaklık verilerini görselleştirin
ds['t2m'].isel(valid_time=0).plot(cmap='coolwarm')
plt.title("2 Metre Sıcaklık Haritası (Kelvin)")
plt.show()


# Zaman boyutuna göre ortalama sıcaklık
ds['t2m'].mean(dim=["latitude", "longitude"]).plot()
plt.title("Zamana Göre Ortalama Sıcaklık (Kelvin)")
plt.ylabel("Sıcaklık (K)")
plt.show()


import numpy as np

# İlk zaman dilimi için u10 ve v10 verilerini alın
u = ds['u10'].isel(valid_time=0)
v = ds['v10'].isel(valid_time=0)

# Rüzgar hızı hesaplama
wind_speed = np.sqrt(u**2 + v**2)

# Rüzgar yönü hesaplama
wind_direction = np.arctan2(v, u) * (180 / np.pi)

# Rüzgar Haritası Çizimi
plt.quiver(ds['longitude'], ds['latitude'], u, v)
plt.title("Rüzgar Yönü ve Hızı")
plt.xlabel("Boylam")
plt.ylabel("Enlem")
plt.show()


# Zaman boyutunu datetime formatına çevir
time_series = ds['t2m'].mean(dim=["latitude", "longitude"]).to_series()

# Aylık ortalamaları hesapla
monthly_avg = time_series.resample('M').mean()

# Aylık sıcaklık grafiği
monthly_avg.plot(figsize=(10, 6))
plt.title("Aylık Ortalama Sıcaklık (Kelvin)")
plt.xlabel("Tarih")
plt.ylabel("Sıcaklık (K)")
plt.show()
