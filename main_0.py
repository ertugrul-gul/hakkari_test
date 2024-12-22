# Gerekli kütüphaneleri yükle
import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np

# NetCDF dosya yolu
project_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join("data_H", "monthly", "Data_0.nc")
file_path = os.path.normpath(os.path.join(project_dir, relative_path))

# Dosyanın varlığını kontrol et
if os.path.exists(file_path):
    print("Dosya bulundu. Yükleniyor...")
    ds = xr.open_dataset(file_path)
    
    # Dosya içeriğini inceleme
    print("Değişkenler:", list(ds.data_vars.keys()))
    print("Boyutlar:", list(ds.dims.keys()))
else:
    raise FileNotFoundError("Dosya bulunamadı. Dosya yolunu kontrol edin: {}".format(file_path))

# Değişkenlerin temel özelliklerini incele
print(ds['u10'])
print(ds['v10'])
print(ds['t2m'])

# İlk zaman dilimindeki sıcaklık verilerini görselleştirin
ds['t2m'].isel(valid_time=0).plot(cmap='coolwarm')
plt.title("2 Metre Sıcaklık Haritası (Kelvin)")
plt.show()

# Zaman boyutuna göre ortalama sıcaklık
ds['t2m'].mean(dim=["latitude", "longitude"]).plot()
plt.title("Zamana Göre Ortalama Sıcaklık (Kelvin)")
plt.ylabel("Sıcaklık (K)")
plt.show()

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
