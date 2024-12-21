# Gerekli kütüphaneleri yükle
import xarray as xr
import pandas as pd

# NetCDF dosya yolunu belirtin
file_path = "C:/Users/ertu_/Desktop/test/data_H/Data_0.nc"

# NetCDF dosyasını yükle
ds = xr.open_dataset(file_path)

# Verileri DataFrame'e dönüştür
df = ds[['t2m', 'u10', 'v10']].to_dataframe().reset_index()

# İlk satırları görüntüle
print(df.head())




# Rüzgar hızını ve yönünü hesapla
import numpy as np

# Rüzgar hızını hesapla
df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)

# Rüzgar yönünü hesapla
df['wind_dir'] = np.arctan2(df['v10'], df['u10']) * (180 / np.pi)

# Sıcaklığı Kelvin'den Celsius'a dönüştür
df['temperature'] = df['t2m'] - 273.15

# Gereksiz sütunları kaldır
df.drop(columns=['t2m', 'u10', 'v10'], inplace=True)
print(df.head())



from sklearn.model_selection import train_test_split

# Girdiler ve hedef değişken
X = df[['wind_speed', 'wind_dir']]
y = df['temperature']

# Eğitim ve test veri setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Model oluştur ve eğit
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahminleri yap
predictions = model.predict(X_test)

# Performansı değerlendirme
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")



import matplotlib.pyplot as plt

# Gerçek ve tahmini sıcaklıkları karşılaştırma
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.6, color='blue')
plt.title("Gerçek vs Tahmini Sıcaklık (°C)")
plt.xlabel("Gerçek Sıcaklık")
plt.ylabel("Tahmini Sıcaklık")
plt.grid(True)
plt.show()
