# Gerekli Kütüphaneler
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# NetCDF dosya yolunu belirtin
file_path = "C:/Users/ertu_/Desktop/test/data_H/Data_0.nc"

# NetCDF dosyasını yükle
ds = xr.open_dataset(file_path)

# Verileri DataFrame'e dönüştür
df = ds[['t2m', 'u10', 'v10']].to_dataframe().reset_index()
df['valid_time'] = pd.to_datetime(df['valid_time'], errors='coerce')

# Geçersiz tarihleri kaldır
valid_dates = df.dropna(subset=['valid_time']).sort_values('valid_time').reset_index(drop=True)

# Rüzgar hızını ve yönünü hesapla
valid_dates['wind_speed'] = np.sqrt(valid_dates['u10']**2 + valid_dates['v10']**2)
valid_dates['wind_dir'] = np.arctan2(valid_dates['v10'], valid_dates['u10']) * (180 / np.pi)

# Sıcaklığı Kelvin'den Celsius'a dönüştür
valid_dates['temperature'] = valid_dates['t2m'] - 273.15

# Gereksiz sütunları kaldır
valid_dates.drop(columns=['t2m', 'u10', 'v10'], inplace=True)

# Girdiler ve hedef değişken
X = valid_dates[['wind_speed', 'wind_dir']]
y = valid_dates['temperature']

# Eğitim ve test veri setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Modeli
model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_xgb.fit(X_train, y_train)

# Tahminleri Yıllık Ortalama ile Hesaplama
forecast_xgb = model_xgb.predict(X_test)
forecast_xgb = pd.DataFrame({
    'valid_time': pd.date_range(start=pd.Timestamp("1950-01-01"), periods=len(forecast_xgb), freq='D'),
    'predicted_temperature': forecast_xgb
})
forecast_xgb = forecast_xgb.set_index('valid_time')

# Gerçek veriler için 2024'e kadar olan tarihleri koru
y_test.index = pd.date_range(start=pd.Timestamp("1950-01-01"), periods=len(y_test), freq='D')
y_test = y_test[y_test.index <= pd.Timestamp("2024-12-31")]

# Gerçek değerlerin yıllık ortalamasını al
actual_annual = y_test.resample('Y').mean()

# Tahminlerin yıllık ortalamasını al
tahmin_annual = forecast_xgb.resample('Y').mean()

# Grafik Çizimi
plt.figure(figsize=(15, 8))
plt.plot(actual_annual.index, actual_annual, label="Gerçek Yıllık Ortalama", color='blue', alpha=0.7)
plt.plot(tahmin_annual.index, tahmin_annual, label="XGBoost Yıllık Tahmini", color='red', alpha=0.7)
plt.title("1950-2100 Yıllık Ortalama Sıcaklık Tahmini - XGBoost")
plt.xlabel("Yıllar")
plt.ylabel("Sıcaklık (°C)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
