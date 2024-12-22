# Gerekli Kütüphaneler
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.dates as mdates
import os

# NetCDF dosya yolunu platform bağımsız yap
project_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join("data_H", "monthly", "Data_0.nc")
file_path = os.path.join(project_dir, relative_path)

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

# 1950-2000 yılları arasındaki verilerle eğitim setini oluştur
train_data = valid_dates[(valid_dates['valid_time'] >= "1950-01-01") & (valid_dates['valid_time'] < "2000-01-01")]
X_train = train_data[['wind_speed', 'wind_dir']]
y_train = train_data['temperature']

# 2000-2024 yılları arasındaki verilerle test setini oluştur
test_data = valid_dates[(valid_dates['valid_time'] >= "2000-01-01") & (valid_dates['valid_time'] <= "2024-12-31")]
X_test = test_data[['wind_speed', 'wind_dir']]
y_test = test_data['temperature']

# Geleceğe yönelik tahmin aralığı oluştur
future_dates = pd.date_range(start="2025-01-01", end="2100-12-31", freq='Q')
X_future = pd.DataFrame({
    'wind_speed': np.random.uniform(X_test['wind_speed'].min(), X_test['wind_speed'].max(), len(future_dates)),
    'wind_dir': np.random.uniform(X_test['wind_dir'].min(), X_test['wind_dir'].max(), len(future_dates))
})

# XGBoost Modeli
model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_xgb.fit(X_train, y_train)

# Tahminleri Mevsimlik Ortalama ile Hesaplama
forecast_xgb = model_xgb.predict(X_test)
forecast_xgb_future = model_xgb.predict(X_future)

forecast_xgb = pd.DataFrame({
    'valid_time': test_data['valid_time'].values,
    'predicted_temperature': forecast_xgb
})
forecast_xgb['valid_time'] = pd.to_datetime(forecast_xgb['valid_time'], errors='coerce')
forecast_xgb = forecast_xgb.set_index('valid_time')

# Gelecekteki tahminleri ekle
future_forecast_xgb = pd.DataFrame({
    'valid_time': future_dates,
    'predicted_temperature': forecast_xgb_future
})
future_forecast_xgb = future_forecast_xgb.set_index('valid_time')

# Gerçek değerlerin mevsimlik ortalamasını al
actual_seasonal = y_test.copy()
actual_seasonal.index = pd.to_datetime(test_data['valid_time'], errors='coerce')
actual_seasonal = actual_seasonal.resample('Q').mean()

# Tahminlerin mevsimlik ortalamasını al
tahmin_seasonal = pd.concat([forecast_xgb, future_forecast_xgb]).resample('Q').mean()

# Grafik Çizimi
plt.figure(figsize=(15, 8))
plt.plot(actual_seasonal.index, actual_seasonal, label="Gerçek Mevsimlik Ortalama", color='blue', alpha=0.7)
plt.plot(tahmin_seasonal.index, tahmin_seasonal, label="XGBoost Mevsimlik Tahmini", color='red', alpha=0.7)
plt.title("2000-2100 Mevsimlik Ortalama Sıcaklık Tahmini - XGBoost")
plt.xlabel("Yıllar")
plt.ylabel("Sıcaklık (°C)")

# Tarih formatlama
plt.gca().xaxis.set_major_locator(mdates.YearLocator(10))  # Her 10 yılda bir
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Yılları göster

# Mevsimleri göstergede tanımla
plt.legend(title="Mevsimler", labels=["Gerçek Mevsimlik Ortalama - Kış/İlkbahar/Yaz/Sonbahar", "XGBoost Mevsimlik Tahmini - Kış/İlkbahar/Yaz/Sonbahar"])

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
