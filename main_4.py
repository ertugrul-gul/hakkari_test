# Gerekli Kütüphaneler
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# NetCDF dosya yolunu belirtin
file_path = "C:/Users/ertu_/Desktop/test/data_H/Data_0.nc"

# NetCDF dosyasını yükle
ds = xr.open_dataset(file_path)

# Verileri DataFrame'e dönüştür
df = ds[['t2m', 'u10', 'v10']].to_dataframe().reset_index()
df['valid_time'] = pd.to_datetime(df['valid_time'], errors='coerce')

# Geçersiz tarihleri ve satırları kaldır
valid_dates = df.dropna(subset=['valid_time', 'latitude', 'longitude']).sort_values('valid_time').reset_index(drop=True)

# Rüzgar hızını ve yönünü hesapla
valid_dates['wind_speed'] = np.sqrt(valid_dates['u10']**2 + valid_dates['v10']**2)
valid_dates['wind_dir'] = np.arctan2(valid_dates['v10'], valid_dates['u10']) * (180 / np.pi)
# 0-360 derece aralığına dönüştür
valid_dates['wind_dir'] = (valid_dates['wind_dir'] + 360) % 360

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

# Geleceğe yönelik tahmin aralığı oluştur (Şimdilik XGBoost'taki gibi kalsın, ARIMA için güncellenecek)
future_dates = pd.date_range(start="2025-01-01", end="2100-12-31", freq='Q')
X_future = pd.DataFrame({
    'wind_speed': np.random.choice(X_test['wind_speed'], size=len(future_dates)),
    'wind_dir': np.random.choice(X_test['wind_dir'], size=len(future_dates))
})

#----------------------------------------
# Random Forest Modeli
#----------------------------------------
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Test seti üzerinde tahminler
forecast_rf = model_rf.predict(X_test)
forecast_rf_future = model_rf.predict(X_future)

forecast_rf = pd.DataFrame({
    'valid_time': test_data['valid_time'].values,
    'predicted_temperature': forecast_rf
})
forecast_rf = forecast_rf.set_index('valid_time')

# Gelecekteki tahminleri ekle
future_forecast_rf = pd.DataFrame({
    'valid_time': future_dates,
    'predicted_temperature': forecast_rf_future
})
future_forecast_rf = future_forecast_rf.set_index('valid_time')

#----------------------------------------
# ARIMA Modeli
#----------------------------------------
# Eğitim verilerini mevsimlik olarak ayrıştır
y_train_ts = y_train.copy()
y_train_ts.index = pd.to_datetime(train_data['valid_time'], errors='coerce')
y_train_ts = y_train_ts.resample('M').mean() # Aylık ortalamaya indirge

# Durağanlık kontrolü (Augmented Dickey-Fuller Testi)
result = adfuller(y_train_ts)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# ACF ve PACF grafikleri
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(y_train_ts, lags=20, ax=axes[0])
plot_pacf(y_train_ts, lags=20, ax=axes[1])
plt.show()

# ARIMA modelini oluştur (p, d, q değerlerini ACF ve PACF grafiklerine göre belirleyin)
# Örnek olarak (p=2, d=1, q=2) ve mevsimsellik için (P=1, D=1, Q=1, s=12) kullanılmıştır.
# Bu değerleri kendi verinize göre ayarlamanız gerekir.
order = (2, 1, 2)
seasonal_order = (1, 1, 1, 12)
model_arima = SARIMAX(y_train_ts, order=order, seasonal_order=seasonal_order)
model_arima_fit = model_arima.fit(disp=False)

# Test verisi için tahmin aralığını oluştur
test_dates = pd.date_range(start="2000-01-01", end="2024-12-31", freq='M')

# Test seti için tahminler
forecast_arima = model_arima_fit.predict(start=len(y_train_ts), end=len(y_train_ts) + len(test_dates)-1)
forecast_arima.index = test_dates

# Gelecekteki tahminler için dinamik tahminler kullan (önceki tahminleri kullanarak)
forecast_arima_future = model_arima_fit.predict(start=len(y_train_ts) + len(test_dates), end=len(y_train_ts) + len(test_dates) + len(future_dates)-1, dynamic=True)
forecast_arima_future.index = future_dates

#----------------------------------------
# Modellerin Değerlendirilmesi ve Grafik
#----------------------------------------

# Gerçek değerlerin mevsimlik ortalamasını al
actual_seasonal = y_test.copy()
actual_seasonal.index = pd.to_datetime(test_data['valid_time'], errors='coerce')
actual_seasonal = actual_seasonal.resample('Q').mean()

# Tahminlerin mevsimlik ortalamasını al (Random Forest)
tahmin_seasonal_rf = pd.concat([forecast_rf, future_forecast_rf]).resample('Q').mean()

# Tahminlerin mevsimlik ortalamasını al (ARIMA)
tahmin_seasonal_arima = pd.concat([forecast_arima, forecast_arima_future]).resample('Q').mean()

# Model değerlendirmesi (Test seti üzerinde)
y_pred_rf = model_rf.predict(X_test)

# ARIMA için: actual_seasonal'ı reindex ile düzenle
# ARIMA tahminlerinin başlangıç ve bitiş tarihlerini al
arima_start_date = forecast_arima.index.min()
arima_end_date = forecast_arima.index.max()

# actual_seasonal'ı ARIMA tahminlerinin tarih aralığına göre filtrele
actual_seasonal = actual_seasonal[(actual_seasonal.index >= arima_start_date) & (actual_seasonal.index <= arima_end_date)]

# actual_seasonal'ı forecast_arima.index ile aynı indekse sahip olacak şekilde yeniden indeksle
actual_seasonal = actual_seasonal.reindex(forecast_arima.index)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

mae_arima = mean_absolute_error(actual_seasonal, forecast_arima)
rmse_arima = np.sqrt(mean_squared_error(actual_seasonal, forecast_arima))
r2_arima = r2_score(actual_seasonal, forecast_arima)

print("Random Forest:")
print(f"MAE: {mae_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R-kare: {r2_rf:.2f}")

print("\nARIMA:")
print(f"MAE: {mae_arima:.2f}")
print(f"RMSE: {rmse_arima:.2f}")
print(f"R-kare: {r2_arima:.2f}")

# Grafik Çizimi
plt.figure(figsize=(15, 8), dpi=100)
plt.plot(actual_seasonal.index, actual_seasonal, label="Gerçek Mevsimlik Ortalama", color='blue', alpha=0.7)
plt.plot(tahmin_seasonal_rf.index, tahmin_seasonal_rf, label="Random Forest Mevsimlik Tahmini", color='green', alpha=0.7)
plt.plot(tahmin_seasonal_arima.index, tahmin_seasonal_arima, label="ARIMA Mevsimlik Tahmini", color='orange', alpha=0.7)

plt.title("2000-2100 Mevsimlik Ortalama Sıcaklık Tahmini")
plt.xlabel("Yıllar")
plt.ylabel("Sıcaklık (°C)")

# Tarih formatlama
plt.gca().xaxis.set_major_locator(mdates.YearLocator(10))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Gösterge
plt.legend()

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()