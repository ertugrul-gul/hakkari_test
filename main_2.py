# Gerekli Kütüphaneler
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

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

# Girdiler ve hedef değişken
X = valid_dates[['wind_speed', 'wind_dir']]
y = valid_dates['temperature']

# Eğitim ve test veri setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Performans Değerlendirme Fonksiyonu
def evaluate_model_with_plot(y_true, y_pred, model_name, y_test_index):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    plt.figure(figsize=(15, 8))
    plt.plot(y_test_index, y_true, label="Gerçek Değerler", color='blue', alpha=0.7)
    plt.plot(y_test_index, y_pred, label=f"{model_name} Tahmini", color='red', alpha=0.7)
    plt.title(f"Gerçek vs Tahmini Sıcaklık - {model_name}")
    plt.xlabel("Tarih")
    plt.ylabel("Sıcaklık (°C)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 1. SARIMA Modeli (Güncelleme)
try:
    y_train.index = pd.date_range(start=pd.Timestamp("1950-01-01"), periods=len(y_train), freq='D', name="valid_time")
    model_sarima = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
    result_sarima = model_sarima.fit(disp=False)
    forecast_sarima = result_sarima.get_forecast(steps=len(y_test)).predicted_mean
    forecast_sarima.index = y_test.index
    
    evaluate_model_with_plot(y_test, forecast_sarima, "SARIMA", y_test.index)
except Exception as e:
    print(f"SARIMA Model Hatası: {e}")

# 2. XGBoost Modeli
model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_xgb.fit(X_train, y_train)
forecast_xgb = model_xgb.predict(X_test)
evaluate_model_with_plot(y_test, forecast_xgb, "XGBoost", y_test.index)
