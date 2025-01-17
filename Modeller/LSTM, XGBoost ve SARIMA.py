import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# NetCDF veri yükleme
file_temp = "data_H/data_0_m.nc"  # Sıcaklık dosyası
data_0_m = xr.open_dataset(file_temp)
file_precip = "data_H/data_1_m.nc"  # Yağış dosyası
data_1_m = xr.open_dataset(file_precip)

# NetCDF içeriğini kontrol et
print("Sıcaklık dosyası içeriği:")
print(data_0_m)
print("Yağış dosyası içeriği:")
print(data_1_m)

# Sıcaklığı Celsius'a, Yağışı mm'ye dönüştürme
t2m_celsius = data_0_m['t2m'] - 273.15
tp_mm = data_1_m['tp'] * 1000

# Zaman eksenini alma
print("Zaman ekseni kontrolü:")
time_t2m = data_0_m['valid_time']
time_tp = data_1_m['valid_time']
print("Sıcaklık zaman ekseni:", time_t2m)
print("Yağış zaman ekseni:", time_tp)

# Ortalama sıcaklık ve toplam yağış hesaplama
mean_t2m_celsius = t2m_celsius.mean(dim=['latitude', 'longitude'], skipna=True)
total_tp_mm = tp_mm.sum(dim=['latitude', 'longitude'], skipna=True)

# Veriyi DataFrame'e dönüştürme
temperature_data = pd.DataFrame({
    'time': pd.to_datetime(time_t2m.values),
    'temperature_celsius': mean_t2m_celsius.values
})
precipitation_data = pd.DataFrame({
    'time': pd.to_datetime(time_tp.values),
    'precipitation_mm': total_tp_mm.values
})

# Zaman indekslerini ayarlama
temperature_data.set_index('time', inplace=True)
precipitation_data.set_index('time', inplace=True)

# Tüm zaman dilimlerini genişletme
full_time_index = pd.date_range(
    start=min(temperature_data.index.min(), precipitation_data.index.min()),
    end=max(temperature_data.index.max(), precipitation_data.index.max()),
    freq='M'  # Aylık veriler için
)

temperature_data = temperature_data.reindex(full_time_index)
precipitation_data = precipitation_data.reindex(full_time_index)

# Eksik değerleri kontrol et
print("Missing values in temperature data:")
print(temperature_data.isnull().sum())
print("Missing values in precipitation data:")
print(precipitation_data.isnull().sum())

# Verileri birleştirme
combined_data = pd.concat([temperature_data, precipitation_data], axis=1)
print("Combined data shape:", combined_data.shape)

# Combined data'yi Excel dosyasına kaydetme
combined_data.to_excel("combined_data_raw.xlsx", index=True)
print("Combined data saved to 'combined_data_raw.xlsx'.")

# Boş veri kontrolü
if combined_data.isnull().sum().sum() > 0:
    print("Combined dataset contains missing values.")

# Veriyi eğitim ve test setlerine ayırma
train_data, test_data = train_test_split(combined_data.dropna(), test_size=0.2, shuffle=False)
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Normalizasyon için scaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Zaman serisi için veri hazırlama fonksiyonu
def create_lstm_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :-1])  # Son sütun dışında hepsi (giriş)
        y.append(data[i + look_back, -1])      # Son sütun (hedef)
    return np.array(X), np.array(y)

# LSTM için zaman serisi veri hazırlığı
look_back = 12  # 12 aylık geçmişe bakarak tahmin yapacağız
X_train, y_train = create_lstm_dataset(train_scaled, look_back)
X_test, y_test = create_lstm_dataset(test_scaled, look_back)

# LSTM modelini tanımlama
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Tahminler
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Performans değerlendirme
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

# Tahminlerin görselleştirilmesi
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.title('LSTM Predictions vs Actual Values')
plt.xlabel('Time Step')
plt.ylabel('Scaled Value')
plt.legend()
plt.show()

# XGBoost Modeli
train_df = pd.DataFrame(train_scaled, columns=['temperature', 'precipitation'])
test_df = pd.DataFrame(test_scaled, columns=['temperature', 'precipitation'])

X_train_xgb = train_df[['temperature']]
y_train_xgb = train_df['precipitation']
X_test_xgb = test_df[['temperature']]
y_test_xgb = test_df['precipitation']

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
xgb_model.fit(X_train_xgb, y_train_xgb)

xgb_predictions = xgb_model.predict(X_test_xgb)

xgb_rmse = np.sqrt(mean_squared_error(y_test_xgb, xgb_predictions))
print(f"XGBoost RMSE: {xgb_rmse}")

# SARIMA Modeli
sarima_model = SARIMAX(train_df['precipitation'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_results = sarima_model.fit()

sarima_predictions = sarima_results.forecast(steps=len(test_df))
sarima_rmse = np.sqrt(mean_squared_error(test_df['precipitation'], sarima_predictions))
print(f"SARIMA RMSE: {sarima_rmse}")

# Modellerin Tahminlerini Görselleştirme
plt.figure(figsize=(14, 7))
plt.plot(y_test_xgb, label='Actual')
plt.plot(xgb_predictions, label='XGBoost Predicted')
plt.plot(sarima_predictions, label='SARIMA Predicted', alpha=0.7)
plt.title('Model Predictions vs Actual Values')
plt.xlabel('Time Step')
plt.ylabel('Precipitation (Scaled)')
plt.legend()
plt.show()