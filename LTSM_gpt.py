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

# Veri Yükleme
# İki dosyayı birleştirerek işlem yapacağız
data_path_0 = 'data_H/data_0.nc'
data_path_1 = 'data_H/data_1.nc'
data_0 = xr.open_dataset(data_path_0)
data_1 = xr.open_dataset(data_path_1)
data = xr.concat([data_0, data_1], dim='time')

# Değişkenleri birleştirme
df = data.to_dataframe().reset_index()

# Tarih sütununu işleme
df['year'] = pd.to_datetime(df['time']).dt.year
df['month'] = pd.to_datetime(df['time']).dt.month
df['day'] = pd.to_datetime(df['time']).dt.day
df['dayofyear'] = pd.to_datetime(df['time']).dt.dayofyear
df['week'] = pd.to_datetime(df['time']).dt.isocalendar().week
df['weekday'] = pd.to_datetime(df['time']).dt.weekday
time_column = df['time']
df = df.drop(columns=['time'])

# Eksik Değerleri Doldurma
imputer = SimpleImputer(strategy='mean')
filled_data = imputer.fit_transform(df.select_dtypes(include=[np.number]))
df = pd.DataFrame(filled_data, columns=df.select_dtypes(include=[np.number]).columns)

# Tarih sütununu geri ekleme
df['time'] = time_column

# Tarih sütunu oluşturma
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Tarih dizinini sıralama
df = df.sort_index()

# Değişkenlerin normalize edilmesi
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)

# Eğitim, Doğrulama ve Test Verilerinin Ayrılması
train = df_scaled['1940':'2000']
validation = df_scaled['2001':'2024']

# Veriyi Mevsimlere Göre Ayırma
def seasonal_split(data, season):
    if season == 'winter':
        return data[data.index.month.isin([12, 1, 2])]
    elif season == 'spring':
        return data[data.index.month.isin([3, 4, 5])]
    elif season == 'summer':
        return data[data.index.month.isin([6, 7, 8])]
    elif season == 'autumn':
        return data[data.index.month.isin([9, 10, 11])]

# LSTM için Veri Hazırlığı
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :])  # Tüm değişkenler
        y.append(data[i + sequence_length, -1])  # Sıcaklık hedef değişken
    return np.array(X), np.array(y)

sequence_length = 30  # 30 günlük verilerden tahmin yap

train_winter = seasonal_split(train, 'winter')
X_train, y_train = create_sequences(train_winter.values, sequence_length)

validation_winter = seasonal_split(validation, 'winter')
if not validation_winter.empty:
    X_validation, y_validation = create_sequences(validation_winter.values, sequence_length)
else:
    X_validation, y_validation = None, None

# Giriş boyutlarının kontrolü
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
if X_validation is not None:
    print(f"X_validation shape: {X_validation.shape}, y_validation shape: {y_validation.shape}")
else:
    print("Validation data is empty, skipping validation.")

# LSTM Modeli Oluşturma
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Modeli Eğitme
if X_validation is not None:
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=20, batch_size=32)
else:
    model.fit(X_train, y_train, epochs=20, batch_size=32)

# 2100 Yılı Tahmini
test_data = train_winter[-sequence_length:].values  # Son 30 günlük veri
predictions = []

for _ in range(365):
    X_test = test_data[-sequence_length:, :].reshape(1, sequence_length, -1)
    prediction = model.predict(X_test)
    predictions.append(prediction[0, 0])
    next_row = np.append(test_data[-1, :-1], prediction)
    test_data = np.vstack([test_data, next_row])

# Tahminlerin Ters Dönüşümü
predictions_array = np.array(predictions).reshape(-1, 1)
predictions_scaled = np.zeros((predictions_array.shape[0], df.shape[1]))
predictions_scaled[:, -1] = predictions_array[:, 0]
predictions_inverse = scaler.inverse_transform(predictions_scaled)
predicted_df = pd.DataFrame(predictions_inverse[:, -1],
                            index=pd.date_range('2100-01-01', '2100-12-31'),
                            columns=['Predicted Temperature'])

# Sonuçları Görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(predicted_df, label='Tahmin Edilen Sıcaklık (2100)')
plt.xlabel('Tarih')
plt.ylabel('Sıcaklık')
plt.title('2100 Yılı Günlük Sıcaklık Tahmini')
plt.legend()
plt.show()
