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
data_path = 'data_H/final_merged_data.nc'  # Dosya yolu
data = xr.open_dataset(data_path)

# Değişkenleri birleştirme
df = data.to_dataframe().reset_index()

# Tarih sütununu geçici olarak kaldırma
time_column = df['time']
df = df.drop(columns=['time'])

# Eksik Değerleri Doldurma
imputer = SimpleImputer(strategy='mean')
filled_data = imputer.fit_transform(df.dropna(axis=1, how='all'))
df = pd.DataFrame(filled_data, columns=df.columns)

# Tarih sütununu geri ekleme
df['time'] = time_column

# Tarih sütunu oluşturma
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

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
        X.append(data[i:i + sequence_length, :-1])  # Tüm değişkenler (son sütun hariç)
        y.append(data[i + sequence_length, -1])    # Sıcaklık hedef değişken
    return np.array(X), np.array(y)

sequence_length = 30  # 30 günlük verilerden tahmin yap

train_winter = seasonal_split(train, 'winter')
X_train, y_train = create_sequences(train_winter.values, sequence_length)

validation_winter = seasonal_split(validation, 'winter')
X_validation, y_validation = create_sequences(validation_winter.values, sequence_length)

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
model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=20, batch_size=32)

# 2100 Yılı Tahmini
test_data = train_winter[-sequence_length:].values  # Son 30 günlük veri
predictions = []

for _ in range(365):
    X_test = test_data[-sequence_length:, :-1].reshape(1, sequence_length, -1)
    prediction = model.predict(X_test)
    predictions.append(prediction[0, 0])
    next_row = np.append(test_data[-1, 1:], prediction)
    test_data = np.vstack([test_data, next_row])

# Tahminlerin Mevsimlere Göre Ayrılması
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
predicted_df = pd.DataFrame(predictions, index=pd.date_range('2100-01-01', '2100-12-31'))

# Sonuçları Görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(predicted_df, label='Tahmin Edilen Sıcaklık (2100)')
plt.xlabel('Tarih')
plt.ylabel('Sıcaklık')
plt.title('2100 Yılı Günlük Sıcaklık Tahmini')
plt.legend()
plt.show()
