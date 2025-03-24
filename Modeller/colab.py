
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Veri yükleme ve tarih dönüştürme
df = pd.read_csv('../base_data/hakkari_0.csv')
df['valid_time'] = pd.to_datetime(df['valid_time'], format='%Y-%m-%d')

# Ölçeklendirme için sayısal sütunları seç (rüzgar yönü dahil)
numerical_cols = ['sp', 'u10', 'v10', 't2m', 'tp', 'ws', 'ws']

# Verileri tarihe göre bölme
train_data = df[df['valid_time'] < '1995-01-01']
test_data = df[(df['valid_time'] >= '1995-01-01') & (df['valid_time'] < '2020-01-01')]

# Özellikleri ve hedefi ayırma
X_train = train_data[numerical_cols].values
y_train = train_data['t2m'].values  # Hedef değişken: t2m (sıcaklık)
X_test = test_data[numerical_cols].values
y_test = test_data['t2m'].values

# Verileri 0-1 aralığına ölçeklendirme
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# LSTM için verileri yeniden şekillendirme (örnekler, zaman adımları, özellikler)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# LSTM modelini oluşturma
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1))

# Modeli derleme
model.compile(loss='mean_squared_error', optimizer='adam')

# Modeli eğitme
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Test verileri üzerinde tahmin yapma
predictions = model.predict(X_test)

# ... (Tahminlerin değerlendirilmesi ve sonuçların yorumlanması)


import numpy
print(numpy.__file__)  # NumPy'nin yüklü olduğu yolu gösterir




import numpy
print(hasattr(numpy, "_ARRAY_API"))
