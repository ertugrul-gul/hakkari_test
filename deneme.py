import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Dropout, Input  # Input eklenmeli!

from sklearn.preprocessing import MinMaxScaler

# NetCDF dosya yolları
data_dir = "data_H/hakkari.csv"
df = pd.read_csv(data_dir)

# İlk birkaç satırı göster
df.info()
df.describe()

df['valid_time'] = pd.to_datetime(df['valid_time'])
df.set_index('valid_time', inplace=True)

# Görselleştirme
plt.figure(figsize=(15, 6))
df['t2m'].plot()
plt.title("Mean Temperature")
plt.show()

plt.figure(figsize=(15, 6))
df['tp'].plot()
plt.title("Mean Pressure")
plt.show()

# Veriyi hazırlama
dataset = df["t2m"]
dataset = pd.DataFrame(dataset)
data = dataset.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(data))

train_size = int(len(data) * 0.75)
test_size = len(data) - train_size
print("Train Size:", train_size, "Test Size:", test_size)

train_data = scaled_data[0:train_size, :]

# Eğitim seti oluşturma
x_train, y_train = [], []
time_steps = 60
n_cols = 1

for i in range(time_steps, len(train_data)):
    x_train.append(train_data[i-time_steps:i, :n_cols])
    y_train.append(train_data[i, :n_cols])

x_train, y_train = np.array(x_train), np.array(y_train)

# Şekil uyumu için reshape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], n_cols))

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# **📌 Model oluşturma (Hatalar Düzeltildi!)**
model = Sequential([
    Input(shape=(x_train.shape[1], x_train.shape[2])),  # `X_train` yerine `x_train`
    LSTM(50, return_sequences=True),  
    LSTM(64, return_sequences=False),  
    Dense(32),  
    Dense(16),  
    Dense(n_cols)  
])

# **📌 Model derleme (Hata Düzeltildi!)**
model.compile(optimizer='adam', loss='mse', metrics=["mean_absolute_error"])

model.summary()

# Fitting the LSTM to the Training set
import os
import json

# 📌 Eğitim geçmişi dosyası
history_file = "training_history.json"

# 📌 Modeli eğitme kontrolü
if os.path.exists(history_file):
    # Eğer eğitim geçmişi dosyası varsa, onu yükle ve eğitimi atla
    with open(history_file, "r") as f:
        history_data = json.load(f)
    
    print("\nDaha önce eğitilmiş model bulundu. Eğitimi tekrar başlatmıyorum.")
    print(f"Son Epoch Kayıpları: {history_data['loss'][-1]}")
    print(f"Son Epoch MAE: {history_data['mean_absolute_error'][-1]}")

else:
    # Eğer eğitim geçmişi yoksa modeli eğit
    print("\nModel eğitiliyor...")
    history = model.fit(x_train, y_train, epochs=100, batch_size=32)

    # Eğitim bittikten sonra sonuçları ekrana yazdır
    print("\nModel Eğitimi Tamamlandı!")
    print(f"Son Epoch Kayıpları: {history.history['loss'][-1]}")
    print(f"Son Epoch MAE: {history.history['mean_absolute_error'][-1]}")

    # 📌 Eğitim geçmişini kaydet
    with open(history_file, "w") as f:
        json.dump(history.history, f)

#BURADA KALDIM
plt.figure(figsize=(12, 8))
plt.plot(history2.history["loss"])
plt.plot(history2.history["mean_absolute_error"])
plt.legend(['Mean Squared Error','Mean Absolute Error'])
plt.title("Losses")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# Creating a testing set with 60 time-steps and 1 output
time_steps = 60
test_data = scaled_data[train_size - time_steps:, :]

x_test = []
y_test = []
n_cols = 4

for i in range(time_steps, len(test_data)):
    x_test.append(test_data[i-time_steps:i, 0:n_cols])
    y_test.append(test_data[i, 0:n_cols])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], n_cols))

x_test.shape , y_test.shape
# Get Prediction
predictions = model2.predict(x_test)
#inverse y_test scaling
y_test = scaler.inverse_transform(y_test)
RMSE = np.sqrt(np.mean( y_test - predictions )**2).round(2)
RMSE

from datetime import timedelta
def insert_end(Xin, new_input):
    timestep = 60
    for i in range(timestep - 1):
        Xin[:, i, :] = Xin[:, i+1, :]
    Xin[:, timestep - 1, :] = new_input
    return Xin

future = 30
forcast = []
Xin = x_test[-1 :, :, :]
time = []
for i in range(0, future):
    out = model2.predict(Xin, batch_size=5)
    forcast.append(out[0]) 
    print(forcast)
    Xin = insert_end(Xin, out[0, 0]) 
    time.append(pd.to_datetime(df.index[-1]) + timedelta(days=i))
    
forcasted_output = np.asanyarray(forcast)   
forcasted_output = scaler.inverse_transform(forcasted_output)

forcasted_output = pd.DataFrame(forcasted_output)
date = pd.DataFrame(time)
df_result = pd.concat([date,forcasted_output], axis=1)
df_result.columns = "valid_time", 'sp', 'u10', 'v10', 'tp'
df_result.head()

plt.figure(figsize=(20, 10))
plt.title('Next 30 Days')

plt.subplot(2, 2, 1)
plt.plot(df['meantemp'])
plt.plot(df_result.set_index('Date')[['meantemp']])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Temp' ,fontsize=18)

plt.subplot(2, 2, 2)
plt.plot(df['humidity'])
plt.plot(df_result.set_index('Date')[['humidity']])
plt.xlabel('Date', fontsize=18)
plt.ylabel('humidity' ,fontsize=18)

plt.subplot(2, 2, 3)
plt.plot(df['wind_speed'])
plt.plot(df_result.set_index('Date')[['wind_speed']])
plt.xlabel('Date', fontsize=18)
plt.ylabel('wind_speed' ,fontsize=18)

plt.subplot(2, 2, 4)
plt.plot(df['meanpressure'])
plt.plot(df_result.set_index('Date')[['meanpressure']])
plt.xlabel('Date', fontsize=18)
plt.ylabel('meanpressure' ,fontsize=18)

plt.tight_layout()
plt.show()