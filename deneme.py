import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from datetime import timedelta
import os
import json

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.preprocessing import MinMaxScaler

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# =============================================================================
# Veri Yükleme ve Ön İşleme
# =============================================================================

data_dir = "data_H/hakkari.csv"
df = pd.read_csv(data_dir)

df['valid_time'] = pd.to_datetime(df['valid_time'])
df.set_index('valid_time', inplace=True)

# İlk grafikleri çizme
plt.figure(figsize=(15, 6))
df['t2m'].plot()
plt.title("Mean Temperature")
plt.show()

plt.figure(figsize=(15, 6))
df['tp'].plot()
plt.title("Mean Pressure")
plt.show()

# =============================================================================
# Tek Değişkenli Model (t2m)
# =============================================================================

# Veri hazırlama
dataset = df["t2m"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
print("Train Size:", train_size, "Test Size:", test_size)

train_data = scaled_data[:train_size, :]

# Eğitim setini oluşturma
time_steps = 60
n_cols = 1
x_train, y_train = [], []
for i in range(time_steps, len(train_data)):
    x_train.append(train_data[i-time_steps:i, :n_cols])
    y_train.append(train_data[i, :n_cols])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], n_cols)
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Model oluşturma veya yükleme
model_path = "trained_model.keras"
history_file = "training_history.json"

if os.path.exists(model_path) and os.path.exists(history_file):
    model = load_model(model_path)
    with open(history_file, "r") as f:
        history_data = json.load(f)
    # history_data yapısını history.history gibi kullanarak ilerleyelim
    history = history_data
    print("\nDaha önce eğitilmiş model bulundu. Eğitimi tekrar başlatmıyorum.")
    print(f"Son Epoch Kayıpları: {history['loss'][-1]}")
    print(f"Son Epoch MAE: {history['mean_absolute_error'][-1]}")
else:
    model = Sequential([
        Input(shape=(x_train.shape[1], x_train.shape[2])),
        LSTM(50, return_sequences=True),
        LSTM(64, return_sequences=False),
        Dense(32),
        Dense(16),
        Dense(n_cols)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=["mean_absolute_error"])
    model.summary()

    print("\nModel eğitiliyor...")
    history_obj = model.fit(x_train, y_train, epochs=100, batch_size=32)
    history = history_obj.history

    with open(history_file, "w") as f:
        json.dump(history, f)
    model.save(model_path)

    print("\nModel Eğitimi Tamamlandı!")
    print(f"Son Epoch Kayıpları: {history['loss'][-1]}")
    print(f"Son Epoch MAE: {history['mean_absolute_error'][-1]}")

# Eğitim kayıplarını çizme
plt.figure(figsize=(12, 8))
plt.plot(history["loss"], label="Mean Squared Error")
plt.plot(history["mean_absolute_error"], label="Mean Absolute Error")
plt.legend()
plt.title("Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Test verisi oluşturma ve tahmin
test_data = scaled_data[train_size - time_steps:, :]
x_test, y_test = [], []
for i in range(time_steps, len(test_data)):
    x_test.append(test_data[i-time_steps:i, :n_cols])
    y_test.append(test_data[i, :n_cols])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], n_cols)

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

RMSE = np.sqrt(np.mean((y_test - predictions)**2)).round(2)
print("RMSE:", RMSE)

# Tahmin ve gerçek değerleri DataFrame'e aktarma
preds_acts = pd.DataFrame({
    'Predictions': predictions.flatten(),
    'Actuals': y_test.flatten()
})
test_dates = df.index[train_size:]
preds_acts['valid_time'] = test_dates
print(preds_acts.head())

preds_acts.to_excel("output.xlsx", index=False)

plt.figure(figsize=(10, 4))
plt.plot(preds_acts['valid_time'], preds_acts['Predictions'], label='Predictions')
plt.plot(preds_acts['valid_time'], preds_acts['Actuals'], label='Actuals')
plt.legend()
plt.xlabel('Valid Time')
plt.ylabel('Temperature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output_plot.png")
plt.show()

# Orijinal verileri ve tahminleri zaman serisi üzerinde gösterme
train_df = pd.DataFrame(dataset[:train_size, 0:1], columns=["t2m"])
test_df = pd.DataFrame(dataset[train_size:, 0:1], columns=["t2m"])
test_df['Predictions'] = predictions
train_df.index = df.index[:train_size]
test_df.index = df.index[train_size:]

plt.figure(figsize=(16, 6))
plt.title('Temperature Prediction', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Temperature', fontsize=14)
plt.plot(train_df['t2m'], linewidth=1)
plt.plot(test_df['t2m'], linewidth=1)
plt.plot(test_df['Predictions'], linewidth=1)
plt.legend(['Train', 'Test', 'Predictions'])
plt.savefig("output_plot3.png")
plt.show()

# =============================================================================
# Çoklu Değişkenli Model
# =============================================================================

history_file2 = "training_history2.json"
model_file2 = "trained_model2.keras"

n_cols_multi = 4
cols = ['t2m', 'latitude', 'longitude', 'sp', 'u10', 'v10', 'tp']
dataset_multi = df[cols].copy()

if dataset_multi.empty:
    raise ValueError("HATA: 'dataset_multi' değişkeni boş! CSV dosyanızda gerekli sütunlar eksik olabilir.")

data_multi = dataset_multi.values
print(f"data_multi shape: {data_multi.shape}")

# Sadece ilk 4 sütunu kullanarak scaler oluşturma ve uygulama
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.fit(data_multi[:, :n_cols_multi])
scaled_data_multi = scaler_y.transform(data_multi[:, :n_cols_multi])

train_size_multi = int(len(data_multi) * 0.75)
test_size_multi = len(data_multi) - train_size_multi
print("Train Size:", train_size_multi, "Test Size:", test_size_multi)

train_data_multi = scaled_data_multi[:train_size_multi, :]

# Eğitim setini oluşturma
x_train_multi, y_train_multi = [], []
for i in range(time_steps, len(train_data_multi)):
    x_train_multi.append(train_data_multi[i-time_steps:i, :n_cols_multi])
    y_train_multi.append(train_data_multi[i, :n_cols_multi])
x_train_multi, y_train_multi = np.array(x_train_multi), np.array(y_train_multi)
x_train_multi = x_train_multi.reshape(x_train_multi.shape[0], x_train_multi.shape[1], n_cols_multi)
print("x_train_multi shape:", x_train_multi.shape, "y_train_multi shape:", y_train_multi.shape)

# Model oluşturma veya yükleme
if os.path.exists(history_file2) and os.path.exists(model_file2):
    model2 = load_model(model_file2)
    with open(history_file2, "r") as f:
        history_data2 = json.load(f)
    history2 = history_data2  # history2'yi tanımlıyoruz
    print("\nDaha önce eğitilmiş model bulundu. Eğitimi tekrar başlatmıyorum.")
    print(f"Son Epoch Kayıpları: {history2['loss'][-1]}")
    print(f"Son Epoch MAE: {history2['mean_absolute_error'][-1]}")
else:
    model2 = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train_multi.shape[1], n_cols_multi)),
        LSTM(64, return_sequences=False),
        Dense(32),
        Dense(16),
        Dense(n_cols_multi)
    ])
    model2.compile(optimizer='adam', loss='mse', metrics=["mean_absolute_error"])
    model2.summary()

    print("\nModel2 eğitiliyor...")
    history_obj2 = model2.fit(x_train_multi, y_train_multi, epochs=100, batch_size=32)
    history2 = history_obj2.history
    print("\nModel2 Eğitimi Tamamlandı!")
    print(f"Son Epoch Kayıpları: {history2['loss'][-1]}")
    print(f"Son Epoch MAE: {history2['mean_absolute_error'][-1]}")

    model2.save(model_file2)
    with open(history_file2, "w") as f:
        json.dump(history2, f)

plt.figure(figsize=(12, 8))
plt.plot(history2["loss"], label="Mean Squared Error")
plt.plot(history2["mean_absolute_error"], label="Mean Absolute Error")
plt.legend()
plt.title("Model2 Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Test seti oluşturma
test_data_multi = scaled_data_multi[train_size_multi - time_steps:, :]
x_test_multi, y_test_multi = [], []
for i in range(time_steps, len(test_data_multi)):
    x_test_multi.append(test_data_multi[i-time_steps:i, :n_cols_multi])
    y_test_multi.append(test_data_multi[i, :n_cols_multi])
x_test_multi, y_test_multi = np.array(x_test_multi), np.array(y_test_multi)
x_test_multi = x_test_multi.reshape(x_test_multi.shape[0], x_test_multi.shape[1], n_cols_multi)

predictions_multi = model2.predict(x_test_multi)
y_test_multi_inv = scaler_y.inverse_transform(y_test_multi)
predictions_multi_inv = scaler_y.inverse_transform(predictions_multi)

RMSE_multi = np.sqrt(np.mean((y_test_multi_inv - predictions_multi_inv) ** 2)).round(2)
print("Multi RMSE:", RMSE_multi)

# Gelecek 30 gün tahmini
def insert_end(Xin, new_input):
    Xin[:, :-1, :] = Xin[:, 1:, :]
    Xin[:, -1, :] = new_input
    return Xin

future = 3650
forecast = []
Xin = x_test_multi[-1:, :, :]
time_forecast = [df.index[-1] + timedelta(days=i) for i in range(1, future+1)]

for i in range(future):
    out = model2.predict(Xin, batch_size=1)
    forecast.append(out[0])
    Xin = insert_end(Xin, out)

forecasted_output = np.array(forecast)
forecasted_output_inv = scaler_y.inverse_transform(forecasted_output)

df_result = pd.DataFrame({
    "t2m": forecasted_output_inv[:, 0],
    "latitude": forecasted_output_inv[:, 1],
    "longitude": forecasted_output_inv[:, 2],
    "sp": forecasted_output_inv[:, 3]
}, index=time_forecast)

plt.figure(figsize=(12, 6))
plt.plot(df.index, df["t2m"], label="Gerçek Veriler")
plt.plot(df_result.index, df_result["t2m"], label="Tahmin Edilen 30 Gün")
plt.xlabel("Tarih")
plt.ylabel("Sıcaklık (t2m)")
plt.title("Sonraki 1 yıl Tahmini - Çoklu Değişken Model")
plt.legend()

plt.show()

plt.show()


plt.show()

