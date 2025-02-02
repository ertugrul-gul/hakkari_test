import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import os
import json

# 📌 Veriyi yükleme
data_dir = "data_H/hakkari.csv"
df = pd.read_csv(data_dir)

# 📌 Tarih formatına çevirme ve index olarak ayarlama
df['valid_time'] = pd.to_datetime(df['valid_time'])
df.set_index('valid_time', inplace=True)

# 📌 İlk grafikleri çizme
plt.figure(figsize=(15, 6))
df['t2m'].plot()
plt.title("Mean Temperature")
plt.show()

plt.figure(figsize=(15, 6))
df['tp'].plot()
plt.title("Mean Pressure")
plt.show()

# 📌 Veriyi hazırlama (t2m sütunu)
dataset = df["t2m"].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
print("Train Size:", train_size, "Test Size:", test_size)

train_data = scaled_data[:train_size, :]

# 📌 Eğitim seti oluşturma
x_train, y_train = [], []
time_steps = 60
n_cols = 1

for i in range(time_steps, len(train_data)):
    x_train.append(train_data[i-time_steps:i, :n_cols])
    y_train.append(train_data[i, :n_cols])

x_train, y_train = np.array(x_train), np.array(y_train)

# 📌 Şekil uyumu için reshape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], n_cols))

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# 📌 Model oluşturma veya yükleme
model_path = "trained_model.keras"
history_file = "training_history.json"

if os.path.exists(model_path) and os.path.exists(history_file):
    # 📌 Modeli yükle
    model = load_model(model_path)
    
    # 📌 Eğitim geçmişini yükle
    with open(history_file, "r") as f:
        history_data = json.load(f)

    class History:
        history = history_data

    history = History()
    
    print("\nDaha önce eğitilmiş model bulundu. Eğitimi tekrar başlatmıyorum.")
    print(f"Son Epoch Kayıpları: {history.history['loss'][-1]}")
    print(f"Son Epoch MAE: {history.history['mean_absolute_error'][-1]}")
else:
    # 📌 Modeli oluştur
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

    # 📌 Modeli eğit
    print("\nModel eğitiliyor...")
    history = model.fit(x_train, y_train, epochs=100, batch_size=32)

    # 📌 Eğitim geçmişini kaydet
    with open(history_file, "w") as f:
        json.dump(history.history, f)

    model.save(model_path)  # 📌 Modeli kaydet

    print("\nModel Eğitimi Tamamlandı!")
    print(f"Son Epoch Kayıpları: {history.history['loss'][-1]}")
    print(f"Son Epoch MAE: {history.history['mean_absolute_error'][-1]}")

# 📌 Eğitim kayıplarını çizme
plt.figure(figsize=(12, 8))
plt.plot(history.history["loss"], label="Mean Squared Error")
plt.plot(history.history["mean_absolute_error"], label="Mean Absolute Error")
plt.legend()
plt.title("Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# =============================================================================
# Test verisinin oluşturulması ve tahmin işlemleri
# =============================================================================

# Test verisi oluşturma: eğitim verisinin son 60 adımını da dahil ediyoruz.
test_data = scaled_data[train_size - time_steps:, :]

x_test = []
y_test = []
for i in range(time_steps, len(test_data)):
    x_test.append(test_data[i-time_steps:i, 0:n_cols])
    y_test.append(test_data[i, 0:n_cols])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], n_cols))

# Model ile tahmin yapma
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Tahminleri ölçekten çıkarma
y_test = scaler.inverse_transform(y_test)               # Gerçek değerleri ölçekten çıkarma

RMSE = np.sqrt(np.mean((y_test - predictions)**2)).round(2)
print("RMSE:", RMSE)

# =============================================================================
# Tahmin ve Gerçek Değerleri DataFrame'e aktarma ve valid_time ekleme
# =============================================================================

# DataFrame oluşturma
preds_acts = pd.DataFrame({
    'Predictions': predictions.flatten(),
    'Actuals': y_test.flatten()
})

# Test verilerine karşılık gelen valid_time değerlerini belirleme
# Eğitim verisinin %75'i alındıktan sonra kalan veriler test tarihlerini oluşturur.
# df.index içerisindeki sıralamayı kullanıyoruz.
test_dates = df.index[train_size:]  
# Eğer test_dates uzunluğu, preds_acts uzunluğundan farklıysa, zaman dizisini
# test_data içerisindeki indexlere göre de oluşturabilirsiniz. Ancak burada
# orijinal veri seti sıralaması korunduğu varsayılmıştır.

# DataFrame'e valid_time sütununu ekleyelim
preds_acts['valid_time'] = test_dates

# DataFrame'in ilk birkaç satırını kontrol edelim
print(preds_acts.head())

# =============================================================================
# Excel Dosyasına Kaydetme
# =============================================================================

# Excel dosyasına kaydetmek için (openpyxl kütüphanesi gereklidir)
preds_acts.to_excel("output.xlsx", index=False)

# =============================================================================
# Grafik Çizimi: x ekseninde valid_time kullanarak
# =============================================================================

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


train = dataset.iloc[:train_size , 0:1]
test = dataset.iloc[train_size: , 0:1]
test['Predictions'] = predictions

plt.figure(figsize= (16, 6))
plt.title('Temperature Prediction', fontsize= 18)
plt.xlabel('Date', fontsize= 18)
plt.ylabel('Temperature', fontsize= 18)
plt.savefig("output_plot3.png")
plt.plot(train['t2m'], linewidth= 3)
plt.plot(test['t2m'], linewidth= 3)
plt.plot(test["Predictions"], linewidth= 3)
plt.legend(['Train', 'Test', 'Predictions'])
