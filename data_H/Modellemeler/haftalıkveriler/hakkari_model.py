import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import os
import pickle  # Prophet modelini kaydetmek için kullanacağız



df = pd.read_csv("hakkari_0.csv")
# valid_time'ı datetime formatına çevir
df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
df.info()

# 3. Eksik veri kontrolü
print("\n🔹 Eksik Veri Kontrolü:")
print(df.isnull().sum())

# 4. Farklı koordinat sayısını kontrol et
print("\n🔹 Toplam Farklı Koordinat Sayısı:")
print(df.groupby(["lat", "lon"]).size().reset_index().shape[0])

# 5. Farklı tarihler ve eksik veri olup olmadığını incele
print("\n🔹 Toplam Farklı Tarih Sayısı:")
print(df["valid_time"].nunique())

# 6. Her koordinat için tüm tarihlerde eksiksiz veri olup olmadığını kontrol et
print("\n🔹 Koordinat-Tarih Bazında Eksik Veri Sayısı:")
missing_values = df.groupby(["lat", "lon", "valid_time"]).size().unstack(fill_value=0)
print(missing_values.apply(lambda x: x.isnull().sum(), axis=1))

# 7. İlk birkaç satırı görüntüle
print("\n🔹 İlk 5 Satır:")
print(df.head())



# 1. Tarih bilgilerini yıl, ay, gün olarak ayıralım
df["year"] = df["valid_time"].dt.year
df["month"] = df["valid_time"].dt.month
df["day"] = df["valid_time"].dt.day

# 2. Giriş değişkenlerini belirleyelim (Bağımsız değişkenler X)
X = df[["year", "month", "day", "lat", "lon", "sp", "u10", "v10"]]

# 3. Çıkış değişkenlerini belirleyelim (Bağımlı değişkenler Y)
y = df[["t2m", "tp"]]

# 4. Veriyi eğitim ve test setlerine ayıralım (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 5. Veriyi ölçekleyelim (StandardScaler ile normalleştirme yapalım)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_Y.fit_transform(y_train)
y_test_scaled = scaler_Y.transform(y_test)

print("🔹 Veri başarıyla işlendi ve ölçeklendi!")
print(f"X_train boyutu: {X_train.shape}, y_train boyutu: {y_train.shape}")
print(f"X_test boyutu: {X_test.shape}, y_test boyutu: {y_test.shape}")




# 1️⃣ XGBoost Modeli Eğitme
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# 2️⃣ Random Forest Modeli Eğitme
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 3️⃣ Tahminleri Alalım
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

# 4️⃣ Model Performanslarını Ölçelim
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n🔹 {model_name} Performansı:")
    print(f"📌 MAE: {mae}")
    print(f"📌 RMSE: {rmse}")

evaluate_model(y_test, y_pred_xgb, "XGBoost")
evaluate_model(y_test, y_pred_rf, "Random Forest")



# 1️⃣ LSTM İçin Veriyi 3D Formata Getir
X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# 2️⃣ LSTM Modelini Oluştur
lstm_model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
    Dropout(0.2),
    LSTM(25, activation='relu'),
    Dropout(0.2),
    Dense(2)  # 2 Çıkış: t2m (sıcaklık) ve tp (yağış)
])

# 3️⃣ Modeli Derle
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 4️⃣ Modeli Eğit
lstm_model.fit(X_train_lstm, y_train_scaled, epochs=20, batch_size=32, validation_data=(X_test_lstm, y_test_scaled), verbose=1)

# 5️⃣ Tahminleri Al
y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)

# 6️⃣ Ölçeklendirilmiş Tahminleri Geri Dönüştür
y_pred_lstm = scaler_Y.inverse_transform(y_pred_lstm_scaled)

# 7️⃣ Model Performansını Değerlendir
evaluate_model(y_test, y_pred_lstm, "LSTM")

# 1️⃣ MLP Modelini Oluştur
mlp_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(2)  # 2 Çıkış: t2m (sıcaklık) ve tp (yağış)
])

# 2️⃣ Modeli Derle
mlp_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 3️⃣ Modeli Eğit
mlp_model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test_scaled), verbose=1)

# 4️⃣ Tahminleri Al
y_pred_mlp_scaled = mlp_model.predict(X_test_scaled)

# 5️⃣ Ölçeklendirilmiş Tahminleri Geri Dönüştür
y_pred_mlp = scaler_Y.inverse_transform(y_pred_mlp_scaled)

# 6️⃣ Model Performansını Değerlendir
evaluate_model(y_test, y_pred_mlp, "MLP")



# 1️⃣ VAR İçin Veriyi Hazırlama (Tarih bazında sıralama)
df_var = df[["valid_time", "t2m", "tp", "sp", "u10", "v10"]].copy()
df_var.set_index("valid_time", inplace=True)
df_var.sort_index(inplace=True)

# 2️⃣ VAR Modeli Eğitme
var_model = VAR(df_var)
var_results = var_model.fit(maxlags=5)

# 3️⃣ Tahmin Yapma (Test seti kadar ileriye tahmin yap)
var_forecast = var_results.forecast(df_var.values[-5:], steps=len(y_test))

# 4️⃣ Gerçek ve Tahmin Edilen Değerleri Karşılaştırma
y_pred_var = var_forecast[:, :2]  # İlk iki sütun t2m ve tp

# 5️⃣ Model Performansını Değerlendir
evaluate_model(y_test, y_pred_var, "VAR")



# Tarih sütununu datetime formatına çevirip index olarak ayarla
df_var.index = pd.DatetimeIndex(df_var.index).to_period('D')  # Günlük veri olduğunu belirtiyoruz




# 1️⃣ SARIMA Modeli Eğitme (Sadece sıcaklık tahmini yapacak)
sarima_model = SARIMAX(df_var["t2m"], order=(1,1,1), seasonal_order=(1,1,1,12), freq='D')
sarima_results = sarima_model.fit()

# 2️⃣ Tahmin Yapma
y_pred_sarima = sarima_results.forecast(steps=len(y_test))

# 3️⃣ Model Performansını Değerlendir
evaluate_model(y_test["t2m"], y_pred_sarima, "SARIMA")



# 1️⃣ Prophet Modeli İçin Veriyi Hazırlayalım
df_prophet = df_var.reset_index()[["valid_time", "t2m"]]  # Prophet için sadece tarih ve sıcaklık
df_prophet.columns = ["ds", "y"]  # Prophet formatı: Tarih (ds), Değer (y)

# 2️⃣ Prophet Modelini Önceden Kaydedilmiş Dosya Var mı Kontrol Et
prophet_model_path = "prophet_model.pkl"

if os.path.exists(prophet_model_path):
    # Kaydedilmiş modeli yükle
    with open(prophet_model_path, "rb") as f:
        prophet_model = pickle.load(f)
    print("🔹 Prophet modeli önceden eğitilmiş, kaydedilmiş model yükleniyor...")
else:
    # Yeni Prophet modeli oluştur
    prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)

    # Modeli eğit
    prophet_model.fit(df_prophet)

    # Modeli kaydet
    with open(prophet_model_path, "wb") as f:
        pickle.dump(prophet_model, f)

    print("✅ Prophet modeli eğitildi ve kaydedildi!")

# 3️⃣ Tahmin Yapma
future_dates = prophet_model.make_future_dataframe(periods=len(y_test))
forecast = prophet_model.predict(future_dates)

# 4️⃣ Test setine denk gelen tahminleri alalım
y_pred_prophet = forecast.iloc[-len(y_test):]["yhat"].values

# 5️⃣ Model Performansını Değerlendir
evaluate_model(y_test["t2m"], y_pred_prophet, "Prophet")
from prophet import Prophet
import os
import pickle  # Prophet modelini kaydetmek için kullanacağız

# 1️⃣ Prophet Modeli İçin Veriyi Hazırlayalım
df_prophet = df_var.reset_index()[["valid_time", "t2m"]]  # Prophet için sadece tarih ve sıcaklık
df_prophet.columns = ["ds", "y"]  # Prophet formatı: Tarih (ds), Değer (y)

# 2️⃣ Prophet Modelini Önceden Kaydedilmiş Dosya Var mı Kontrol Et
prophet_model_path = "prophet_model.pkl"

if os.path.exists(prophet_model_path):
    # Kaydedilmiş modeli yükle
    with open(prophet_model_path, "rb") as f:
        prophet_model = pickle.load(f)
    print("🔹 Prophet modeli önceden eğitilmiş, kaydedilmiş model yükleniyor...")
else:
    # Yeni Prophet modeli oluştur
    prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)

    # Modeli eğit
    prophet_model.fit(df_prophet)

    # Modeli kaydet
    with open(prophet_model_path, "wb") as f:
        pickle.dump(prophet_model, f)

    print("✅ Prophet modeli eğitildi ve kaydedildi!")

# 3️⃣ Tahmin Yapma
future_dates = prophet_model.make_future_dataframe(periods=len(y_test))
forecast = prophet_model.predict(future_dates)

# 4️⃣ Test setine denk gelen tahminleri alalım
y_pred_prophet = forecast.iloc[-len(y_test):]["yhat"].values

# 5️⃣ Model Performansını Değerlendir
evaluate_model(y_test["t2m"], y_pred_prophet, "Prophet")

