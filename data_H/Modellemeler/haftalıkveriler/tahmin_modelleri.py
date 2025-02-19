import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Makine Öğrenmesi Modelleri
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Derin Öğrenme Modelleri
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Geleneksel Zaman Serisi Modelleri
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# 📌 TensorFlow'u GPU yerine CPU modunda çalıştır
tf.config.set_visible_devices([], 'GPU')
print("✅ TensorFlow CPU modunda çalışacak.")

# 📌 Model Performansını Ölçme Fonksiyonu
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n🔹 {model_name} Performansı:")
    print(f"📌 MAE: {mae}")
    print(f"📌 RMSE: {rmse}")
    return mae, rmse

# 📌 Veri Setini Yükleme ve Hazırlama
df = pd.read_csv("hakkari_0.csv")
df["valid_time"] = pd.to_datetime(df["valid_time"])

# Tarih Bilgilerini Ayrıştırma
df["year"] = df["valid_time"].dt.year
df["month"] = df["valid_time"].dt.month
df["day"] = df["valid_time"].dt.day

# Giriş ve Çıkış Değişkenlerini Belirleme
X = df[["year", "month", "day", "lat", "lon", "sp", "u10", "v10"]]
y = df[["t2m", "tp"]]

# Veriyi Eğitim ve Test Setlerine Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Veriyi Ölçekleme
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_Y.fit_transform(y_train)
y_test_scaled = scaler_Y.transform(y_test)

# 📌 VAR Modeli
df_var = df[["valid_time", "t2m", "tp", "sp", "u10", "v10"]].copy()
df_var.set_index("valid_time", inplace=True)
df_var.sort_index(inplace=True)
df_var.index = pd.date_range(start=df_var.index.min(), periods=len(df_var), freq="D")  # 📌 Tarih frekansı düzeltildi

var_model_path = "var_model.pkl"
if os.path.exists(var_model_path):
    with open(var_model_path, "rb") as f:
        var_results = pickle.load(f)
    print("🔹 VAR modeli yüklendi.")
else:
    var_model = VAR(df_var)
    var_results = var_model.fit(maxlags=5)
    with open(var_model_path, "wb") as f:
        pickle.dump(var_results, f)
    print("✅ VAR modeli eğitildi ve kaydedildi!")

var_forecast = var_results.forecast(df_var.values[-5:], steps=len(y_test))
y_pred_var = var_forecast[:, :2]
evaluate_model(y_test, y_pred_var, "VAR")

# 📌 Prophet Modeli
df_prophet = df_var.reset_index()[["valid_time", "t2m"]]
df_prophet.columns = ["ds", "y"]
df_prophet["ds"] = df_prophet["ds"].dt.to_timestamp()  # 📌 Prophet için tarih formatı düzeltildi

prophet_model_path = "prophet_model.pkl"
if os.path.exists(prophet_model_path):
    with open(prophet_model_path, "rb") as f:
        prophet_model = pickle.load(f)
    print("🔹 Prophet modeli yüklendi.")
else:
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    prophet_model.fit(df_prophet)
    with open(prophet_model_path, "wb") as f:
        pickle.dump(prophet_model, f)
    print("✅ Prophet modeli eğitildi ve kaydedildi.")

future_dates = prophet_model.make_future_dataframe(periods=len(y_test))
forecast = prophet_model.predict(future_dates)
y_pred_prophet = forecast.iloc[-len(y_test):]["yhat"].values
evaluate_model(y_test["t2m"], y_pred_prophet, "Prophet")
