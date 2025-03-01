import numpy as np
import pandas as pd
from prophet import Prophet
import os
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 🔹 Model değerlendirme fonksiyonu
def evaluate_model(y_true, y_pred, model_name, coord):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n🔹 {model_name} Performansı - {coord}:")
    print(f"📌 MAE: {mae}")
    print(f"📌 RMSE: {rmse}")

# 🔹 Veriyi yükle
df = pd.read_csv("../base_data/hakkari_0.csv")

# 🔹 Tarih sütununu datetime formatına çevir ve indeks olarak ayarla
df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
df.set_index("valid_time", inplace=True)

# 🔹 Kullanılacak tüm koordinatları (lat, lon) al
coordinate_list = df.groupby(["lat", "lon"]).size().index.tolist()

# 🔹 Tahminleri saklamak için bir sözlük
prophet_predictions = {}

# 🔹 Her koordinat için ayrı ayrı model eğit
for coord in coordinate_list:
    print(f"📌 Prophet modeli çalıştırılıyor: {coord} koordinatı için...")

    # 🔹 Belirli koordinattaki veriyi al
    df_coord = df[(df["lat"] == coord[0]) & (df["lon"] == coord[1])]

    # 🔹 Eğer yeterli veri yoksa, geç
    if len(df_coord) < 30:
        print(f"⚠️ {coord} için yeterli veri yok, atlanıyor...")
        continue

    # 🔹 Prophet'in kullanacağı formatta veriyi hazırla
    df_prophet = df_coord.reset_index()[["valid_time", "t2m"]].copy()
    df_prophet.columns = ["ds", "y"]

    # 🔹 Eğitim ve test verisini ayır
    train_size = int(len(df_prophet) * 0.8)
    train, test = df_prophet.iloc[:train_size], df_prophet.iloc[train_size:]

    # 🔹 Eğer eğitim seti çok küçükse geç
    if len(train) < 30:
        print(f"⚠️ {coord} için eğitim verisi çok az, atlanıyor...")
        continue

    # 🔹 Prophet modelini tanımla ve eğit
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    prophet_model.fit(train)

    # 🔹 Gelecekteki tahminler için test seti uzunluğunda yeni tarihler oluştur
    future_dates = prophet_model.make_future_dataframe(periods=len(test))

    # 🔹 Tahmin yap
    forecast = prophet_model.predict(future_dates)
    y_pred_prophet = forecast.iloc[-len(test):]["yhat"].values

    # 🔹 Modeli değerlendir
    evaluate_model(test["y"], y_pred_prophet, "Prophet", coord)

    # 🔹 Sonuçları sakla
    prophet_predictions[coord] = pd.DataFrame(y_pred_prophet, index=test.index, columns=["t2m_pred"])

    print(f"✅ {coord} için model tamamlandı.")

# 🔹 Tüm tahminleri büyük bir DataFrame olarak birleştir
pred_df = pd.concat(prophet_predictions, names=["lat", "lon"])

# 🔹 Tahminleri CSV olarak kaydet
pred_df.to_csv("prophet_predictions.csv")

print("\n📌 Tüm koordinatlar için tahminler 'prophet_predictions.csv' dosyasına kaydedildi.")
