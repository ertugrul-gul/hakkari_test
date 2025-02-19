import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 🔹 Model değerlendirme fonksiyonu
def evaluate_model(y_true, y_pred, model_name, coord):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n🔹 {model_name} Performansı - {coord}:")
    print(f"📌 MAE: {mae}")
    print(f"📌 RMSE: {rmse}")

# 🔹 Veriyi yükle
df = pd.read_csv("hakkari_0.csv")

# 🔹 Tarih sütununu datetime formatına çevir ve indeks olarak ata
df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
df.set_index("valid_time", inplace=True)

# 🔹 Eksik verileri doldurma
df = df.interpolate()

# 🔹 Veriyi pivot formatına getir
df_pivot = df.pivot_table(index="valid_time", columns=["lat", "lon"], values=["t2m", "tp"])

# 🔹 Mevcut tüm (latitude, longitude) çiftlerini al
coordinate_list = list(df_pivot["t2m"].columns)  # Sadece t2m için koordinatları al

# 🔹 Tahminleri saklamak için bir sözlük
sarima_predictions = {}

# 🔹 Her koordinat için SARIMA modeli uygula
for coord in coordinate_list:
    print(f"📌 SARIMA modeli çalıştırılıyor: {coord} koordinatı için...")

    # Koordinatı tuple olarak al
    coord = tuple(map(float, coord))

    # 🔹 MultiIndex yapısına uygun sütunu seç
    try:
        sarima_series = df_pivot.xs(key=("t2m", coord[0], coord[1]), axis=1)
    except KeyError:
        print(f"⚠️ Uyarı: {coord} için veri bulunamadı, atlanıyor...")
        continue

    # 🔹 Eksik verileri doldur ve günlük frekansı ayarla
    sarima_series = sarima_series.interpolate().asfreq("D")

    # 🔹 Eğer veri çok azsa, atla
    if sarima_series.dropna().shape[0] < 30:
        print(f"⚠️ {coord} için yeterli veri yok, atlanıyor...")
        continue

    # 🔹 Veriyi eğitim ve test olarak ayır
    train_size = int(len(sarima_series) * 0.8)
    train, test = sarima_series.iloc[:train_size], sarima_series.iloc[train_size:]

    # 🔹 Eğer eğitim verisi çok azsa, atla
    if len(train) < 30:
        print(f"⚠️ {coord} için eğitim verisi çok az, atlanıyor...")
        continue

    # 🔹 Test setindeki NaN'ları temizle
    test = test.dropna()

    # 🔹 SARIMA modelini tanımla ve eğit
    try:
        sarima_model = SARIMAX(
            train, order=(1,1,1), seasonal_order=(1,1,1,12),
            enforce_stationarity=False, enforce_invertibility=False
        )
        sarima_results = sarima_model.fit(disp=False)

        # 🔹 Test seti uzunluğu kadar tahmin yap
        y_pred_sarima = sarima_results.forecast(steps=len(test))

        # 🔹 Eğer tahminde NaN varsa, doldur
        y_pred_sarima = y_pred_sarima.fillna(method="bfill").fillna(method="ffill")

    except Exception as e:
        print(f"⚠️ {coord} için SARIMA modeli başarısız oldu: {e}")
        continue

    # 🔹 Model performansını değerlendir
    evaluate_model(test, y_pred_sarima, "SARIMA", coord)

    # 🔹 Sonuçları sakla
    sarima_predictions[coord] = y_pred_sarima

    print(f"✅ {coord} için model tamamlandı.")

# 🔹 Tahminleri DataFrame olarak kaydet
pred_df = pd.DataFrame(sarima_predictions, index=test.index)
pred_df.to_csv("sarima_predictions.csv")

print("\n📌 Tahminler 'sarima_predictions.csv' dosyasına kaydedildi.")
