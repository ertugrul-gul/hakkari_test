import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 🔹 Model değerlendirme fonksiyonu
def evaluate_model(y_true, y_pred, model_name, coord):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n🔹 {model_name} Performansı - {coord}:")
    print(f"📌 MAE: {mae}")
    print(f"📌 RMSE: {rmse}")

# 🔹 Veriyi yükleme
df = pd.read_csv("../base_data/hakkari_0.csv")

# 🔹 Tarih sütununu datetime formatına çevir ve indeks olarak ayarla
df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
df.set_index("valid_time", inplace=True)

# 🔹 Zaman indeksini sıralama
df.sort_index(inplace=True)

# 🔹 Kullanılacak tüm koordinatları (lat, lon) al
coordinate_list = df.groupby(["lat", "lon"]).size().index.tolist()

# 🔹 Tahminleri saklamak için bir sözlük
var_predictions = {}

# 🔹 Her koordinat için ayrı ayrı model eğit
for coord in coordinate_list:
    print(f"📌 VAR modeli çalıştırılıyor: {coord} koordinatı için...")

    # 🔹 Belirli koordinattaki veriyi al.  Artık df'den filtreleme yapmıyoruz, zaten gruplanmış veriyle uğraşıyoruz.
    lat, lon = coord  # Koordinatları ayır
    df_coord = df[(df["lat"] == lat) & (df["lon"] == lon)].copy()  # Copy'i ekledim

    # 🔹 Eğer yeterli veri yoksa, geç
    if len(df_coord) < 30:
        print(f"⚠️ {coord} için yeterli veri yok, atlanıyor...")
        continue

    # 🔹 Günlük frekansı ayarla ve eksik verileri doldur
    df_coord = df_coord.asfreq("D")
    df_coord = df_coord.interpolate()

    # 🔹 Model için kullanılacak değişkenleri seç
    df_var = df_coord[["t2m", "tp", "sp", "u10", "v10"]]

    # 🔹 Veriyi eğitim ve test olarak ayır
    train_size = int(len(df_var) * 0.8)
    train, test = df_var.iloc[:train_size], df_var.iloc[train_size:]

    # 🔹 Eğer eğitim seti çok küçükse geç
    if len(train) < 30:
        print(f"⚠️ {coord} için eğitim verisi çok az, atlanıyor...")
        continue

    # 🔹 VAR modelini eğit
    var_model = VAR(train)
    var_results = var_model.fit(maxlags=5)

    # 🔹 Gelecekteki tahminler için test setinin son 5 gözlemini kullanarak tahmin yap
    var_forecast = var_results.forecast(train.values[-5:], steps=len(test))

    # 🔹 Sadece "t2m" ve "tp" tahminlerini al
    y_pred_var = var_forecast[:, :2]

    # 🔹 Modeli değerlendir
    evaluate_model(test[["t2m", "tp"]].values, y_pred_var, "VAR", coord) # test verisini numpy array'ine dönüştürdüm

    # 🔹 Sonuçları DataFrame olarak sakla
    var_predictions[coord] = pd.DataFrame(y_pred_var, index=test.index, columns=["t2m_pred", "tp_pred"])

    # 🔹 Grafik çizimi
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train["t2m"], label="Eğitim Verisi (t2m)")
    plt.plot(test.index, test["t2m"], label="Gerçek Değerler (t2m)")
    plt.plot(test.index, y_pred_var[:, 0], label="Tahminler (t2m)", linestyle="--")
    plt.title(f"{coord} - T2M Tahminleri")
    plt.xlabel("Tarih")
    plt.ylabel("Sıcaklık (t2m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    print(f"✅ {coord} için model tamamlandı.")

# 🔹 Tüm tahminleri büyük bir DataFrame olarak birleştir
pred_df = pd.concat(var_predictions, names=["lat", "lon"])

# 🔹 Tahminleri CSV olarak kaydet
pred_df.to_csv("var_predictions.csv")

print("\n📌 Tüm koordinatlar için tahminler 'var_predictions.csv' dosyasına kaydedildi.")