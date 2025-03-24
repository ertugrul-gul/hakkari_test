import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore


# 📌 CSV dosyasını oku
df = pd.read_csv("base_data/hakkari_0.csv")

# 📌 1️⃣ `valid_time` sütununu datetime formatına çevir
df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")

# 📌 2️⃣ `(lat, lon)` bilgisini kaldırarak günlük ortalamaları hesapla
df_daily_mean = df.groupby("valid_time").mean(numeric_only=True)  # Tüm sayısal sütunları ortalamaya alır

# 📌 3️⃣ Yeni CSV olarak kaydet
df_daily_mean.to_csv("daily_avg_fixed.csv")
numeric_cols = ["lat", "lon", "sp", "u10", "v10", "t2m", "tp", "ws"]

# Z-score hesapla ve aykırı değerleri belirle
z_scores = df[numeric_cols].apply(zscore)  # Her sütun için Z-score hesaplar
threshold = 3  # Aykırılık eşiği (genellikle 3 kullanılır)
outliers_mask = (np.abs(z_scores) > threshold).any(axis=1)  # En az bir sütunda aykırı olanları bul

# Aykırı değerleri temizlenmiş yeni DataFrame
df_cleaned = df[~outliers_mask]  # Aykırı satırları çıkar

df_cleaned.to_csv = ("1.csv")


"""
# 📌 4️⃣ Ortalama veriyi kontrol et
print(df_daily_mean.head())  # İlk 5 satırı göster

# 📌 5️⃣ Günlük ortalama sıcaklık verisini çiz
plt.figure(figsize=(10, 5))
plt.plot(df_daily_mean.index, df_daily_mean["t2m"], label="Günlük Ortalama Sıcaklık (°C)", color='r')
plt.xlabel("Tarih")
plt.ylabel("t2m (°C)")
plt.title("Günlük Ortalama Değerler (Tüm Konumlar)")
plt.legend()
plt.grid()
plt.show()
"""