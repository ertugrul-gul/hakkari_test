# %%
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, TimeDistributed, Bidirectional
from tensorflow.keras.models import model_from_json

# %%
df = pd.read_csv("/home/ertugrulgul/PycharmProjects/hakkari_test/data_H/Modellemeler/haftalıkveriler/hakkari.csv")

# %%
df['t2m'] = df['t2m'] - 273.15
df['tp'] = df['tp']*1000
df['sp'] = df['sp']/100
# %%
df.head()

# %%
df.info()

# %%
df.describe()

# %%
df.isna().sum()

# %%
print(df.columns)

# %%
# Tarih indeksini datetime formatına çevir
df['valid_time'] = pd.to_datetime(df['valid_time'])  # Tarih formatına çevir
df.set_index('valid_time', inplace=True)  # İndeks olarak ayarla
df = df.sort_index()  # Zaman sırasına göre sıralayın (bazı algoritmalar için önemli)

# Enlem ve boylam bazında gruplama yaparak her bir bölge için veri oluştur
grouped = df.groupby(["lat", "lon"])

# %%
print(df.columns)

# %%
# Her grup için "valid_time" sütununun korunmasını sağlıyoruz
for (lat, lon), group in grouped:
    print(f"\n📌 İşlenen Lokasyon: ({lat}, {lon})")

    # Lokasyon bazlı DataFrame oluştur
    location_df = group.copy()

    # Tarih indeksini geri getir
    location_df = location_df.reset_index().set_index("valid_time").sort_index()

    print(location_df.head())  # İlk 5 satırı göster

# %%
for (lat, lon), group in grouped:
    print(f"\n📌 Lokasyon: ({lat}, {lon}) için veri işleniyor...")
    print(group.columns)  # Sütunları göster
    break  # Sadece ilk grup için kontrol et



# %%
df.dtypes

# %%
print(type(df.index))  # İndeksin türünü kontrol et
print(df.index.dtype)  # İndeksin veri tipini kontrol et

# %%
df_daily = df.resample('D').mean()
df_weekly = df.resample('W').mean()
df_month = df.resample('ME').mean()
df_year = df.resample ('YE').mean()

# %%
# Tarih aralıklarını kontrol et
print(df.index.min(), df.index.max())  # Veri setinin başlangıç ve bitiş tarihleri
print(df.index.to_series().diff().value_counts())  # Zaman aralıklarının düzenliliğini kontrol et

# %%
#IQR Yöntemiyle Aykırı Değerleri CSV'ye Kaydetme

# Sayısal sütunları seç
numeric_cols = df.select_dtypes(include=["number"]).columns  

# Aykırı değerleri saklamak için boş bir DataFrame oluştur
outliers_df = pd.DataFrame()

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)  
    Q3 = df[col].quantile(0.75)  
    IQR = Q3 - Q1  

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Aykırı değerleri belirle
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].copy()  

    if not outliers.empty:
        print(f"\n📌 {col} sütunundaki aykırı değerler:")
        print(f"🔹 Alt sınır: {lower_bound:.2f}, Üst sınır: {upper_bound:.2f}")
        print(outliers[[col]])  # Sadece ilgili sütunu göster

        # ✅ Hangi sütundan geldiğini belirtmek için yeni bir sütun ekleyelim
        outliers.loc[:, "Aykırı_Sütun"] = col  

        # ✅ Aykırı değerleri birleştir
        outliers_df = pd.concat([outliers_df, outliers])  

# ✅ CSV olarak kaydet (Eğer aykırı değer bulunduysa)
if not outliers_df.empty:
    outliers_df.to_csv("aykiri_veriler_IQR.csv", index=True, encoding="utf-8")
    print("✅ Aykırı değerler 'aykiri_veriler.csv' dosyasına kaydedildi!")
else:
    print("⚠️ Aykırı değer bulunamadı.")

# %%
#Z-Score Yöntemiyle Aykırı Değerleri CSV'ye Kaydetme

# Sayısal sütunları seç
numeric_cols = df.select_dtypes(include=["number"]).columns  

outliers_df = pd.DataFrame()  # Aykırı değerleri saklayacak DataFrame

threshold = 3  # Aykırılık eşiği

for col in numeric_cols:
    mean = df[col].mean()
    std_dev = df[col].std()
    z_scores = stats.zscore(df[col])

    # Aykırı değerleri belirle
    outliers = df[abs(z_scores) > threshold].copy()  

    if not outliers.empty:
        print(f"\n📌 {col} sütunundaki aykırı değerler:")
        print(f"🔹 Ortalama: {mean:.2f}, Standart Sapma: {std_dev:.2f}")
        print(f"🔹 Aykırı sınır: {mean - 3*std_dev:.2f} ile {mean + 3*std_dev:.2f}")
        print(outliers[[col]])  # Sadece ilgili sütunu göster  

        # ✅ Hangi sütundan geldiğini ekleyelim
        outliers.loc[:, "Aykırı_Sütun"] = col  

        # ✅ Aykırı değerleri ana DataFrame'e ekleyelim
        outliers_df = pd.concat([outliers_df, outliers])  

# ✅ CSV olarak kaydet
if not outliers_df.empty:
    outliers_df.to_csv("aykiri_veriler_zscore.csv", index=True, encoding="utf-8")
    print("✅ Aykırı değerler 'aykiri_veriler_zscore.csv' dosyasına kaydedildi!")
else:
    print("⚠️ Aykırı değer bulunamadı.")

# %%
import pandas as pd

# Başlangıçtaki satır sayısını kaydet
initial_rows = df.shape[0]  

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]  # Aykırı değerleri kaldır

# Silinen satır sayısını hesapla
deleted_rows = initial_rows - df.shape[0]

# Çıktı oluştur
if deleted_rows > 0:
    print(f"✅ {deleted_rows} aykırı değer başarıyla silindi!")
else:
    print("⚠️ Silinecek aykırı değer bulunamadı.")

# %%
import pandas as pd
from scipy import stats

# Başlangıçtaki satır sayısını kaydet
initial_rows = df.shape[0]

threshold = 3  # Aykırılık sınırı

# Tüm sütunlarda aykırı olan satırları saklamak için boş bir liste
outlier_indices = set()

# 1️⃣ HER SÜTUN İÇİN AYKIRI SATIRLARI TESPİT ET
for col in numeric_cols:
    z_scores = stats.zscore(df[col])
    outliers = df.index[abs(z_scores) > threshold]  # Aykırı satırların index'leri
    outlier_indices.update(outliers)  # Set içine ekleyerek tekrarları önlüyoruz

# 2️⃣ SADECE TEK SEFERDE AYKIRI SATIRLARI SİL
df = df.drop(index=outlier_indices)

# Silinen satır sayısını hesapla
deleted_rows = initial_rows - df.shape[0]

# Çıktı oluştur
if deleted_rows > 0:
    print(f"✅ {deleted_rows} aykırı değer başarıyla silindi!")
else:
    print("⚠️ Silinecek aykırı değer bulunamadı.")

# %%
df.head()

# %%
df.info()

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(df.index, df["t2m"], label="Sıcaklık")
plt.title("Zaman Serisi Grafiği")
plt.xlabel("Tarih")
plt.ylabel("Sıcaklık")
plt.legend()
plt.show()

# %%
df["t2m_MA"] = df["t2m"].rolling(window=30).mean()  # 30 günlük ortalama
df[["t2m", "t2m_MA"]].plot(figsize=(12,5))
plt.title("30 Günlük Hareketli Ortalama ile Sıcaklık")
plt.show()

# %%
from statsmodels.tsa.stattools import adfuller

result = adfuller(df["t2m"].dropna())  # NaN'leri çıkar
print(f"ADF Test p-değeri: {result[1]}")
if result[1] < 0.05:
    print("✅ Zaman serisi durağan (stationary).")
else:
    print("⚠️ Zaman serisi durağan değil, fark alma işlemi gerekebilir.")

# %%
plt.figure(figsize=(12,5))
plt.plot(df.index, df["t2m"], label="Günlük Sıcaklık")
plt.title("Sıcaklık Zaman Serisi Grafiği")
plt.xlabel("Tarih")
plt.ylabel("Sıcaklık (°C)")
plt.legend()
plt.show()


# %%
df["t2m_ma"] = df["t2m"].rolling(window=365).mean()  # Yıllık hareketli ortalama
df[["t2m", "t2m_ma"]].plot(figsize=(12,5))
plt.title("Yıllık Hareketli Ortalama ile Mevsimsellik")
plt.show()

'''
# %%
from statsmodels.tsa.seasonal import seasonal_decompose

decompose_result = seasonal_decompose(df["t2m"], model="additive", period=365)  # Günlük veriler için 1 yıl = 365 gün

plt.figure(figsize=(12, 8))
decompose_result.plot()
plt.show()

# %%
df = df.asfreq("D")  # Günlük frekans belirle

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

for (lat, lon), group in grouped:
    print(f"\n📌 SARIMA Modeli Eğitiliyor: ({lat}, {lon})")

    # Lokasyon bazlı DataFrame
    location_df = group.copy().reset_index()  # "valid_time" sütununu geri getir
    location_df = location_df.set_index("valid_time").sort_index()
    
    # Günlük frekans ayarla
    location_df = location_df.asfreq("D")

    # Eksik günleri doldur
    location_df.ffill(inplace=True)

    # SARIMA Modeli (Trend + Mevsimsellik İçin)
    model = SARIMAX(location_df["t2m"], order=(3,1,3), seasonal_order=(1,1,1,12))
    model_fit = model.fit()

    # 30 Günlük Tahmin Yap
    forecast = model_fit.forecast(steps=30)

    # Tahminleri Görselleştir
    plt.figure(figsize=(10,4))
    plt.plot(location_df.index, location_df["t2m"], label="Gerçek Sıcaklık")
    plt.plot(pd.date_range(location_df.index[-1], periods=30, freq="D"), forecast, label="Tahmin", color="red")
    plt.title(f"SARIMA Tahmini - ({lat}, {lon})")
    plt.xlabel("Tarih")
    plt.ylabel("Sıcaklık")
    plt.legend()
    plt.show()
'''