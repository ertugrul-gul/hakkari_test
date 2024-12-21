# Gerekli Kütüphaneler
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# NetCDF dosya yolunu belirtin
file_path = "C:/Users/ertu_/Desktop/test/data_H/Data_0.nc"

# NetCDF dosyasını yükle
ds = xr.open_dataset(file_path)

# Verileri DataFrame'e dönüştür
df = ds[['t2m', 'u10', 'v10']].to_dataframe().reset_index()
df['valid_time'] = pd.to_datetime(df['valid_time'], errors='coerce')

# Geçersiz tarihleri ve satırları kaldır
valid_dates = df.dropna(subset=['valid_time', 'latitude', 'longitude']).sort_values('valid_time').reset_index(drop=True)

# Rüzgar hızını ve yönünü hesapla
valid_dates['wind_speed'] = np.sqrt(valid_dates['u10']**2 + valid_dates['v10']**2)
valid_dates['wind_dir'] = np.arctan2(valid_dates['v10'], valid_dates['u10']) * (180 / np.pi)
# 0-360 derece aralığına dönüştür
valid_dates['wind_dir'] = (valid_dates['wind_dir'] + 360) % 360

# Sıcaklığı Kelvin'den Celsius'a dönüştür
valid_dates['temperature'] = valid_dates['t2m'] - 273.15

# Gereksiz sütunları kaldır
valid_dates.drop(columns=['t2m', 'u10', 'v10'], inplace=True)

# 1950-2000 yılları arasındaki verilerle eğitim setini oluştur
train_data = valid_dates[(valid_dates['valid_time'] >= "1950-01-01") & (valid_dates['valid_time'] < "2000-01-01")]
X_train = train_data[['wind_speed', 'wind_dir']]
y_train = train_data['temperature']

# 2000-2024 yılları arasındaki verilerle test setini oluştur
test_data = valid_dates[(valid_dates['valid_time'] >= "2000-01-01") & (valid_dates['valid_time'] <= "2024-12-31")]
X_test = test_data[['wind_speed', 'wind_dir']]
y_test = test_data['temperature']

# Geleceğe yönelik tahmin aralığı oluştur (Şimdilik XGBoost'taki gibi kalsın, ARIMA için güncellenecek)
future_dates = pd.date_range(start="2025-01-01", end="2100-12-31", freq='Q')
X_future = pd.DataFrame({
    'wind_speed': np.random.choice(X_test['wind_speed'], size=len(future_dates)),
    'wind_dir': np.random.choice(X_test['wind_dir'], size=len(future_dates))
})

#----------------------------------------
# Random Forest Modeli
#----------------------------------------
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Test seti üzerinde tahminler
forecast_rf = model_rf.predict(X_test)
forecast_rf_future = model_rf.predict(X_future)

forecast_rf = pd.DataFrame({
    'valid_time': test_data['valid_time'].values,
    'predicted_temperature': forecast_rf
})
forecast_rf = forecast_rf.set_index('valid_time')

# Gelecekteki tahminleri ekle
future_forecast_rf = pd.DataFrame({
    'valid_time': future_dates,
    'predicted_temperature': forecast_rf_future
})
future_forecast_rf = future_forecast_rf.set_index('valid_time')

#----------------------------------------
# ARIMA Modeli
#----------------------------------------
# Eğitim verilerini mevsimlik olarak ayrıştır
y_train_ts = y_train.copy()
y_train_ts.index = pd.to_datetime(train_data['valid_time'], errors='coerce')
y_train_ts = y_train_ts.resample('M').mean() # Aylık ortalamaya indirge

# Durağanlık kontrolü (Augmented Dickey-Fuller Testi)
result = adfuller(y_train_ts)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# ACF ve PACF grafikleri
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(y_train_ts, lags=20, ax=axes[0])
plot_pacf(y_train_ts, lags=20, ax=axes[1])
plt.show()

# SARIMA modelini oluştur (p, d, q) ve (P, D, Q, s) değerlerini yukarıdaki tahminlere göre belirleyin
order = (1, 0, 1) # Daha basit bir parametre kombinasyonu
seasonal_order = (1, 0, 1, 12) # Daha basit bir parametre kombinasyonu

model_arima = SARIMAX(y_train_ts, order=order, seasonal_order=seasonal_order)
try:
    model_arima_fit = model_arima.fit(disp=False, method='lbfgs', maxiter=1000)  # Farklı bir optimizasyon algoritması deneyin ve iterasyonu artırın
except np.linalg.LinAlgError as e:
    print(f"Hata: {e}")
    print("Model optimizasyonu sırasında bir hata oluştu. Farklı parametreler veya optimizasyon ayarları deneyin.")
    # Bu noktada farklı bir model veya parametre kombinasyonu denemek mantıklı olabilir.
    # Örneğin: 
    # order = (2, 0, 1) # Deneme
    # model_arima = SARIMAX(y_train_ts, order=order, seasonal_order=seasonal_order)
    # model_arima_fit = model_arima.fit(disp=False, method='powell', maxiter=1000) 
    exit() # Hata durumunda kodu durdur

# Test verisi için tahmin aralığını oluştur
test_dates = pd.date_range(start="2000-01-01", end="2024-12-31", freq='M')

# Test seti için tahminler
forecast_arima = model_arima_fit.predict(start=len(y_train_ts), end=len(y_train_ts) + len(test_dates)-1)
forecast_arima.index = test_dates

# Gelecekteki tahminler için dinamik tahminler kullan (önceki tahminleri kullanarak)
forecast_arima_future = model_arima_fit.predict(start=len(y_train_ts) + len(test_dates), end=len(y_train_ts) + len(test_dates) + len(future_dates)-1, dynamic=True)
forecast_arima_future.index = future_dates

#----------------------------------------
# Modellerin Değerlendirilmesi ve Grafik
#----------------------------------------

# Gerçek değerlerin mevsimlik ortalamasını al
actual_seasonal = y_test.copy()
actual_seasonal.index = pd.to_datetime(test_data['valid_time'], errors='coerce')
actual_seasonal = actual_seasonal.resample('Q').mean()

# Tahminlerin mevsimlik ortalamasını al (Random Forest)
tahmin_seasonal_rf = pd.concat([forecast_rf, future_forecast_rf]).resample('Q').mean()

# Tahminlerin mevsimlik ortalamasını al (ARIMA)
tahmin_seasonal_arima = pd.concat([forecast_arima, forecast_arima_future]).resample('Q').mean()

# Model değerlendirmesi (Test seti üzerinde)
y_pred_rf = model_rf.predict(X_test)

# ARIMA için: actual_seasonal'ı reindex ile düzenle
# ARIMA tahminlerinin başlangıç ve bitiş tarihlerini al
arima_start_date = forecast_arima.index.min()
arima_end_date = forecast_arima.index.max()

# actual_seasonal'ı ARIMA tahminlerinin tarih aralığına göre filtrele
actual_seasonal = actual_seasonal[(actual_seasonal.index >= arima_start_date) & (actual_seasonal.index <= arima_end_date)]

# actual_seasonal'ı forecast_arima.index ile aynı indekse sahip olacak şekilde yeniden indeksle
actual_seasonal = actual_seasonal.reindex(forecast_arima.index)

# NaN değerlerini ffill ile doldur
actual_seasonal = actual_seasonal.fillna(method='ffill')

# Kalan NaN değerlerini kaldır
actual_seasonal = actual_seasonal.dropna()

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

mae_arima = mean_absolute_error(actual_seasonal, forecast_arima.loc[actual_seasonal.index])
rmse_arima = np.sqrt(mean_squared_error(actual_seasonal, forecast_arima.loc[actual_seasonal.index]))
r2_arima = r2_score(actual_seasonal, forecast_arima.loc[actual_seasonal.index])

print("--------------------------------------------------")
print("Model Değerlendirme Sonuçları:")
print("--------------------------------------------------")

print("\nRandom Forest:")
print(f"  MAE: {mae_rf:.2f}")
print(f"  RMSE: {rmse_rf:.2f}")
print(f"  R-kare: {r2_rf:.2f}")

print("\nARIMA:")
print(f"  MAE: {mae_arima:.2f}")
print(f"  RMSE: {rmse_arima:.2f}")
print(f"  R-kare: {r2_arima:.2f}")


# Model karşılaştırma
print("\n--------------------------------------------------")
print("Model Karşılaştırması ve Yorum:")
print("--------------------------------------------------")
if mae_rf < mae_arima:
    print("Random Forest, ARIMA'ya göre daha düşük MAE'ye sahip. Bu, Random Forest'in ortalama olarak daha doğru tahminler yaptığını gösterir.")
elif mae_arima < mae_rf:
      print("ARIMA, Random Forest'e göre daha düşük MAE'ye sahip. Bu, ARIMA'nın ortalama olarak daha doğru tahminler yaptığını gösterir.")
else:
    print("Random Forest ve ARIMA'nın MAE değerleri birbirine yakın.")
if rmse_rf < rmse_arima:
    print("Random Forest, ARIMA'ya göre daha düşük RMSE'ye sahip. Bu, Random Forest'in büyük hataları cezalandırmada daha iyi olduğunu gösterir.")
elif rmse_arima < rmse_rf:
    print("ARIMA, Random Forest'e göre daha düşük RMSE'ye sahip. Bu, ARIMA'nın büyük hataları cezalandırmada daha iyi olduğunu gösterir.")
else:
     print("Random Forest ve ARIMA'nın RMSE değerleri birbirine yakın.")
if r2_rf > r2_arima:
    print("Random Forest, ARIMA'ya göre daha yüksek R-kare değerine sahip. Bu, Random Forest'in varyansı açıklama konusunda daha iyi olduğunu gösterir.")
elif r2_arima > r2_rf:
     print("ARIMA, Random Forest'e göre daha yüksek R-kare değerine sahip. Bu, ARIMA'nın varyansı açıklama konusunda daha iyi olduğunu gösterir.")
else:
     print("Random Forest ve ARIMA'nın R-kare değerleri birbirine yakın.")


# Model seçim yorumu
print("\n--------------------------------------------------")
print("Model Seçim Yorumu:")
print("--------------------------------------------------")

if (mae_rf < mae_arima and rmse_rf < rmse_arima and r2_rf > r2_arima):
      print("Genel olarak, Random Forest, bu veriler için daha uygun bir model gibi görünmektedir. Metrikler açısından daha iyi sonuçlar vermiştir.")
elif (mae_arima < mae_rf and rmse_arima < rmse_rf and r2_arima > r2_rf):
    print("Genel olarak, ARIMA, bu veriler için daha uygun bir model gibi görünmektedir. Metrikler açısından daha iyi sonuçlar vermiştir.")
elif (mae_rf < mae_arima and rmse_rf < rmse_arima) or (mae_arima < mae_rf and rmse_arima < rmse_rf):
    print("MAE ve RMSE değerlerine bakıldığında, bu veriler için bir model diğerine göre hafifçe daha uygun görünmektedir.")
elif r2_rf > r2_arima:
      print("R-kare değerine göre, Random Forest bu verilerin varyansını açıklama konusunda daha iyi performans göstermektedir.")
elif r2_arima > r2_rf:
      print("R-kare değerine göre, ARIMA bu verilerin varyansını açıklama konusunda daha iyi performans göstermektedir.")
else:
      print("Random Forest ve ARIMA modelleri metrikler açısından benzer performans sergilemektedir. İhtiyaçlarınıza ve modelin yorumlanabilirliğine göre bir seçim yapabilirsiniz.")

print("\nModel seçimi yaparken verinizin özelliklerini, modellerin yorumlanabilirliğini ve analiz hedeflerinizi göz önünde bulundurmanız önemlidir. Bu sonuçlar, sadece bu veri seti ve bu model parametreleri için geçerlidir. Farklı parametreler ve veriler için farklı sonuçlar elde edebilirsiniz.")

# Grafik Çizimi
plt.figure(figsize=(15, 8), dpi=100)
plt.plot(actual_seasonal.index, actual_seasonal, label="Gerçek Mevsimlik Ortalama", color='blue', alpha=0.7)
plt.plot(tahmin_seasonal_rf.index, tahmin_seasonal_rf, label="Random Forest Mevsimlik Tahmini", color='green', alpha=0.7)
plt.plot(tahmin_seasonal_arima.index, tahmin_seasonal_arima, label="ARIMA Mevsimlik Tahmini", color='orange', alpha=0.7)

plt.title("2000-2100 Mevsimlik Ortalama Sıcaklık Tahmini")
plt.xlabel("Yıllar")
plt.ylabel("Sıcaklık (°C)")

# Tarih formatlama
plt.gca().xaxis.set_major_locator(mdates.YearLocator(10))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Gösterge
plt.legend()

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()