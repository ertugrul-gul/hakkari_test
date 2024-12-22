import xarray as xr
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

# Veri yükleme
file_path = "data_H/final_merged_data.nc"
data = xr.open_dataset(file_path)

# Hedef ve giriş değişkenleri
temperature = data['t2m']
features = data[['sp', 'u10', 'v10', 'tp', 'z', 't', 'u', 'v', 'q', 'r']]

# Zaman aralıklarını belirleme
train_time = slice("1940-01-01", "2000-12-31")
test_time = slice("2001-01-01", "2023-12-31")

# Eğitim ve test verilerini seçme
train_features = features.sel(valid_time=train_time)
test_features = features.sel(valid_time=test_time)

train_temperature = temperature.sel(valid_time=train_time)
test_temperature = temperature.sel(valid_time=test_time)

# Pandas DataFrame'e dönüştürme
df_train_features = train_features.to_dataframe().reset_index()
df_test_features = test_features.to_dataframe().reset_index()

df_train_temperature = train_temperature.to_dataframe().reset_index()
df_test_temperature = test_temperature.to_dataframe().reset_index()

# Mevsim bilgisi ekleme
def add_season_column(df):
    def assign_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'

    df['month'] = pd.to_datetime(df['valid_time']).dt.month
    df['season'] = df['month'].apply(assign_season)
    return df

df_train_features = add_season_column(df_train_features)
df_test_features = add_season_column(df_test_features)
df_train_temperature = add_season_column(df_train_temperature)
df_test_temperature = add_season_column(df_test_temperature)

# Sayısal sütunları seçme
numeric_columns = df_train_features.select_dtypes(include=["float64", "int64"]).columns

# Eksik değerleri doldurma (Tüm sayısal sütunlara uygulanan imputer)
imputer = IterativeImputer(random_state=42, max_iter=20, tol=1e-4)
#imputer = SimpleImputer(strategy="mean")

# Eğitim verisi eksiklerini doldurma
df_train_features[numeric_columns] = imputer.fit_transform(df_train_features[numeric_columns])

# Test verisi eksiklerini doldurma
df_test_features[numeric_columns] = imputer.transform(df_test_features[numeric_columns])

# Hedef değişkende eksik değerleri doldurma
df_train_temperature['t2m'] = df_train_temperature['t2m'].fillna(df_train_temperature['t2m'].mean())
df_test_temperature['t2m'] = df_test_temperature['t2m'].fillna(df_test_temperature['t2m'].mean())

# Model eğitimi ve doğrulama
X_train = df_train_features.drop(columns=['valid_time', 'latitude', 'longitude', 'number', 'expver', 'season', 'month'])
y_train = df_train_temperature['t2m']

X_test = df_test_features.drop(columns=['valid_time', 'latitude', 'longitude', 'number', 'expver', 'season', 'month'])
y_test = df_test_temperature['t2m']

# Sabit değerli sütunları kontrol etme ve kaldırma
constant_columns = X_train.columns[X_train.nunique() <= 1]
X_train = X_train.drop(columns=constant_columns)
X_test = X_test.drop(columns=constant_columns)

# Model oluşturma (GridSearchCV ile hiperparametre optimizasyonu)
param_grid = {
    'max_iter': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Model seçimi (GradientBoostingRegressor veya HistGradientBoostingRegressor)
use_hist_gradient_boosting = True #  HistGradientBoosting kullanmayı zorunlu hale getirdik
if use_hist_gradient_boosting:
    grid_search = GridSearchCV(HistGradientBoostingRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
else:
    grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
    
alm_model = grid_search.fit(X_train, y_train).best_estimator_

# Modeli eğit
alm_model.fit(X_train, y_train)

# Tahmin ve performans değerlendirme
y_pred_train = alm_model.predict(X_train)
y_pred_test = alm_model.predict(X_test)

print("Eğitim Hatası (MSE):", mean_squared_error(y_train, y_pred_train))

# Eğitim sonuçlarını görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(y_train.values[:100], label='Gerçek Değerler (Eğitim)')
plt.plot(y_pred_train[:100], label='Tahminler (Eğitim)')
plt.title('Eğitim Seti - Tahmin vs Gerçek')
plt.xlabel('Örnek')
plt.ylabel('Sıcaklık')
plt.legend()
plt.show()

print("Test Hatası (MSE):", mean_squared_error(y_test, y_pred_test))

# Test sonuçlarını görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:100], label='Gerçek Değerler (Test)')
plt.plot(y_pred_test[:100], label='Tahminler (Test)')
plt.title('Test Seti - Tahmin vs Gerçek')
plt.xlabel('Örnek')
plt.ylabel('Sıcaklık')
plt.legend()
plt.show()

# 2100 yılı için tahmin (günlük)
future_dates = pd.date_range("2100-01-01", "2100-12-31", freq="D")

# Geçmiş verilerden mevsimsel ortalamaları hesaplama
seasonal_means = df_train_features.groupby("season").mean(numeric_only=True)

# 2100 yılı için giriş verisi oluşturma
future_seasons = future_dates.month.map(lambda month: 
    "Winter" if month in [12, 1, 2] else 
    "Spring" if month in [3, 4, 5] else 
    "Summer" if month in [6, 7, 8] else "Autumn")

simulated_future_features = pd.DataFrame([
    seasonal_means.loc[season] for season in future_seasons
], index=future_dates)

# Sütunları eşitleme
X_future = simulated_future_features[X_train.columns]

# nan kontrolü
print(X_future.isnull().sum())

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_future_scaled = scaler.transform(X_future)
# Ölçeklendirme öncesi nan doldurma
X_future_scaled=np.nan_to_num(X_future_scaled, nan=0.0)
print(np.isnan(X_future_scaled).any())
# Tahmin
future_predictions = alm_model.predict(X_future_scaled)
# Tahmin
#future_predictions = alm_model.predict(X_future)

# Tahmin sonuçlarını kaydetme
future_results = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Temperature': future_predictions
})
future_results.to_csv("2100_temperature_predictions.csv", index=False)

# 2100 tahminlerini görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(future_dates, future_predictions, label='Tahmin Edilen Sıcaklık (2100)', color='red')
plt.title('2100 Yılı Sıcaklık Tahminleri')
plt.xlabel('Tarih')
plt.ylabel('Sıcaklık')
plt.legend()
plt.show()

print("2100 yılı tahminleri kaydedildi.")
