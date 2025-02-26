## 1️⃣ XGBoost Modeli (xgboost_model.py)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n🔹 {model_name} Performansı:")
    print(f"📌 MAE: {mae}")
    print(f"📌 RMSE: {rmse}")

# Veriyi yükleme
def load_data():
    df = pd.read_csv("base_data/hakkari_0.csv")
    df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
    df["year"] = df["valid_time"].dt.year
    df["month"] = df["valid_time"].dt.month
    df["day"] = df["valid_time"].dt.day
    return df

df = load_data()
X = df[["year", "month", "day", "lat", "lon", "sp", "u10", "v10"]]
y = df[["t2m", "tp"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_Y.fit_transform(y_train)
y_test_scaled = scaler_Y.transform(y_test)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_scaled)
evaluate_model(y_test, y_pred_xgb, "XGBoost")