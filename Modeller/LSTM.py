## 3️⃣ LSTM Modeli (lstm_model.py)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n🔹 {model_name} Performansı:")
    print(f"📌 MAE: {mae}")
    print(f"📌 RMSE: {rmse}")

# Veriyi yükleme

df = pd.read_csv("../base_data/hakkari_0.csv")
df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
df["year"] = df["valid_time"].dt.year
df["month"] = df["valid_time"].dt.month
df["day"] = df["valid_time"].dt.day
X = df[["year", "month", "day", "lat", "lon", "sp", "u10", "v10"]]
y = df[["t2m", "tp"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_Y.fit_transform(y_train)
y_test_scaled = scaler_Y.transform(y_test)

X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lstm_model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
    Dropout(0.2),
    LSTM(25, activation='relu'),
    Dropout(0.2),
    Dense(2)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
lstm_model.fit(X_train_lstm, y_train_scaled, epochs=20, batch_size=32, validation_data=(X_test_lstm, y_test_scaled), verbose=1)

y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
y_pred_lstm = scaler_Y.inverse_transform(y_pred_lstm_scaled)

evaluate_model(y_test, y_pred_lstm, "LSTM")