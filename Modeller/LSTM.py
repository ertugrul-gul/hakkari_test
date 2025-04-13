import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Veriyi oku
df = pd.read_csv("../base_data/combined_data_cleaned_final.csv")

# Sonuçları toplamak için liste
results = []

sequence_length = 12
epochs = 20

groups = df.groupby(["latitude", "longitude"])

for (lat, lon), group in groups:
    print(f"Model eğitiliyor: lat={lat}, lon={lon} ...")

    group = group.sort_values("valid_time")

    # Girdi ve hedef
    X = group.drop(columns=["valid_time", "latitude", "longitude", "t2m", "tp"])
    y_t2m = group["t2m"].values.reshape(-1, 1)
    y_tp = group["tp"].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y_t2m = MinMaxScaler()
    scaler_y_tp = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_t2m_scaled = scaler_y_t2m.fit_transform(y_t2m)
    y_tp_scaled = scaler_y_tp.fit_transform(y_tp)

    split_idx = int(len(group) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_t2m_train, y_t2m_test = y_t2m_scaled[:split_idx], y_t2m_scaled[split_idx:]
    y_tp_train, y_tp_test = y_tp_scaled[:split_idx], y_tp_scaled[split_idx:]

    if len(X_train) <= sequence_length or len(X_test) <= sequence_length:
        print(f"lat={lat}, lon={lon} için veri yetersiz, atlanıyor.")
        continue

    gen_t2m = TimeseriesGenerator(X_train, y_t2m_train, length=sequence_length, batch_size=32)
    gen_tp = TimeseriesGenerator(X_train, y_tp_train, length=sequence_length, batch_size=32)

    # --- Sıcaklık (t2m) LSTM ---
    model_t2m = Sequential([
        Input(shape=(sequence_length, X.shape[1])),
        LSTM(64, activation='relu'),
        Dense(1)
    ])
    model_t2m.compile(optimizer='adam', loss='mse')
    model_t2m.fit(gen_t2m, epochs=epochs, verbose=0)

    X_test_seq = TimeseriesGenerator(X_test, y_t2m_test, length=sequence_length, batch_size=1)
    y_t2m_pred_scaled = model_t2m.predict(X_test_seq)
    y_t2m_true_scaled = y_t2m_test[sequence_length:]

    y_t2m_pred = scaler_y_t2m.inverse_transform(y_t2m_pred_scaled)
    y_t2m_true = scaler_y_t2m.inverse_transform(y_t2m_true_scaled)

    rmse_t2m = np.sqrt(mean_squared_error(y_t2m_true, y_t2m_pred))
    r2_t2m = r2_score(y_t2m_true, y_t2m_pred)

    # --- Yağış (tp) LSTM ---
    model_tp = Sequential([
        Input(shape=(sequence_length, X.shape[1])),
        LSTM(64, activation='relu'),
        Dense(1)
    ])
    model_tp.compile(optimizer='adam', loss='mse')
    model_tp.fit(gen_tp, epochs=epochs, verbose=0)

    X_test_seq_tp = TimeseriesGenerator(X_test, y_tp_test, length=sequence_length, batch_size=1)
    y_tp_pred_scaled = model_tp.predict(X_test_seq_tp)
    y_tp_true_scaled = y_tp_test[sequence_length:]

    y_tp_pred = scaler_y_tp.inverse_transform(y_tp_pred_scaled)
    y_tp_true = scaler_y_tp.inverse_transform(y_tp_true_scaled)

    rmse_tp = np.sqrt(mean_squared_error(y_tp_true, y_tp_pred))
    r2_tp = r2_score(y_tp_true, y_tp_pred)

    results.append({
        "latitude": lat,
        "longitude": lon,
        "rmse_t2m": rmse_t2m,
        "r2_t2m": r2_t2m,
        "rmse_tp": rmse_tp,
        "r2_tp": r2_tp
    })

# Sonuçları kaydet
results_df = pd.DataFrame(results)
results_df.to_csv("LSTM_performance_by_coordinate(sequence_length = 12 epochs = 20).csv", index=False)
print("Tüm modeller uyarısız şekilde eğitildi ve sonuçlar kaydedildi.")
