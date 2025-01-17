import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Saçma sapan sıcaklık verileri oluştur
dates = pd.date_range('1990-01-01', '2020-12-31', freq='M')
data = pd.DataFrame(np.random.randn(len(dates)) * 10 + 15, index=dates, columns=['Temperature'])  # Ortalama 15°C, sapma 10°C

# ARIMA ile biraz eğlenelim
model = ARIMA(data['Temperature'], order=(1, 1, 1))  # Evet, bu parametreler uydurma olabilir, sen de uydurabilirsin
results = model.fit()

# Geleceği tahmin edelim, bakalım ne yapacak bu model
future_dates = pd.date_range(start=data.index[-1], periods=120, freq='M')
forecast = results.forecast(steps=120)

# Şimdi bunları grafik olarak görselleştir
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Temperature'], label='Geçmiş Sıcaklıklar (Gerçekten bunlar mı?)')
plt.plot(future_dates, forecast, label='Gelecekteki Sıcaklıklar (Umarım haklı çıkarız)')
plt.legend()
plt.title('Gelecekteki Sıcaklıklar - Belki de Tamamen Yanlış')
plt.show()