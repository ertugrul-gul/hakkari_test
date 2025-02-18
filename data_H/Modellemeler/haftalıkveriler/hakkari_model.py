import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("hakkari_0.csv")

plt.figure(figsize=(12,5))
plt.plot(df.index, df["t2m"], label="Sıcaklık")
plt.title("Zaman Serisi Grafiği")
plt.xlabel("Tarih")
plt.ylabel("Sıcaklık")
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(df.index, df["sp"], label="Basınç")
plt.title("Zaman Serisi Grafiği")
plt.xlabel("Tarih")
plt.ylabel("Basınç")
plt.legend()
plt.show()



