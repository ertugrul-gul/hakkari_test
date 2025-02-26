import pandas as pd

# CSV dosyasını oku
df = pd.read_csv("dosya.csv")

# 4. ve 5. sütunu sil (Dikkat: Python'da index 0'dan başlar!)
df = df.drop(columns=df.columns[[3, 4]])

# Yeni CSV dosyasını kaydet
df.to_csv("yeni_dosya.csv", index=False)

print("Sütunlar başarıyla silindi ve yeni dosya kaydedildi.")
