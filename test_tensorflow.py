import tensorflow as tf

# GPU'ları kontrol et
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU'lar başarıyla algılandı: {gpus}")
else:
    print("GPU algılanamadı, sadece CPU kullanılacak.")
