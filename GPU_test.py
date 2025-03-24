import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

# # GPU'nun algılanıp algılanmadığını kontrol et
gpu_available = tf.config.list_physical_devices('GPU')
print("GPU Available:", gpu_available)
#

#tf.config.list_physical_devices('GPU')

