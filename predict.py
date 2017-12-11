import tensorflow as tf
from tensorflow.python.keras.models import load_model
import numpy as np

image = tf.image.decode_jpeg('cat.jpg')
print image.shape
x = tf.image.resize_images(image, [32,32])
x = x.flatten()

print x.shape

model = load_model('cifar.h5')
y = model.predict(x)
print(np.argmax(y))