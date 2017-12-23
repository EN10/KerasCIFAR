from PIL import Image
import numpy as np

x = Image.open("cat.png")
x = np.array(x)
x = x.reshape(x.shape[0]*x.shape[1], -1).T
x = x.reshape(1,3072)
x = x / 255.

from tensorflow.python.keras.models import load_model
model = load_model('keras_cifar10_trained_model.h5')
y = model.predict(x)
print(np.argmax(y))