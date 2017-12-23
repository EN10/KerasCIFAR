from PIL import Image
import numpy as np

x = Image.open("lcat.png")
x = np.array(x)
x = np.resize(x,(32, 32, 3))

from tensorflow.python.keras.models import load_model
model = load_model('keras_cifar10_trained_model.h5')
y = model.predict(x)
print(np.argmax(y))