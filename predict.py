from PIL import Image
import numpy as np

x = Image.open("image.png")
x = np.array(x)
x = np.resize(x,(1, 32, 32, 3))

from tensorflow.python.keras.models import load_model
model = load_model('keras_cifar10_trained_model.h5')

labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
y = model.predict(x)
print(labels[np.argmax(y)])