from PIL import Image
import numpy as np
from tensorflow.python.keras.models import load_model

x = Image.open("cat.jpg")
x = np.array(x)

x.tofile('cat.txt', "\n")

x = x.reshape(1,3072) / 255

model = load_model('cifar.h5')
y = model.predict(x)
print(np.argmax(y))