from PIL import Image
from tensorflow.python.keras.models import load_model
import numpy as np

x = Image.open("cat.jpg")
x = np.array(x)
x = x.reshape(1,3072)
#x.tofile('catpil.txt', "\n")

model = load_model('cifar.h5')
y = model.predict(x)
print(np.argmax(y))