from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
import numpy as np

img = image.load_img('cat.jpg', target_size=(32, 32))
x = image.img_to_array(img)
x = x.reshape(1,3072)   #   (32, 32, 3) to  (1, 3072)
x = x.astype(int)

#x.tofile('cat.txt', "\n")

model = load_model('cifar.h5')
y = model.predict(x)
print(np.argmax(y))