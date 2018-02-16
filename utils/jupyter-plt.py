from tensorflow.python.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)

import matplotlib.pyplot as plt
labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

i = 1
plt.imshow(x_train[i])
print labels[int(y_train[i])]