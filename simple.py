from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Activation
from tensorflow.python.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# preprocessing
x_train = x_train.astype('float32') # for division
x_test = x_test.astype('float32')
x_train /= 255 # normalise
x_test /= 255

# One-hot encode the labels
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(4, input_shape=x_train.shape[1:]))
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, batch_size=32)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', scores[1])  # 0.399