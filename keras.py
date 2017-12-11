from load_cifar import *

download_extract()
imagearray, labelarray = load_batch()
train_set_x, train_set_y, test_set_x, test_set_y = train_test(imagearray, labelarray, 200, 50, 3)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
              
model.fit(x_train, y_train, epochs=5, batch_size=32)

print model.evaluate(x_test, y_test, batch_size=128)