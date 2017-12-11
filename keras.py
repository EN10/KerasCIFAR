from load_cifar import *

download_extract()
imagearray, labelarray = load_batch()
x_train, y_train, x_test, y_test = train_test(imagearray, labelarray, 200, 50, 3)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=imagearray.shape[1]))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
              
model.fit(x_train, y_train, epochs=5, batch_size=32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print loss_and_metrics