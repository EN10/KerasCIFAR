from tensorflow.python.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# preprocessing
x_train = x_train.astype('float32') # for division
x_test = x_test.astype('float32')
x_train /= 255 # normalise
x_test /= 255

from tensorflow.python.keras.utils import to_categorical
# One-hot encode the labels
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


from tensorflow.python.keras.layers import Input, Conv2D, AveragePooling2D, concatenate, Flatten, Dense
input_img = Input(shape = x_train.shape[1:]) # (32, 32, 3)

def inception(input):
  pool_1 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
  conv_1 = Conv2D(64, (1,1), padding='same', activation='relu')(pool_1)
  conv_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input)
  conv_3 = Conv2D(64, (3,3), padding='same', activation='relu')(conv_2)
  conv_4 = Conv2D(64, (3,3), padding='same', activation='relu')(conv_3)
  concat = concatenate([conv_1, conv_2, conv_3, conv_4], axis = 3)
  return concat
 
concat_1 = inception(input_img)

output = Flatten()(concat_1)
out    = Dense(10, activation='softmax')(output)

from tensorflow.python.keras.models import Model
model = Model(inputs = input_img, outputs = out)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


from tensorflow.python.keras.models import load_model
from keras.callbacks import ModelCheckpoint
filepath='checkpoint.h5'
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)

import os.path
if os.path.exists(filepath): model = load_model(filepath)

model.fit(x_train, y_train, batch_size=32,
          validation_data=(x_test, y_test),
          callbacks=[checkpoint],
          epochs=5)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', scores[1]) #  0.6446
