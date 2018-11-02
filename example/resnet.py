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

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, add, MaxPooling2D, Flatten, Dense

def Unit(x,filters,pool=False):
    res = x
    if pool:
        x = MaxPooling2D(pool_size=(2, 2))(x)
        res = Conv2D(filters=filters,kernel_size=[1,1],strides=(2,2),padding="same")(res)
    out = Conv2D(filters, (1,1), padding='same', activation='relu')(x)
    out = Conv2D(filters, (1,1), padding='same', activation='relu')(out)
    out = add([res,out]) # res & out must be same shape
    return out   
  
images = Input(shape = x_train.shape[1:])
net = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(images)
net = Unit(net,64,pool=True)
net = Unit(net,128,pool=True)

net = Flatten()(net)
net = Dense(units=256,activation="relu")(net)
net = Dense(units=10,activation="softmax")(net)

model = Model(inputs=images,outputs=net)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.summary()

model.fit(x_train, y_train, batch_size=32,
          validation_data=(x_test, y_test),
          epochs=5)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', scores[1]) #  0.6964
