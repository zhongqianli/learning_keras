from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras import optimizers, losses
import keras
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model

import numpy as np
import cv2 as cv

epochs = 2
batch_size = 1000
img_rows, img_cols = 28, 28
num_classes = 10

def load_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def MLP():
    inputs = Input(shape=(784,))
    x = Dense(1024, activation='relu')(inputs)
    x = Dense(512, activation='relu')(x)
    predctions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predctions)
    return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

    model = MLP()

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adadelta(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])