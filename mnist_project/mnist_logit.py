from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import MaxPool2D, Flatten
from keras import optimizers, losses

from keras.models import Model
from keras.models import Input

import keras

import sys
from load_mnist_data import load_mnist_data_vectors

def logistic_regression(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    outputs = Dense(num_classes, activation='softmax')(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist_data_vectors()

    epochs = 10
    batch_size = 128

    input_shape = (784,)
    num_classes = 10

    model = logistic_regression(input_shape, num_classes)

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs_logit')
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='./ckpt_logit/weigths.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss')

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[tensorboard, checkpointer],
              validation_data=(x_test, y_test))

    model.save('model/mnist_logit.h5')
    del model
    model = keras.models.load_model('model/mnist_logit.h5')

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])