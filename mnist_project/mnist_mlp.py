from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import MaxPool2D, Flatten
from keras import optimizers, losses

from keras.models import Model
from keras.models import Input

import keras

import sys
from load_mnist_data import load_mnist_data_vectors

def MLP(input_shape, num_classes):
    '''
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    '''

    inputs = Input(shape=input_shape)
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)

    return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist_data_vectors()

    epochs = 3
    batch_size = 128

    input_shape = (784,)
    num_classes = 10

    model = MLP(input_shape, num_classes)

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs_mlp')
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='./ckpt_mlp/weight.{epoch:02d}-{val_loss:.2f}.h5',
                                                   monitor='val_loss')

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[tensorboard, checkpointer],
                        validation_data=(x_test, y_test))

    model.save('model/mnist_mlp.h5')

    del model
    model = keras.models.load_model('model/mnist_mlp.h5')

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])