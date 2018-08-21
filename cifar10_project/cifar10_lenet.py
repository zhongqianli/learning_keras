from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import MaxPool2D, Flatten
from keras import optimizers, losses

from keras.models import load_model

import keras

from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.models import Model
from keras.models import Input

from LeNet import LeNet
from load_cifar10_data import load_cifar10_data_metrics

import os

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_cifar10_data_metrics()

    log_dir = 'logs_cifar10_lenet'
    ckpt_dir = 'ckpt_cifar10_lenet'
    model_dir = 'model'

    if os.path.exists(log_dir) is False:
        os.makedirs(log_dir)
    if os.path.exists(ckpt_dir) is False:
        os.makedirs(ckpt_dir)
    if os.path.exists(model_dir) is False:
        os.makedirs(model_dir)

    epochs = 3
    batch_size = 2000
    input_shape = (32, 32, 3)
    num_classes = 10

    model = LeNet(input_shape, num_classes)

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adadelta(),
                  metrics=['accuracy'])

    print(model.summary())

    tensorboard = TensorBoard(log_dir='{0}'.format(log_dir))
    # weights.{epoch:02d}-{val_loss:.2f}.hdf5 : a kind of code rule
    ckpt_filepath = './{0}'.format(ckpt_dir) + '/weights.{epoch:02d}-{val_loss:.2f}.h5'
    checkpointer = ModelCheckpoint(filepath=ckpt_filepath, monitor='val_loss')

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[tensorboard, checkpointer],
                        validation_data=(x_test, y_test))

    model_name = '{0}/cifar10_lenet.h5'.format(model_dir)

    model.save(model_name)

    del model

    model = load_model(model_name)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
