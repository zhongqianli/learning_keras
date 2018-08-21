from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import MaxPool2D, Flatten
from keras import optimizers, losses

from keras.models import load_model

import keras

from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.models import Model
from keras.models import Input

import sys
from LeNet import LeNet
from load_mnist_data import load_mnist_data_metrics

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist_data_metrics()

    initial_epoch = 0
    epochs = 20
    batch_size = 2000
    input_shape = (28, 28, 1)
    num_classes = 10

    import glob
    ckpt_filelist = glob.glob('./ckpt_lenet/*.h5')
    if len(ckpt_filelist) == 0:
        model = LeNet(input_shape, num_classes)
        model.compile(loss=losses.categorical_crossentropy,
                      optimizer=optimizers.Adadelta(),
                      metrics=['accuracy'])
    else:
        filename = ckpt_filelist[-1]
        model = load_model(filename)
        initial_epoch = int(filename.split('weights.')[-1].split('-')[0])
        print('restore from ckpt, initial_echo = {0}'.format(initial_epoch))

    print(model.summary())

    tensorboard = TensorBoard(log_dir='./logs_lenet')
    # weights.{epoch:02d}-{val_loss:.2f}.hdf5 : a kind of code rule
    checkpointer = ModelCheckpoint(filepath='./ckpt_lenet/weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss')

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[tensorboard, checkpointer],
                        validation_data=(x_test, y_test),
                        initial_epoch=initial_epoch)

    model.save('model/mnist_lenet.h5')

    del model

    model = load_model('model/mnist_lenet.h5')

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
