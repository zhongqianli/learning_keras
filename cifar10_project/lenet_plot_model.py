from keras.layers import Dense, Conv2D
from keras.layers import MaxPool2D, Flatten
from keras import optimizers, losses

from keras.models import load_model

import keras

from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.models import Model
from keras.models import Input

import sys
sys.path.append('cnn_models')
from cnn_models.LeNet import LeNet

sys.path.append('utils')
from utils.load_cifar10_data import load_cifar10_data_metrics

from keras.utils import plot_model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_cifar10_data_metrics()

    epochs = 3
    batch_size = 2000
    input_shape = (32, 32, 3)
    num_classes = 10

    model = LeNet(input_shape, num_classes)

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adadelta(),
                  metrics=['accuracy'])

    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')
    print(model.summary())