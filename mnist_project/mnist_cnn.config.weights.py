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
sys.path.append('cnn_models')
from cnn_models.LeNet import LeNet

sys.path.append('utils')
from utils.load_mnist_data import load_mnist_data_metrics

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist_data_metrics()

    model = load_model('model/mnist_lenet.h5')
    model.save_weights('model/mnist_lenet.weights.h5')

    model_config = model.to_json()
    print(model_config)

    import json
    with open('model/mnist_lenet.config.json', 'w') as f:
        json.dump(model_config, f)

    del model
    with open('model/mnist_lenet.config.json', 'r') as f:
        model_config = json.load(f)
    model = keras.models.model_from_json(model_config)
    model.load_weights('model/mnist_lenet.weights.h5', by_name=False)

    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adadelta(),
                  metrics=['accuracy'])

    print(model.summary())

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
