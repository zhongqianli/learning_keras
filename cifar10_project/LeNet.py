from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import MaxPool2D, Flatten
from keras import optimizers, losses

from keras.models import Model
from keras.models import Input

def LeNet(input_shape, num_classes):
    inputs = Input(input_shape)
    x = Conv2D(6, (5, 5), activation='relu')(inputs)
    x = MaxPool2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu')(x)
    x = MaxPool2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(84, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model =  Model(inputs=inputs, outputs=outputs)

    return model