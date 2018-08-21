from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import MaxPool2D, Flatten
from keras import optimizers, losses

from keras.models import Model
from keras.models import Input

def LeNet(input_shape, num_classes):
    '''
    model = Sequential()
    model.add(Conv2D(6, (4, 4), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(16, (4, 4), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    '''

    '''
    model = Sequential(
        [
            Conv2D(6, (4, 4), activation='relu', input_shape=input_shape),
            MaxPool2D((2, 2)),
            Conv2D(6, (4, 4), activation='relu', input_shape=input_shape),
            MaxPool2D((2, 2)),
            Flatten(),
            Dense(120, activation='relu'),
            Dense(84, activation='relu'),
            Dense(num_classes, activation='softmax')
        ]
    )
    '''

    inputs = Input(input_shape, name='input_1')
    x = Conv2D(6, (5, 5), activation='relu', name='conv1')(inputs)
    x = MaxPool2D((2, 2), name='max_pooling2d_1')(x)
    x = Conv2D(16, (5, 5), activation='relu', name='conv2')(x)
    x = MaxPool2D((2, 2), name='max_pooling2d_2')(x)
    x = Flatten(name='flatten_1')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    model =  Model(inputs=inputs, outputs=predictions)

    return model

if __name__ == '__main__':
    import sys
    sys.path.append('../utils')
    from load_mnist_data import load_mnist_data_metrics
    (x_train, y_train), (x_test, y_test) = load_mnist_data_metrics()

    batch_size = 128
    epochs = 1

    input_shape = (28, 28, 1)
    num_classes = 10

    model = LeNet(input_shape, num_classes)
    optimizer = optimizers.SGD(lr=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=[x_test, y_test])
    score = model.evaluate(x_test, y_test)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

