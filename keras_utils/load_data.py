import keras
from keras.preprocessing.image import ImageDataGenerator

def load_data_from_directory_generator(dirpath, target_size = (224, 224), batch_size=32, color_mode='rgb', class_mode='categorical', classes_subdir_list=None):
    '''
    dirpath: base dir, e.g.: xxx/datasets/mnist, which contains subdirs: train and test
    target_size: resize to target_size
    batch_size:
    color_mode: rgb or grayscale
    class_mode:
    classes_subdir_list: list of subdirs in train or test,
        e.g.: xxx/datasets/mnist/train/0, xxx/datasets/mnist/train/1, ..., xxx/datasets/mnist/train/9
        classes_subdir_list = ['0', '1', '2', '3', '4', ..., '9'], they means subdir_names
    '''

    train_datagen = ImageDataGenerator(rescale=1 / 255.0)

    test_datagen = ImageDataGenerator(rescale=1 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        '{0}/train'.format(dirpath),
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode=color_mode,
        classes=classes_subdir_list)

    test_generator = test_datagen.flow_from_directory(
        '{0}/test'.format(dirpath),
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode=color_mode,
        classes=classes_subdir_list)

    return train_generator, test_generator

if __name__ == '__main__':
    import cv2
    import numpy as np
    train_generator, test_generator = load_data_from_directory_generator(dirpath='E:/0-ML_database/mnist_abc',
                                                                         target_size=(224, 224),
                                                                         batch_size=32,
                                                                         color_mode='grayscale',
                                                                         class_mode='categorical',
                                                                         classes_subdir_list=['0', '1', '2'])

    for data in train_generator:
        xs, ys = data
        for x, y in zip(xs, ys):
            cv2.imshow('x', x)
            print(y)
            cv2.waitKey()