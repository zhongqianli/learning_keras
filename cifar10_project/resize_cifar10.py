from keras.datasets import cifar10
import cv2
import os
import sys

def resize_cifar10(filepath, width, height):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    count = 0
    for x, y in zip(x_train, y_train):
        count = count + 1
        dir = '{0}/train/{1}'.format(filepath, int(y))
        if os.path.exists(dir) is False:
            os.makedirs(dir)
        filename = '{0}/{1}.jpg'.format(dir, count)
        print(filename)
        if os.path.exists(filename) is False:
            x = cv2.resize(x, (width, height))
            cv2.imwrite(filename, x)
        # cv2.imshow('train', x)
        # cv2.waitKey()

    for x, y in zip(x_test, y_test):
        count = count + 1
        dir = '{0}/test/{1}'.format(filepath, int(y))
        if os.path.exists(dir) is False:
            os.makedirs(dir)
        filename = '{0}/{1}.jpg'.format(dir, count)
        print(filename)
        if os.path.exists(filename) is False:
            x = cv2.resize(x, (width, height))
            cv2.imwrite(filename, x)

if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv) < 4:
        print('usage: {0} <filepath> <width> <height>'.format(sys.argv[0]))
        exit(-1)
    filepath = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    resize_cifar10(filepath, width, height)