from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)