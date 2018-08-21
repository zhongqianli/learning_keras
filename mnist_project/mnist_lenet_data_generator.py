from keras.layers import Dense, Flatten
from keras.models import Model
import keras
from LeNet import LeNet

from load_mnist_data import load_mnist_data_from_directory_generator

import glob
import sys
import os
import platform

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def mnist_lenet(input_shape, classes):
    model = LeNet(input_shape, classes)

    optimizer = keras.optimizers.SGD(lr=0.01)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model


#/opt/win/tim.zhong/database/cifar10.224x224

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: {0} <dirpath>'.format(sys.argv[0]))
        # dirpath = 'E:/0-ML_database/handwriting/digits'
        if platform.system() == 'Windows':
            dirpath = 'E:/0-ML_database/mnist_abc'
        elif platform.system() == 'Linux':
            dirpath = '/opt/win/tim.zhong/database/mnist'
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            KTF.set_session(sess)
    else:
        dirpath = sys.argv[1]

    input_shape = (32, 32, 1)
    # mnist/train or mnist/test
    classes_subdir_list = ['0', '1', '2']
    num_classes = len(classes_subdir_list)
    batch_size = 128
    epochs = 10
    initial_epoch = 0
    script_name = os.path.basename(__file__).split('.')[0]
    log_dir = './logs_{0}'.format(script_name)
    ckpt_dir = './ckpt_{0}'.format(script_name)

    if os.path.exists(ckpt_dir) is False:
        os.makedirs(ckpt_dir)
    ckpt_filepath = ckpt_dir + '/weights.{epoch:02d}-{val_loss:.2f}.h5'

    model_dir = './model'
    if os.path.exists(model_dir) is False:
        os.makedirs(model_dir)
    model_name = '{0}/{1}.h5'.format(model_dir, script_name)

    rows = input_shape[0]
    cols = input_shape[1]

    train_generator, test_generator = load_mnist_data_from_directory_generator(
        dirpath=dirpath,
        target_size=(rows, cols),
        batch_size=batch_size,
        color_mode='grayscale',
        classes=classes_subdir_list)

    ckpt_filelist = glob.glob('{0}/weights.*.h5'.format(ckpt_dir))
    if len(ckpt_filelist) == 0:
        model = mnist_lenet(input_shape, num_classes)
    else:
        filename = ckpt_filelist[-1]
        model = keras.models.load_model(filepath=filename)
        initial_epoch = int(filename.split('weights.')[-1].split('-')[0])
        print('restore from {0}, initial_epoch = {1}'.format(filename, initial_epoch))

    model.summary()
    # plot_model(model, to_file='cifar10_resnet50.png', show_shapes=True)

    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath)

    history = model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        verbose=1,
                        callbacks=[tensorboard, checkpointer],
                        validation_data=test_generator,
                        validation_steps=len(test_generator),
                        initial_epoch=initial_epoch)

    model.save(model_name)

    del model
    model =  keras.models.load_model(model_name)

    score_evaluate = model.evaluate_generator(generator=test_generator, steps=len(test_generator))

    print(score_evaluate)

