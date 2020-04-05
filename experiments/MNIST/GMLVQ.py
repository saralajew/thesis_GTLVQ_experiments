# -*- coding: utf-8 -*-
""" This is the script to reproduce the results on the MNIST dataset of
Section 3.4.2 for the GMLVQ and GMLVQ-1M model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import os

from keras.layers import Input
from keras.models import Model
from keras.datasets import mnist
from keras import callbacks
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator

from anysma import Capsule
from anysma.capsule import InputModule
from anysma.modules.measuring import OmegaDistance
from anysma.modules.routing import SqueezeRouting
from anysma.modules.competition import NearestCompetition
from anysma.losses import GlvqLossOverDissimilarities as GlvqLoss

import cv2

import matplotlib
matplotlib.use('Agg')  # needed to avoid cloud errors

# load dataset and preprocess
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
y_train = to_categorical(y_train.astype('float32'))
y_test = to_categorical(y_test.astype('float32'))

# network parameters
input_shape = (28, 28, 1)
batch_size = 128
epochs = 150
lr = 0.001


def get_model(protos_per_class=1):
    inputs = Input(shape=input_shape)
    diss = OmegaDistance(linear_factor=None,
                         squared_dissimilarity=True,
                         matrix_scope='global',
                         matrix_constraint='OmegaNormalization',
                         signal_output='signals',
                         matrix_initializer='identity')

    caps = Capsule(prototype_distribution=(protos_per_class, 10))
    caps.add(InputModule(signal_shape=(-1, np.prod(input_shape)),
                         trainable=False,
                         init_diss_initializer='zeros'))
    caps.add(diss)
    caps.add(SqueezeRouting())
    caps.add(NearestCompetition())

    output = caps(inputs)[1]

    # pre-train the model over 10000 random digits
    # skip the svd for GMLVQ
    _, matrix = diss.get_weights()

    idx = np.random.randint(0, len(x_train) - 1, (min(10000, len(x_train)),))
    pre_train_model = Model(inputs=inputs, outputs=diss.input[0])
    diss_input = pre_train_model.predict(x_train[idx, :],
                                         batch_size=batch_size)
    diss.pre_training(diss_input, y_train[idx], capsule_inputs_are_equal=True)

    # set identity matrices
    centers, _ = diss.get_weights()
    diss.set_weights([centers, matrix])

    # define model and return
    model = Model(inputs, output)

    return model


def plot(model, protos_per_class, path):
    if not os.path.exists(path):
        os.mkdir(path)

    protos = model.get_layer(index=2).get_weights()[0]
    protos = np.reshape(protos, (-1, 28, 28, 1))
    for i in range(10):
        for j in range(protos_per_class):
            # color pixels outside [0, 1] blue and red
            proto = protos[i * protos_per_class + j, :, :]
            lower = proto < 0
            greater = proto > 1
            proto = np.repeat(proto, 3, 2)
            proto = proto * (1 - lower) + lower * [[[1, 0, 0]]]
            proto = proto * (1 - greater) + greater * [[[0, 0, 1]]]

            cv2.imwrite('{}/proto_{}_{}.png'.format(path, i, j),
                        (proto * 255).astype('uint8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights",
                        help="Load h5 model trained weights.")
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--gpu', default=0, type=int,
                        help='Select GPU device.')
    parser.add_argument('--eval', action='store_true',
                        help='Only perform the evaluation.')
    parser.add_argument('-m', action='store_true',
                        help='Call 1M parameter setting.')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define number of prototypes
    if args.m:
        protos_per_class = 49
    else:
        protos_per_class = 1

    # get model
    train_model = get_model(protos_per_class)
    train_model.summary()

    # load weights if available
    if args.weights:
        train_model.load_weights(args.weights)

    # accuracy metric for GLVQ methods (sign has to be switched since the
    # minimal value determines the winner)
    def acc(y_true, y_pred):
        return metrics.categorical_accuracy(y_true, -y_pred)

    train_model.compile(optimizer=Adam(lr=lr),
                        loss=GlvqLoss(),
                        metrics=[acc])

    # get training data generator with augmentation
    def train_generator(x, y, batch_size):
        train_datagen = ImageDataGenerator(width_shift_range=2,
                                           height_shift_range=2,
                                           rotation_range=15)

        generator = train_datagen.flow(x, y, batch_size=batch_size)

        while True:
            batch_x, batch_y = generator.next()
            yield batch_x, batch_y

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(args.save_dir +
                                           '/weights-{epoch:02d}.h5',
                                           save_best_only=True,
                                           save_weights_only=True, verbose=1)
    csv_logger = callbacks.CSVLogger(args.save_dir + '/log.csv')
    lr_reduce = callbacks.ReduceLROnPlateau(factor=0.5, monitor='val_loss',
                                            mode='min', verbose=1, patience=5)

    # train the model
    if not args.eval:
        train_model.fit_generator(generator=train_generator(x_train, y_train,
                                                            batch_size),
                                  steps_per_epoch=
                                  int(y_train.shape[0] / batch_size),
                                  epochs=epochs,
                                  validation_data=[x_test, y_test],
                                  callbacks=[checkpoint, csv_logger,
                                             lr_reduce],
                                  max_queue_size=40,
                                  workers=3,
                                  use_multiprocessing=True,
                                  verbose=1)

        train_model.save_weights(args.save_dir + '/trained_model.h5')

    print('training results (without augmentation!):')
    result = train_model.evaluate(x_train, y_train, batch_size=batch_size)
    print('loss: {}   accuracy: {}'.format(result[0], result[1]))
    print('test results:')
    result = train_model.evaluate(x_test, y_test, batch_size=batch_size)
    print('loss: {}   accuracy: {}'.format(result[0], result[1]))

    plot(train_model, protos_per_class, args.save_dir + '/prototypes')
