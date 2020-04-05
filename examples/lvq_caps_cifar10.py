# -*- coding: utf-8 -*-
"""Train a LVQ Capsule Network on the Cifar10 dataset.

The network is inspired by the SimpleNet. See the paper for a description.

The network trains to an accuracy of >90% in ~200 epochs.
To train the network of the paper we applied the following schedule:
    0-50 epoch:     lr = 0.001
    51-100 epoch:   lr = 0.0005
    101-150 epoch:  lr = 0.0001
    151-200 epoch:  lr = 0.00005
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import timeit

from keras import layers, models, optimizers
from keras.utils import to_categorical
from keras.layers import *
from keras.initializers import *
from keras.utils.data_utils import get_file
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

from anysma import Capsule
from anysma.modules import InputModule, SplitModule
from anysma.modules.measuring import TangentDistance, RestrictedTangentDistance
from anysma.modules.routing import GibbsRouting
from anysma.modules.final import DissimilarityTransformation
from anysma.losses import generalized_kullback_leibler_divergence
from anysma.regularizers import MaxValue
from anysma.callbacks import TensorBoard
from anysma.datasets import cifar10

import matplotlib
matplotlib.use('Agg')  # needed to avoid cloud errors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

K.set_image_data_format('channels_last')


def LvqCapsNet(input_shape):
    input_img = layers.Input(shape=input_shape)

    # Block 1
    caps0 = Capsule()
    caps0.add(Conv2D(32+1, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps0.add(BatchNormalization())
    caps0.add(Activation('relu'))
    caps0.add(Dropout(0.25))
    x = caps0(input_img)

    # Block 2
    caps1 = Capsule()
    caps1.add(Conv2D(64+1, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps1.add(BatchNormalization())
    caps1.add(Activation('relu'))
    caps1.add(Dropout(0.25))
    x = caps1(x)

    # Block 3
    caps2 = Capsule()
    caps2.add(Conv2D(64+1, (3, 3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    caps2.add(BatchNormalization())
    caps2.add(Activation('relu'))
    caps2.add(Dropout(0.25))
    x = caps2(x)

    # Block 4
    caps3 = Capsule()
    caps3.add(Conv2D(64+1, (5, 5), strides=2, padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    caps3.add(BatchNormalization())
    caps3.add(Activation('relu'))
    caps3.add(Dropout(0.25))
    x = caps3(x)

    # Block 5
    caps4 = Capsule()
    caps4.add(Conv2D(32+1, (3, 3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    caps4.add(Dropout(0.25))
    x = caps4(x)

    # Block 6
    caps5 = Capsule()
    caps5.add(Conv2D(64+1, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps5.add(Dropout(0.25))
    x = caps5(x)

    # Block 7
    caps6 = Capsule()
    caps6.add(Conv2D(64 + 1, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps6.add(SplitModule())
    caps6.add(Activation('relu'), scope_keys=1)
    caps6.add(Flatten(), scope_keys=1)
    x = caps6(x)

    # Caps1
    caps7 = Capsule(prototype_distribution=(1, 8 * 8))
    caps7.add(InputModule(signal_shape=(16 * 16, 64), init_diss_initializer=None, trainable=False))
    diss7 = TangentDistance(squared_dissimilarity=False, epsilon=1.e-12, linear_factor=0.66, projected_atom_shape=16)
    caps7.add(diss7)
    caps7.add(GibbsRouting(norm_axis='channels', trainable=False))
    x = caps7(x)

    # Caps2
    caps8 = Capsule(prototype_distribution=(1, 4 * 4))
    caps8.add(Reshape((8, 8, 64)))
    caps8.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps8.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    caps8.add(InputModule(signal_shape=(8 * 8, 64), init_diss_initializer=None, trainable=False))
    diss8 = TangentDistance(projected_atom_shape=16, squared_dissimilarity=False,
                            epsilon=1.e-12, linear_factor=0.66, signal_output='signals')
    caps8.add(diss8)
    caps8.add(GibbsRouting(norm_axis='channels', trainable=False))
    x = caps8(x)

    # Caps3
    digit_caps = Capsule(prototype_distribution=(1, 10))
    digit_caps.add(Reshape((4, 4, 64)))
    digit_caps.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=glorot_normal()))
    digit_caps.add(InputModule(signal_shape=128, init_diss_initializer=None, trainable=False))
    diss = RestrictedTangentDistance(projected_atom_shape=16, epsilon=1.e-12, squared_dissimilarity=False,
                                     linear_factor=0.66, signal_output='signals')
    digit_caps.add(diss)
    digit_caps.add(GibbsRouting(norm_axis='channels', trainable=False,
                                diss_regularizer=MaxValue(alpha=0.0001)))
    digit_caps.add(DissimilarityTransformation(probability_transformation='neg_softmax', name='lvq_caps'))

    digitcaps = digit_caps(x)

    model = models.Model(input_img, digitcaps[2])

    return model


def train(model, data, args):
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    def train_generator(x, y, batch_size):
        train_datagen = ImageDataGenerator(width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           rotation_range=15,
                                           horizontal_flip=True)
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield (x_batch, y_batch)

    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                     batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
                                           monitor='val_acc', save_best_only=True,
                                           save_weights_only=True, verbose=1)

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1,
                                            patience=10, min_lr=0.00001)

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=generalized_kullback_leibler_divergence,
                  metrics={'lvq_caps': 'accuracy'})

    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[x_test, y_test],
                        callbacks=[log, tb, checkpoint, reduce_lr])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    return model


def test(model, data, args):
    x_test, y_test = data
    print('-' * 30 + 'Begin: test' + '-' * 30)

    start = timeit.default_timer()
    y_pred = model.predict(x_test, batch_size=args.batch_size)
    stop = timeit.default_timer()

    print('Inference time: ' + str(stop-start))
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])
    print('-' * 30 + 'End: test' + '-' * 30)


def plot_heatmaps(model, data, args):
    print('-' * 30 + 'Begin: plot heatmaps' + '-' * 30)

    for _ in range(10):
        idx = np.random.randint(0, data.shape[0])

        plt.clf()
        fig = plt.figure(1)
        gridspec.GridSpec(1, 10)

        plt.subplot2grid((1, 10), (0, 0), colspan=2)
        ax = plt.gca()
        img = plt.imshow(data[idx, :, :, :])
        img.set_cmap('gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('input')

        plt.subplot2grid((1, 10), (0, 2), colspan=2)
        out = models.Model(model.input, model.get_layer('activation_5').output).predict(data[idx:idx+1])
        ax = plt.gca()
        img = ax.imshow(out[0, :, :, 0])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.03)
        plt.colorbar(img, cax=cax)
        ax.set_title('d_in Caps1')

        plt.subplot2grid((1, 10), (0, 4), colspan=2)
        out = np.reshape(models.Model(model.input,
                                      model.get_layer('gibbs_routing_1').output[1]).predict(data[idx:idx+1]),
                         (-1, 8, 8, 1))
        ax = plt.gca()
        img = ax.imshow(out[0, :, :, 0])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.03)
        plt.colorbar(img, cax=cax)
        ax.set_title('d_in Caps2')

        plt.subplot2grid((1, 10), (0, 6), colspan=2)
        out = np.reshape(models.Model(model.input,
                                      model.get_layer('gibbs_routing_2').output[1]).predict(data[idx:idx+1]),
                         (-1, 4, 4, 1))
        ax = plt.gca()
        img = ax.imshow(out[0, :, :, 0])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.03)
        plt.colorbar(img, cax=cax)
        ax.set_title('d_in caps2')

        plt.subplot2grid((1,10), (0,8))
        out = np.reshape(models.Model(model.input,
                                      model.get_layer('gibbs_routing_3').output[1]).predict(data[idx:idx+1]),
                         (-1, 10, 1, 1))
        ax = plt.gca()
        img = ax.imshow(np.tile(out[0, :, :, 0], (1, 4)))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.03)
        plt.colorbar(img, cax=cax)
        ax.set_xticks([])
        ax.set_title('d_out')

        plt.subplot2grid((1, 10), (0, 9))
        out = np.reshape(models.Model(model.input,
                                      model.get_layer('lvq_caps').output[2]).predict(data[idx:idx+1]),
                         (-1, 10, 1, 1))
        ax = plt.gca()
        img = ax.imshow(np.tile(out[0, :, :, 0], (1, 4)))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.03)
        plt.colorbar(img, cax=cax)
        ax.set_xticks([])
        ax.set_title('p_out')

        # we have to call it a couple of times to adjust the plot correctly
        for _ in range(5):
            plt.tight_layout(pad=0, w_pad=0.1, h_pad=0.0)
            fig.set_size_inches(w=9, h=1.5)
        plt.savefig(args.save_dir + '/heatmap_' + str(idx) + '.png')

    print('-' * 30 + 'End: plot heatmaps' + '-' * 30)


def plot_spectral_lines(model, data, args):
    print('-' * 30 + 'Begin: plot spectral lines' + '-' * 30)

    x_test, y_test = data

    out = models.Model(model.input,
                       model.get_layer('gibbs_routing_2').output[1]).predict(x_test[:1000],
                                                                             batch_size=args.batch_size)

    plt.clf()
    fig = plt.figure(1)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for idx in range(10):
        plt.subplot(5, 2, idx + 1)
        tmp = out[y_test[:1000, idx].astype('bool'), :]

        plt.plot(np.arange(1, 17), tmp.transpose())
        ax = plt.gca()
        ax.set_title('class ' + str(idx) + ': ' + classes[idx])

    # we have to call it a couple of times to adjust the plot correctly
    for _ in range(5):
        plt.tight_layout(pad=0, w_pad=0.1, h_pad=0.0)
        fig.set_size_inches(w=10, h=8)
    plt.savefig(args.save_dir + '/spectral_lines' + '.png')

    print('-' * 30 + 'End: plot spectral lines' + '-' * 30)


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="LVQ Capsule network on Cifar10.")
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help='The `weights` argument should be either `None` (random initialization), `cifar10` or the '
                             'path to the weights file to be loaded.')
    args = parser.parse_args()
    print(args)

    # load data
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # define model
    model = LvqCapsNet(input_shape=x_train.shape[1:])
    model.summary(line_length=200, positions=[.33, .6, .67, 1.])

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not (args.weights in {'cifar10', None} or os.path.exists(args.weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `cifar10` '
                         'or the path to the weights file to be loaded.')

    if args.weights is not None:  # init the model weights with provided one
        if args.weights == 'cifar10':
            args.weights = get_file('cifar10.h5',
                                    'http://tiny.cc/anysma_models_cifar10',
                                    cache_subdir='models',
                                    file_hash='d4dbd5ec5babd8f545f81bc231b62564')
        model.load_weights(args.weights)

    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        plot_spectral_lines(model, (x_test, y_test), args)
        plot_heatmaps(model, x_test, args)
        test(model=model, data=(x_test, y_test), args=args)
