# -*- coding: utf-8 -*-
""" This is the script to reproduce the results on the Indian Pine dataset of
Section 3.4.2.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import os

from keras.layers import Input
from keras.models import Model
from keras import callbacks
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import metrics
from keras import backend as K

from anysma import Capsule
from anysma.capsule import InputModule
from anysma.modules.measuring import TangentDistance
from anysma.modules.routing import SqueezeRouting
from anysma.losses import GlvqLossOverDissimilarities as GlvqLoss
from anysma.modules.competition import NearestCompetition

import spectral
import scipy.io as sio
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')  # needed to avoid cloud errors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from spectral import spy_colors


def load_indian_pine_data(data_path='./data'):
    data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))[
        'indian_pines_corrected']
    labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))[
        'indian_pines_gt']

    return data, labels


# load dataset and preprocess
X, Y = load_indian_pine_data()

# reshape
X = X.reshape(-1, 200)
Y = Y.reshape(-1)

# select labeled data points
not_labeled = Y == 0
labeled = Y > 0
x_train = X[labeled]
y_train = Y[labeled]

# make a stratified random split into train and test
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                    test_size=0.2,
                                                    stratify=y_train,
                                                    random_state=12)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = to_categorical(y_train.astype('float32') - 1)
y_test = to_categorical(y_test.astype('float32') - 1)

# sample-wise normalization
x_train = (x_train - np.mean(x_train, 1, keepdims=True)) / \
          np.std(x_train, 1, keepdims=True)
x_test = (x_test - np.mean(x_test, 1, keepdims=True)) / \
         np.std(x_test, 1, keepdims=True)

# get class distribution for weighted training
class_weight = np.sum(y_train, 0) / y_train.shape[0]

# network parameters
input_shape = (200,)
batch_size = 64
epochs = 20
lr = 0.001


def get_model():
    inputs = Input(shape=input_shape)
    diss = TangentDistance(linear_factor=None,
                           squared_dissimilarity=True,
                           projected_atom_shape=15,
                           signal_output='signals')

    # compute the prototype distribution
    proto_distrib = list(np.minimum(
        np.ceil(np.sum(y_train, 0) / 100).astype('int'), 5))
    # class 'Buildings-Grass-Trees-Drives'
    proto_distrib[-2] = 2
    # class 'Corn-mintill'
    proto_distrib[2] = 4
    print('proto_distrib: ' + str(proto_distrib))

    # define capsule network
    caps = Capsule(prototype_distribution=proto_distrib)
    caps.add(InputModule(signal_shape=(-1, np.prod(input_shape)),
                         trainable=False,
                         init_diss_initializer='zeros'))
    caps.add(diss)
    caps.add(SqueezeRouting())
    caps.add(NearestCompetition())

    output = caps(inputs)[1]

    # pre-train the model
    pre_train_model = Model(inputs=inputs, outputs=diss.input[0])
    diss_input = pre_train_model.predict(x_train, batch_size=batch_size)
    diss.pre_training(diss_input, y_train, capsule_inputs_are_equal=True)

    # define model and return
    model = Model(inputs, output)

    return model


def plot_spectral_lines(model, path):
    t, B = model.get_layer(index=2).get_weights()

    # plot samples of learned prototypes
    # prototype number:
    #   0  --> class 0: 'alfalfa'
    #   51 --> class 15: 'stone-steel towers'
    for idx in [0, 51]:
        theta = np.zeros(15)
        plt.clf()
        fig, ax = plt.subplots(1)
        x = t[idx] + np.dot(B[idx], theta)
        plt.plot(np.arange(200), x, 'b', )
        for _ in range(50):
            theta = np.random.randn(15) / 3
            x = t[idx] + np.dot(B[idx], theta)
            ax.plot(np.arange(200), x, 'b', alpha=0.05)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax.set_xlabel('band', fontsize=18)
        ax.set_ylabel('reflectance', fontsize=18)
        plt.ylim([-1.5, 3.5])
        plt.tight_layout()
        plt.savefig(path + '/prototype' + str(idx) + '.png')

    # plot input spectra
    for i in [0, 15]:
        plt.clf()
        fig, ax = plt.subplots(1)
        idx = np.argmax(y_train, -1) == i
        xx = x_train[idx]
        for x in xx:
            ax.plot(np.arange(200), x, 'b', alpha=0.1)

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax.set_xlabel('band', fontsize=18)
        ax.set_ylabel('reflectance', fontsize=18)
        plt.ylim([-1.5, 3.5])
        plt.tight_layout()
        plt.savefig(path + '/input' + str(i) + '.png')


def plot_predicted_output(model, path):
    classes = ['alfalfa',
               'corn-no-till',
               'corn-min-till',
               'corn-clean',
               'grass/pasture',
               'grass/trees',
               'grass/pasture-mowed',
               'hay-windrowed',
               'oats',
               'soybean-no-till',
               'soybean-min-till',
               'soybean-clean',
               'wheat',
               'woods',
               'buildings/grass/trees/drives',
               'stone-steel towers',
               ]

    # plot legend
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(5, 6.5)
    ax = plt.subplot()  # create the axes
    ax.set_axis_off()  # turn off the axis
    labelPatches = [Patch(color=spy_colors[x] / 255.,
                          label=classes[x - 1]) for x in range(1, 17)]
    plt.legend(handles=labelPatches, ncol=1, fontsize=18)
    plt.tight_layout()
    plt.savefig(path + '/legend.png')

    # ground truth
    plt.clf()
    spectral.imshow(classes=Y.reshape(145, 145))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path + '/ground_truth.png')

    # predicted results
    plt.clf()
    xx = X.reshape(-1, 200)
    xx = (xx - np.mean(xx, 1, keepdims=True)) / \
         np.std(xx, 1, keepdims=True)
    y_pred = model.predict(xx.reshape(-1, 200))
    y_pred = np.argmin(y_pred, -1) + 1
    y_pred[not_labeled] = 0
    spectral.imshow(classes=y_pred.reshape(145, 145))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path + '/predicted.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights",
                        help="Load h5 model trained weights.")
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--gpu', default=0, type=int,
                        help='Select GPU device.')
    parser.add_argument('--eval', action='store_true',
                        help='Only perform the evaluation.')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # get model
    train_model = get_model()
    train_model.summary()

    # load weights if available
    if args.weights:
        train_model.load_weights(args.weights)

    # accuracy metric for GLVQ methods (sign has to be switched since the
    # minimal value determines the winner)
    def acc(y_true, y_pred):
        return metrics.categorical_accuracy(y_true, -y_pred)

    glvq_loss = GlvqLoss(squash_func=lambda x: K.relu(x + 0.3))
    train_model.compile(optimizer=Adam(lr=lr),
                        loss=glvq_loss,
                        metrics=[acc])

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(args.save_dir +
                                           '/weights-{epoch:02d}.h5',
                                           save_best_only=True,
                                           save_weights_only=True, verbose=1)
    csv_logger = callbacks.CSVLogger(args.save_dir + '/log.csv')
    lr_reduce = callbacks.ReduceLROnPlateau(factor=0.5, monitor='val_loss',
                                            verbose=1, patience=5)

    # train the model
    if not args.eval:
        train_model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[checkpoint, csv_logger, lr_reduce],
                        class_weight=class_weight,
                        validation_data=[x_test, y_test])

        train_model.save_weights(args.save_dir + '/trained_model.h5')

    print('training results:')
    result = train_model.evaluate(x_train, y_train, batch_size=batch_size)
    print('loss: {}   accuracy: {}'.format(result[0], result[1]))
    print('test results:')
    result = train_model.evaluate(x_test, y_test, batch_size=batch_size)
    print('loss: {}   accuracy: {}'.format(result[0], result[1]))

    # plot results
    plot_predicted_output(train_model, args.save_dir)
    plot_spectral_lines(train_model, args.save_dir)
