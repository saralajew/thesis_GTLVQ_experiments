# -*- coding: utf-8 -*-
""" This is the script to reproduce the results on the Spiral dataset of
Section 3.4.1.
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
from anysma.modules.measuring import MinkowskiDistance, \
    RestrictedTangentDistance, OmegaDistance
from anysma.modules.routing import SqueezeRouting
from anysma.losses import GlvqLossOverDissimilarities as GlvqLoss
from anysma.modules.competition import NearestCompetition

from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')  # needed to avoid cloud errors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap


# load data (if not available call the data generation script)
x = np.load('./data/points.npy')
y = np.load('./data/labels.npy')

# make a stratified random split into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=12)

y_train = to_categorical(y_train.astype('float32'))
y_test = to_categorical(y_test.astype('float32'))

# network parameters
input_shape = (2,)
batch_size = 24
epochs = 200
lr = 0.001


def get_model():
    inputs = Input(shape=input_shape)

    # get the dissimilarity either GLVQ, GTLVQ, or GMLVQ
    if args.mode == 'glvq':
        diss = MinkowskiDistance(linear_factor=None,
                                 squared_dissimilarity=True,
                                 signal_output='signals')
    elif args.mode == 'gtlvq':
        diss = RestrictedTangentDistance(linear_factor=None,
                                         squared_dissimilarity=True,
                                         signal_output='signals',
                                         projected_atom_shape=1)
    elif args.mode == 'gmlvq':
        # get identity matrices for the matrix initialization as we do not
        # use the standard routine of anysma for the initialization
        matrix_init = np.repeat(np.expand_dims(np.eye(2), 0),
                                repeats=np.sum(protos_per_class), axis=0)
        matrix_init.astype(K.floatx())

        diss = OmegaDistance(linear_factor=None,
                             squared_dissimilarity=True,
                             signal_output='signals',
                             matrix_scope='local',
                             matrix_constraint='OmegaNormalization',
                             matrix_initializer=lambda x: matrix_init
                             )

    # define capsule network
    caps = Capsule(prototype_distribution=protos_per_class)
    caps.add(InputModule(signal_shape=(-1, np.prod(input_shape)),
                         trainable=False,
                         init_diss_initializer='zeros'))
    caps.add(diss)
    caps.add(SqueezeRouting())
    caps.add(NearestCompetition())

    output = caps(inputs)[1]

    # pre-train the model and overwrite the standard initialization matrix
    # for GMLVQ
    if args.mode == 'gmlvq':
        _, matrices = diss.get_weights()

    pre_train_model = Model(inputs=inputs, outputs=diss.input[0])
    diss_input = pre_train_model.predict(x_train, batch_size=batch_size)
    diss.pre_training(diss_input, y_train, capsule_inputs_are_equal=True)

    if args.mode == 'gmlvq':
        # set identity matrices
        centers, _ = diss.get_weights()
        diss.set_weights([centers, matrices])

    # define model and return
    model = Model(inputs, output)

    return model


def plot(X, Y, X_test, Y_test, model, fname, p, p_labels, tangents=None):
    # plot the results
    Y = np.argmax(Y, 1)
    Y_test = np.argmax(Y_test, 1)

    h = .02  # step size in the mesh (resolution of decision boundary)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 0] > Z[:, 1]

    # get colormap
    coolwarm = plt.cm.get_cmap('coolwarm', 2)
    newcolors = coolwarm(np.linspace(0, 1, 2))
    blue = np.array([31, 119, 180, 255]) / 255
    orange = np.array([255, 127, 14, 255]) / 255
    newcolors[0, :] = blue
    newcolors[-1, :] = orange
    newcmp = ListedColormap(newcolors)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=newcmp)

    # Plot the training points
    markers = ['o', '^']
    colors = ['#1f77b4', '#ff7f0e']
    for i in range(2):
        plt.scatter(X[Y == i, 0], X[Y == i, 1],
                    c=colors[i],
                    marker=markers[i],
                    edgecolors='w',
                    s=100)
        plt.scatter(X_test[Y_test == i, 0], X_test[Y_test == i, 1],
                    c=colors[i],
                    marker=markers[i],
                    edgecolors='k',
                    s=100)

    # Plot the prototypes
    plt.scatter(p[:, 0], p[:, 1],
                c=p_labels,
                cmap=newcmp,
                marker='*',
                edgecolors='k',
                s=1000,
                linewidths=3)
    if tangents is not None:
        plt.plot(tangents[:, 0], tangents[:, 1],
                 '.',
                 color='k',
                 markersize=6)

    legend_elements = [Line2D([0], [0],
                              marker='o',
                              color='k',
                              label='class 1',
                              markerfacecolor=None,
                              lw=0,
                              markersize=12),
                       Line2D([0], [0],
                              marker='^',
                              color='k',
                              label='class 2',
                              markerfacecolor=None,
                              lw=0,
                              markersize=12),
                       Line2D([0], [0],
                              marker='*',
                              color='k',
                              label='prototypes',
                              markerfacecolor=None,
                              lw=0 if tangents is None else 2,
                              markersize=12),
                       ]
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend(handles=legend_elements, fontsize=18, loc='upper left')
    plt.xticks(())
    plt.yticks(())
    plt.tight_layout()

    plt.savefig('./' + fname + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights",
                        help="Load h5 model trained weights.")
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--gpu', default=0, type=int,
                        help='Select GPU device.')
    parser.add_argument('--mode', default='glvq', type=str.lower,
                        help='Defines the LVQ model (GLVQ, GTLVQ, or GMLVQ).')
    parser.add_argument('--eval', action='store_true',
                        help='Only perform the evaluation.')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # set the number of prototypes
    if args.mode == 'glvq':
        protos_per_class = [10, 10]
    elif args.mode == 'gtlvq':
        protos_per_class = [6, 6]
    elif args.mode == 'gmlvq':
        protos_per_class = [6, 6]
    else:
        raise ValueError('Mode "{}" not defined.'.format(args.mode))

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

    train_model.compile(optimizer=Adam(lr=lr),
                        loss=GlvqLoss(),
                        metrics=[acc])

    # Callbacks
    csv_logger = callbacks.CSVLogger(args.save_dir + '/log.csv')
    lr_reduce = callbacks.ReduceLROnPlateau(factor=0.5, monitor='val_loss',
                                            mode='min', verbose=1, patience=5)

    # train the model
    if not args.eval:
        train_model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[csv_logger, lr_reduce],
                        validation_data=[x_test, y_test])

        train_model.save_weights(args.save_dir + '/trained_model.h5')

    print('training results:')
    result = train_model.evaluate(x_train, y_train, batch_size=batch_size)
    print('loss: {}   accuracy: {}'.format(result[0], result[1]))
    print('test results:')
    result = train_model.evaluate(x_test, y_test, batch_size=batch_size)
    print('loss: {}   accuracy: {}'.format(result[0], result[1]))

    # get prototypes
    if args.mode in 'glvq':
        protos = train_model.get_layer('minkowski_distance_1').get_weights()[0]
        labels = [0] * protos_per_class[0] + [1] * protos_per_class[1]
        tangents = None

    elif args.mode in 'gmlvq':
        protos = train_model.get_layer('omega_distance_1').get_weights()[0]
        labels = [0] * protos_per_class[0] + [1] * protos_per_class[1]
        tangents = None

    elif args.mode == 'gtlvq':
        space = 100
        t, B, c = train_model.get_layer(
            'restricted_tangent_distance_1').get_weights()

        c = np.squeeze(c, -1)
        theta = np.linspace(-c, c, space).transpose()
        theta = np.expand_dims(theta, 1)

        tangents = np.expand_dims(t, -1) + B * theta
        tangents = np.transpose(tangents, [0, 2, 1])
        tangents = np.reshape(tangents, (-1, 2))

        protos = t

        labels = [0] * protos_per_class[0] + \
                 [1] * protos_per_class[1]

    # plot results
    plot(x_train, y_train,
         x_test, y_test,
         train_model,
         args.save_dir + '/spiral_' + args.mode,
         protos, labels,
         tangents)
