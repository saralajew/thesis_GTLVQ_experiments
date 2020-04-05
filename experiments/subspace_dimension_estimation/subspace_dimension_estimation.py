# -*- coding: utf-8 -*-
""" This is the script to reproduce the results of the subspace
dimension estimation experiment of GTLVQ performed in Section 3.4.2.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers
from keras import Input
from keras.datasets import mnist
from keras.metrics import categorical_accuracy

from anysma import Capsule
from anysma.modules import InputModule
from anysma.modules.routing import SqueezeRouting
from anysma.modules.measuring import TangentDistance
from anysma.modules.competition import NearestCompetition

from anysma.losses import GlvqLossOverDissimilarities as GlvqLoss

import os
import numpy as np
import csv

import matplotlib
matplotlib.use('Agg')  # needed to avoid cloud errors
from matplotlib import pyplot as plt

# set Cuda visible device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)


# accuracy metric for GLVQ methods (sign has to be switched since the
# minimal value determines the winner)
def acc(y_true, y_pred):
    return categorical_accuracy(y_true, -y_pred)


# set parameters
batch_size = 128
max_number_tangents = 1
save_dir_base = './output'
number_runs = 3

if not os.path.exists(save_dir_base):
    os.mkdir(save_dir_base)

# get dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train.astype('float32') / 255., -1)
x_test = np.expand_dims(x_test.astype('float32') / 255., -1)
y_train = to_categorical(y_train.astype('float32'))
y_test = to_categorical(y_test.astype('float32'))

# define input node
inputs = Input(x_train.shape[1:])

# iterate over runs
for j in range(number_runs):
    # create run directory
    save_dir = save_dir_base + '/run_' + str(j)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # iterate over number of tangents
    results = []
    for i in range(max_number_tangents + 1):
        number_tangents = i

        # define define GTLVQ network as capsule
        diss = TangentDistance(projected_atom_shape=number_tangents,
                               linear_factor=None,
                               squared_dissimilarity=True,
                               signal_output='signals')

        caps = Capsule(prototype_distribution=(1, 10))
        caps.add(InputModule(signal_shape=(1,) + x_train.shape[1:4],
                             trainable=False,
                             init_diss_initializer='zeros'))
        caps.add(diss)
        caps.add(SqueezeRouting())
        caps.add(NearestCompetition(use_for_loop=False))

        outputs = caps(inputs)[1]

        # pre-train the model over 10000 random digits
        idx = np.random.randint(0, len(x_train) - 1,
                                (min(10000, len(x_train)),))
        pre_train_model = Model(inputs=inputs, outputs=diss.input[0])
        diss_input = pre_train_model.predict(x_train[idx, :],
                                             batch_size=batch_size)
        diss.pre_training(diss_input, y_train[idx],
                          capsule_inputs_are_equal=True)

        # define model
        model = Model(inputs, outputs)

        # compile the model in order to use evaluate
        model.compile(loss=GlvqLoss(),
                      optimizer=optimizers.Adam(),
                      metrics=[acc])

        model.summary()
        model.save_weights(os.path.join(save_dir, 'trained_model_num_tangents_'
                                        + str(i) + '.h5'))

        evaluation_train = model.evaluate(x_train, y_train,
                                          batch_size=batch_size)
        evaluation_test = model.evaluate(x_test, y_test,
                                         batch_size=batch_size)
        evaluation = evaluation_train + evaluation_test
        results.append(evaluation)
        print('num_tangents ' + str(i))
        print(evaluation)

    # write results
    with open(save_dir + '/results.csv', 'w+') as f:
        f.write('num_tangents,train_loss,training_acc,val_loss,val_acc\n')
        for i, result in enumerate(results):
            f.write(str(i) + ',' + str(result[0]) + ',' + str(result[1])
                    + ',' + str(result[2]) + ',' + str(result[3]) + '\n')

# read results of all the runs
results = np.zeros((number_runs, max_number_tangents + 1, 5))
for run in range(number_runs):
    with open(save_dir_base + '/run_' + str(run) + '/results.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV, None)  # skip header
        for i, row in enumerate(readCSV):
            results[run, i] = [float(row[0]), float(row[1]),
                               float(row[2]), float(row[3]), float(row[4])]

# calculate mean and std
mean = np.mean(results, 0)
std = np.std(results, 0)

steps = results[0, :, 0]

# plot it!
fig, ax = plt.subplots(1)
ax.plot(steps, mean[:, 2], label='training')
ax.plot(steps, mean[:, 4], label='test')
plt.axvline(x=12, color='r', linestyle='--', label='selected number')
ax.fill_between(steps, mean[:, 2] + std[:, 2], mean[:, 2] - std[:, 2],
                alpha=0.5)
ax.fill_between(steps, mean[:, 4] + std[:, 4], mean[:, 4] - std[:, 4],
                alpha=0.5)
ax.grid()
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
ax.set_xlabel('subspace dimension', fontsize=17)
ax.set_ylabel('accuracy', fontsize=17)
plt.legend(fontsize=17)
plt.tight_layout()
plt.savefig(save_dir_base + '/initialization_plot.png')
