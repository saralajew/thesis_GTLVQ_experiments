# -*- coding: utf-8 -*-
""" This is the script to generate a Spiral dataset. Credits to
https://github.com/hyounesy/TFPlaygroundPSA/blob/master/src/dataset.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import os

import matplotlib
matplotlib.use('Agg')  # needed to avoid cloud errors
import matplotlib.pyplot as plt


def data_spiral(num_samples, noise):
    """
    Generates the spiral dataset with the given number of samples and noise
    :param num_samples: total number of samples
    :param noise: noise percentage (0 .. 50)
    :return: None

    https://github.com/hyounesy/TFPlaygroundPSA/blob/master/src/dataset.py
    """
    noise *= 0.01
    half = num_samples // 2
    points = np.zeros([num_samples, 2])
    labels = np.zeros(num_samples).astype(int)
    for j in range(num_samples):
        i = j % half
        label = 1
        delta = 0
        if j >= half:  # negative examples
            label = 0
            delta = np.pi
        r = i / half * 5
        t = 1.75 * i / half * 2 * np.pi + delta
        x = r * np.sin(t) + random.uniform(-1, 1) * noise
        y = r * np.cos(t) + random.uniform(-1, 1) * noise
        labels[j] = label
        points[j] = (x, y)
    return points, labels


if __name__ == '__main__':
    if os.path.exists('./data'):
        print('Files available.')
        points = np.load('./data/points.npy')
        labels = np.load('./data/labels.npy')
    else:
        os.mkdir('./data')
        points, labels = data_spiral(1000, 25)

        np.save('./data/points.npy', points)
        np.save('./data/labels.npy', labels)

    plt.scatter(points[labels == 0, 0], points[labels == 0, 1])
    plt.scatter(points[labels == 1, 0], points[labels == 1, 1])
    plt.savefig('./plot.png')
