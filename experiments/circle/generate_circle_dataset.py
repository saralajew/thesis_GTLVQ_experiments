# -*- coding: utf-8 -*-
""" This is the script to generate a Circle dataset. Credits to
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


def data_circle(num_samples, noise):
    """
    Generates the two circles dataset with the given number of samples and noise
    :param num_samples: total number of samples
    :param noise: noise percentage (0 .. 50)
    :return: None

    https://github.com/hyounesy/TFPlaygroundPSA/blob/master/src/dataset.py
    """
    radius = 5

    def get_circle_label(x, y, xc, yc):
        return 1 if np.sqrt((x - xc) ** 2 + (y - yc) ** 2) < (
                    radius * 0.5) else 0

    noise *= 0.01
    points = np.zeros([num_samples, 2])
    labels = np.zeros(num_samples).astype(int)
    # Generate positive points inside the circle.
    for i in range(num_samples // 2):
        r = random.uniform(0, radius * 0.5)
        angle = random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noise_x = random.uniform(-radius, radius) * noise
        noise_y = random.uniform(-radius, radius) * noise
        labels[i] = get_circle_label(x + noise_x, y + noise_y, 0, 0)
        points[i] = (x, y)
    # Generate negative points outside the circle.
    for i in range(num_samples // 2, num_samples):
        r = random.uniform(radius * 0.7, radius)
        angle = random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noise_x = random.uniform(-radius, radius) * noise
        noise_y = random.uniform(-radius, radius) * noise
        labels[i] = get_circle_label(x + noise_x, y + noise_y, 0, 0)
        points[i] = (x, y)

    return points, labels


if __name__ == '__main__':

    if os.path.exists('./data'):
        print('Files available.')
        points = np.load('./data/points.npy')
        labels = np.load('./data/labels.npy')
    else:
        os.mkdir('./data')
        points, labels = data_circle(1000, 10)

        np.save('./data/points.npy', points)
        np.save('./data/labels.npy', labels)

    plt.scatter(points[labels == 0, 0], points[labels == 0, 1])
    plt.scatter(points[labels == 1, 0], points[labels == 1, 1])
    plt.savefig('./plot.png')
