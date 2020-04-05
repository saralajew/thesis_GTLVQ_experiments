# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.regularizers import *

from .utils.linalg_funcs import det


class LogDeterminant(Regularizer):
    # computes ln(det(A)) or ln(det(A'*A))
    # A is assumed as transposed matrix if it is a omega matrix. Thus we symmetrize over the subspace.
    def __init__(self, alpha=0.005):
        self.alpha = alpha

    def __call__(self, w):
        with K.name_scope('log_determinant'):
            return -self.alpha * K.sum(K.log(det(w, keepdims=False)))

    def get_config(self):
        return {'alpha': self.alpha}


class OmegaRegularizer(Regularizer):
    def __init__(self, alpha=0.005):
        self.alpha = alpha

    def __call__(self, w):
        with K.name_scope('omega_regularizer'):
            shape = K.int_shape(w)
            ndim = K.ndim(w)
            # find matrix minimal dimension
            if shape[-2] >= shape[-1]:
                axes = ndim - 2
            # this is needed to regularize if the points are projected to a higher dimensional space
            else:
                axes = ndim - 1

            # batch matrices
            if ndim >= 3:
                w = K.batch_dot(w, w, [axes, axes])
            # non-batch
            else:
                if axes == 1:
                    w = K.dot(w, K.transpose(w))
                else:
                    w = K.dot(K.transpose(w), w)

            log_determinant = LogDeterminant(alpha=self.alpha)
            return log_determinant(w)

    def get_config(self):
        return {'alpha': self.alpha}


class MinValue(Regularizer):
    def __init__(self, alpha=0.001, axis=-1):
        self.alpha = alpha
        self.axis = axis

    def __call__(self, w):
        with K.name_scope('min_value'):
            return self.alpha * K.mean(K.min(w, self.axis))

    def get_config(self):
        return {'alpha': self.alpha,
                'axis': self.axis}


class MaxValue(Regularizer):
    def __init__(self, alpha=0.0001, axis=-1):
        self.alpha = alpha
        self.axis = axis

    def __call__(self, w):
        with K.name_scope('max_value'):
            return self.alpha * K.mean(K.max(w, self.axis))

    def get_config(self):
        return {'alpha': self.alpha,
                'axis': self.axis}


# Aliases (always calling the standard setting):


def log_determinant(w):
    regularizer = LogDeterminant()
    return regularizer(w)


def omega_regularizers(w):
    regularizer = OmegaRegularizer()
    return regularizer(w)


def min_value(w):
    regularizer = MinValue()
    return regularizer(w)


def max_value(w):
    regularizer = MaxValue()
    return regularizer(w)


# copied and modified from Keras!
def serialize(regularizer):
    return serialize_keras_object(regularizer)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='regularizer')


def get(identifier):
    if identifier is None:
        return None
    elif isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        identifier = str(identifier)

        if identifier.islower():
            return deserialize(identifier)
        else:
            config = {'class_name': identifier, 'config': {}}
            return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret regularizer identifier: ' + str(identifier))
