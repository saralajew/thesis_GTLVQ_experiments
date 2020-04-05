# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import six
from keras import backend as K
from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object


# each function should exist as camel-case (class) and snake-case (function). It's important that
# a class is always camel-cass and a function is snake-case. Otherwise deserialization fails.
class ProbabilityTransformation(object):
    """ProbabilityTransformation base class: all prob_trans inherit from this class.
    """
    # all functions must have the parameter normalization to use it automatically for regression
    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, tensor):
        raise NotImplementedError

    def get_config(self):
        return {'axis': self.axis}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Softmax(ProbabilityTransformation):
    def __init__(self,
                 axis=-1,
                 max_stabilization=True):
        self.max_stabilization = max_stabilization

        super(Softmax, self).__init__(axis=axis)

    def __call__(self, tensors):
        with K.name_scope('softmax'):
            if self.max_stabilization:
                tensors = tensors - K.max(tensors, axis=self.axis, keepdims=True)

            return K.softmax(tensors, axis=self.axis)

    def get_config(self):
        config = {'max_stabilization': self.max_stabilization}
        super_config = super(Softmax, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class NegSoftmax(Softmax):
    def __call__(self, tensors):
        with K.name_scope('neg_softmax'):
            tensors = -tensors
            return super(NegSoftmax, self).__call__(tensors)


class NegExp(ProbabilityTransformation):
    def __init__(self,
                 axis=-1,
                 max_normalization=False):
        self.max_normalization = max_normalization

        super(NegExp, self).__init__(axis=axis)

    def __call__(self, tensors):
        with K.name_scope('neg_exp'):
            tensors = -tensors
            if self.max_normalization:
                tensors = tensors - K.max(tensors, axis=self.axis, keepdims=True)

            return K.exp(tensors)

    def get_config(self):
        config = {'max_normalization': self.max_normalization}
        super_config = super(NegExp, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class Flipmax(ProbabilityTransformation):
    def __init__(self,
                 axis=-1,
                 epsilon=K.epsilon()):
        self.epsilon = epsilon
        super(Flipmax, self).__init__(axis=axis)

    def __call__(self, tensors):
        # works only if all numbers are positive and small number means high probability
        # flip the order of the elements
        # vectors = [3, 1, 2, 4] --> [2, 4, 3, 1] --> [0.2, 0.4, 0.3, 0.1]
        with K.name_scope('flipmax'):
            tensors = K.max(tensors, axis=self.axis, keepdims=True) - tensors + \
                      K.min(tensors, axis=self.axis, keepdims=True)

            return tensors / K.maximum(K.sum(tensors, axis=self.axis, keepdims=True), self.epsilon)

    def get_config(self):
        config = {'epsilon': self.epsilon}
        super_config = super(Flipmax, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class Flip(ProbabilityTransformation):
    def __init__(self,
                 axis=-1,
                 lower_bound=K.epsilon()):
        if lower_bound <= 0:
            raise ValueError("The lower bound must be greater than zero.")
        self.lower_bound = lower_bound
        super(Flip, self).__init__(axis=axis)

    def __call__(self, tensors):
        # works only if all numbers are positive and small number means high probability
        # flip the order of the elements
        # vectors = [3, 1, 2, 4] --> [2, 4, 3, 1] --> [0.5, 1, 0.75, 0.25]
        with K.name_scope('flip'):
            max_v = K.max(tensors, axis=self.axis, keepdims=True) + self.lower_bound
            tensors = max_v - tensors + K.min(tensors, axis=self.axis, keepdims=True)

            return tensors / max_v

    def get_config(self):
        config = {'lower_bound': self.lower_bound}
        super_config = super(Flip, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class MarginFlip(ProbabilityTransformation):
    def __init__(self,
                 axis=-1,
                 margin=0.0001,
                 lower_bound=K.epsilon()):
        self.margin = margin
        if lower_bound <= 0:
            raise ValueError("The lower bound must be greater than zero.")
        self.lower_bound = lower_bound
        super(MarginFlip, self).__init__(axis=axis)

    def __call__(self, tensors):
        with K.name_scope('margin_flip'):
            max_v = K.max(tensors, axis=self.axis, keepdims=True) + self.lower_bound
            tensors = K.minimum(max_v - tensors + self.margin, max_v)

            return tensors / max_v

    def get_config(self):
        config = {'lower_bound': self.lower_bound,
                  'margin': self.margin}
        super_config = super(MarginFlip, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


class MarginLogFlip(ProbabilityTransformation):
    def __init__(self,
                 axis=-1,
                 margin=0.0001,
                 lower_bound=K.epsilon()):
        self.margin = margin
        if lower_bound <= 0:
            raise ValueError("The lower bound must be greater than zero.")
        self.lower_bound = lower_bound
        super(MarginLogFlip, self).__init__(axis=axis)

    def __call__(self, tensors):
        # works only if all numbers are positive and small number means high probability
        # flip the order of the elements
        # vectors = [3, 1, 2, 4] --> [2, 4, 3, 1]
        with K.name_scope('margin_log_flip'):
            tensors = K.log(tensors + 1. + self.lower_bound)
            max_v = K.max(tensors, axis=self.axis, keepdims=True)
            tensors = K.minimum(max_v - tensors + self.margin, max_v)

            return tensors / max_v

    def get_config(self):
        config = {'lower_bound': self.lower_bound,
                  'margin': self.margin}
        super_config = super(MarginLogFlip, self).get_config()
        return dict(list(super_config.items()) + list(config.items()))


# Aliases (always calling the standard setting):


def softmax(tensors):
    prob_trans = Softmax()
    return prob_trans(tensors)


def neg_softmax(tensors):
    prob_trans = NegSoftmax()
    return prob_trans(tensors)


def neg_exp(tensors):
    prob_trans = NegExp()
    return prob_trans(tensors)


def flipmax(tensors):
    prob_trans = Flipmax()
    return prob_trans(tensors)


def flip(tensors):
    prob_trans = Flip()
    return prob_trans(tensors)


def margin_flip(tensors):
    prob_trans = MarginFlip()
    return prob_trans(tensors)


def margin_log_flip(tensors):
    prob_trans = MarginLogFlip()
    return prob_trans(tensors)


# copied and modified from Keras!
def serialize(probability_transformation):
    return serialize_keras_object(probability_transformation)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='probability transformation')


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
        raise ValueError('Could not interpret probability transformation identifier: ' + str(identifier))
