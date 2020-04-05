# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.losses import *


# each function should exist as camel-case (class) and snake-case (function). It's important that
# a class is always camel-cass and a function is snake-case. Otherwise deserialization fails.
# axis is always assumed as -1
class Loss(object):
    """Loss base class.
    """
    def __call__(self, y_true, y_pred):
        raise NotImplementedError

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MarginLoss(Loss):
    def __init__(self, lamb=0.5, margin=0.1):
        self.lamb = lamb
        self.margin = margin

    def __call__(self, y_true, y_pred):
        with K.name_scope('margin_loss'):
            loss = y_true * K.square(K.relu(1 - self.margin - y_pred))\
                + self.lamb * (1 - y_true) * K.square(K.relu(y_pred - self.margin))
            return K.sum(loss, axis=-1)

    def get_config(self):
        return {'lamb': self.lamb,
                'margin': self.margin}


class GlvqLoss(Loss):
    # add flip as prob_trans for real GLVQ cost function
    def __init__(self, squash_func=None):
        self.squash_func = squash_func

    def __call__(self, y_true, y_pred):
        with K.name_scope('glvq_loss'):
            # y_true categorical vector (one-hot value)
            dp = K.sum(y_true * y_pred, axis=-1)

            # dp = K.squeeze(K.batch_dot(y_true, y_pred, [1, 1]), -1)

            dm = K.max((1 - y_true) * y_pred + y_true * K.min(y_pred, axis=-1, keepdims=True), axis=-1)

            # mu function
            # scale back by unflipping
            loss = (-dp + dm) / \
                   (2*K.max(y_pred, axis=-1) - dm - dp + 2*K.min(y_pred, axis=-1))

            if self.squash_func is not None:
                loss = self.squash_func(loss)

            return loss

    def get_config(self):
        if self.squash_func is None:
            return {'squash_func': self.squash_func}
        else:
            raise ValueError("Not serializable if `squash_func` is not None.")


class GlvqLossOverDissimilarities(object):
    def __init__(self, squash_func=None):
        self.squash_func = squash_func

    def __call__(self, y_true, y_pred):
        with K.name_scope('glvq_loss'):
            dp = K.sum(y_true * y_pred, axis=-1)

            dm = K.min((1 - y_true) * y_pred + y_true * 1 / K.epsilon(),
                       axis=-1)

            loss = (dp - dm) / (dp + dm)

            if self.squash_func is not None:
                loss = self.squash_func(loss)

            return loss

    def get_config(self):
        if self.squash_func is None:
            return {'squash_func': self.squash_func}
        else:
            raise ValueError("Not serializable if `squash_func` is not None.")


class GeneralizedKullbackLeiblerDivergence(Loss):
    def __call__(self, y_true, y_pred):
        with K.name_scope('generalized_kullback_leibler_divergence'):
            y_true = K.maximum(y_true, K.epsilon())
            y_pred = K.maximum(y_pred, K.epsilon())
            return K.sum(y_true * K.log(y_true / y_pred) - y_true + y_pred, axis=-1)


class ItakuraSaitoDivergence(Loss):
    def __call__(self, y_true, y_pred):
        with K.name_scope('itakura_saito_divergence'):
            y_true = K.maximum(y_true, K.epsilon())
            y_pred = K.maximum(y_pred, K.epsilon())
            return K.sum(y_true / y_pred - K.log(y_true / y_pred) - 1, axis=-1)


class SpreadLoss(Loss):
    def __init__(self, margin=0.2):
        self.margin = margin

    def __call__(self, y_true, y_pred):
        with K.name_scope('spread_loss'):
            # mask for the true label
            true_mask = K.cast(K.equal(y_true, 1), dtype=y_pred.dtype)

            # extract correct prediction
            true = K.sum(y_pred * true_mask, axis=-1, keepdims=True)

            # mask the correct class out of the loss vector
            loss = (1 - true_mask) * K.square((K.maximum(0., self.margin - (true - y_pred))))

            return K.sum(loss, axis=-1)

    def get_config(self):
        return {'margin': self.margin}


# Aliases (always calling the standard setting):


def margin_loss(y_true, y_pred):
    loss_func = MarginLoss()
    return loss_func(y_true, y_pred)


def glvq_loss(y_true, y_pred):
    loss_func = GlvqLoss()
    return loss_func(y_true, y_pred)


def generalized_kullback_leibler_divergence(y_true, y_pred):
    loss_func = GeneralizedKullbackLeiblerDivergence()
    return loss_func(y_true, y_pred)


def itakura_saito_divergence(y_true, y_pred):
    loss_func = ItakuraSaitoDivergence()
    return loss_func(y_true, y_pred)


def spread_loss(y_true, y_pred):
    loss_func = SpreadLoss()
    return loss_func(y_true, y_pred)


# copied and modified from Keras!
def serialize(loss):
    return serialize_keras_object(loss)


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')


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
        raise ValueError('Could not interpret loss function identifier:', str(identifier))
