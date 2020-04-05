# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras import backend as K

from .linalg_funcs import svd, trace
from .caps_utils import mixed_shape


def dynamic_routing_squash(vectors, axis=-1, epsilon=K.epsilon()):
    with K.name_scope('squash'):
        squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
        if epsilon == 0:
            return K.sqrt(squared_norm) / (1 + squared_norm) * vectors
        else:
            return K.sqrt(K.maximum(squared_norm, epsilon)) / (1 + squared_norm) * vectors


def orthogonalization(tensors):
    # orthogonalization via polar decomposition
    with K.name_scope('orthogonalization'):
        _, u, v = svd(tensors, full_matrices=False, compute_uv=True)
        u_shape = mixed_shape(u)
        v_shape = mixed_shape(v)

        # reshape to (num x N x M)
        u = K.reshape(u, (-1, u_shape[-2], u_shape[-1]))
        v = K.reshape(v, (-1, v_shape[-2], v_shape[-1]))

        out = K.batch_dot(u, K.permute_dimensions(v, [0, 2, 1]))

        out = K.reshape(out, u_shape[:-1] + (v_shape[-2],))

        return out


def trace_normalization(tensors, epsilon=K.epsilon()):
    with K.name_scope('trace_normalization'):
        constant = trace(tensors, keepdims=True)

        if epsilon != 0:
            constant = K.maximum(constant, epsilon)

        return tensors / constant


def omega_normalization(tensors, epsilon=K.epsilon()):
    with K.name_scope('omega_normalization'):
        ndim = K.ndim(tensors)

        # batch matrices
        if ndim >= 3:
            axes = ndim - 1
            s_tensors = K.batch_dot(tensors, tensors, [axes, axes])
        # non-batch
        else:
            s_tensors = K.dot(tensors, K.transpose(tensors))

        t = trace(s_tensors, keepdims=True)
        if epsilon == 0:
            constant = K.sqrt(t)
        else:
            constant = K.sqrt(K.maximum(t, epsilon))

        return tensors / constant
