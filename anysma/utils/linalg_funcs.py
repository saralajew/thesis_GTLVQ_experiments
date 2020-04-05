# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras import backend as K
from .caps_utils import mixed_shape, equal_int_shape

import numpy as np


def p_norm(vectors, order_p=2, axis=-1, squared=False, keepdims=False, epsilon=K.epsilon()):
    with K.name_scope('p_norm'):
        # special case 1
        if order_p == 1:
            diss = K.sum(K.abs(vectors), axis=axis, keepdims=keepdims)
        elif order_p == np.inf:
            diss = K.max(K.abs(vectors), axis=axis, keepdims=keepdims)
            if squared:
                raise NotImplementedError("The squared computation of the maximums-norm is not defined.")
        # Euclidean distance
        elif order_p == 2:
            diss = K.sum(K.square(vectors), axis=axis, keepdims=keepdims)
            if not squared:
                if epsilon == 0:
                    diss = K.sqrt(diss)
                else:
                    diss = K.sqrt(K.maximum(diss, epsilon))
        elif order_p % 2 == 0 and order_p > 0:
            diss = K.sum(K.pow(vectors, order_p), axis=axis, keepdims=keepdims)
            if not squared:
                if epsilon == 0:
                    diss = K.pow(diss, 1 / order_p)
                else:
                    diss = K.pow(K.maximum(epsilon, diss), 1 / order_p)
        elif order_p > 1:
            diss = K.sum(K.pow(K.abs(vectors), order_p), axis=axis, keepdims=keepdims)
            if not squared:
                if epsilon == 0:
                    diss = K.pow(diss, 1 / order_p)
                else:
                    diss = K.pow(K.maximum(epsilon, diss), 1 / order_p)
        else:  # < 1
            raise NotImplementedError("The computation of the p-norm with p<1 could be instable. "
                                      "Therefore, we don't support it so far.")

        return diss


def euclidean_norm(vectors, axis=-1, squared=False, keepdims=False, epsilon=K.epsilon()):
    return p_norm(vectors, order_p=2, axis=axis, squared=squared, keepdims=keepdims, epsilon=epsilon)


def euclidean_distance(x, y, squared=False, epsilon=K.epsilon()):
    # last dimension must be the vector dimension!
    # compute the distance via the identity of the dot product. This avoids the memory overhead due to the subtraction!
    #
    # x.shape = (number_of_x_vectors, vector_dim)
    # y.shape = (number_of_y_vectors, vector_dim)
    #
    # output: matrix of distances (number_of_x_vectors, number_of_y_vectors)
    for tensor in [x, y]:
        if K.ndim(tensor) != 2:
            raise ValueError('The tensor dimension must be two. You provide: K.ndim(tensor)=' + str(K.ndim(tensor))
                             + '.')
    if not equal_int_shape([K.int_shape(x)[1]], [K.int_shape(y)[1]]):
        raise ValueError('The vector shape must be equivalent in both tensors. You provide: K.int_shape(x)[1]='
                         + str(K.int_shape(x)[1]) + ' and  K.int_shape(y)[1]=' + str(K.int_shape(y)[1]) + '.')

    with K.name_scope('euclidean_distance'):
        y = K.transpose(y)

        diss = K.sum(K.square(x), axis=1, keepdims=True) - 2 * K.dot(x, y) + K.sum(K.square(y), axis=0, keepdims=True)

        if not squared:
            if epsilon == 0:
                diss = K.sqrt(diss)
            else:
                diss = K.sqrt(K.maximum(diss, epsilon))

        return diss


def svd(tensors, full_matrices=False, compute_uv=True):
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        # return s, u, v
        return tf.svd(tensors, full_matrices=full_matrices, compute_uv=compute_uv)

    elif K.backend() == 'cntk':
        raise NotImplementedError("SVD is not implemented for CNTK")

    elif K.backend() == 'theano':
        raise NotImplementedError("SVD is not implemented for Theano")
    else:
        raise NotImplementedError("Unknown backend `" + K.backend() + "`.")


def det(tensors, keepdims=False):
    if K.backend() == 'tensorflow':
        import tensorflow as tf

        d = tf.linalg.det(tensors)
        if keepdims:
            d = K.expand_dims(K.expand_dims(d, -1), -1)

        return d

    elif K.backend() == 'cntk':
        raise NotImplementedError("determinant is not implemented for CNTK")

    elif K.backend() == 'theano':
        raise NotImplementedError("determinant is not implemented for CNTK")

    else:
        raise NotImplementedError("Unknown backend `" + K.backend() + "`.")


def trace(tensors, keepdims=False):
    # trace of a squared matrix
    with K.name_scope('trace'):
        shape = mixed_shape(tensors)
        int_shape = K.int_shape(tensors)

        if not equal_int_shape([int_shape[-1]], [int_shape[-2]]):
            raise ValueError("The matrix dimension (the two last dimensions) of the tensor must be squared. "
                             "You provide: " + str(int_shape[-2:]) + ".")
        if int_shape[-1] is None and int_shape[-2] is None:
            raise ValueError('At least one dimension of the matrix must be defined. You provide: ' + str(int_shape))

        # K.eye() doesn't accept placeholders. Thus, one dim must be specified.
        if int_shape[-1] is None:
            matrix_dim = shape[-2]
        else:
            matrix_dim = shape[-1]

        t = K.sum(tensors * K.eye(matrix_dim), axis=[-1, -2])
        if keepdims:
            t = K.expand_dims(K.expand_dims(t, -1), -1)

        return t


# Aliases:
norm = euclidean_norm
