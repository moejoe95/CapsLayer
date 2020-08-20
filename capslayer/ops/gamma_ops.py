import tensorflow as tf
import numpy as np


def t_score(c_ij):
    """ 
        See [Peer et al., 2018](https://arxiv.org/pdf/1812.09707.pdf)

        Calculates the T-score to measure whether capsules are 
        coupled in a tree structure (~1) or not (~0) [1]

        param: c_ij - 7D or 5D Tensor representing coupling coefficient
    """
    out_caps = tf.cast(tf.shape(c_ij)[1], tf.float32)
    c_ij = tf.squeeze(c_ij, axis=-1)         # (batch_size, k, k, out_caps, in_caps, 1) or (batch_size, out_caps, in_caps, 1)
    c_ij = tf.squeeze(c_ij, axis=-1)         # (batch_size, k, k, out_caps, in_caps) or (batch_size, out_caps, in_caps)

    t_shape = [0, 1, 2, 4, 3]                # if conv. caps layer
    if len(c_ij.shape) == 3:                 # if fc caps layer
        t_shape = [0, 2, 1]
    c_ij = tf.transpose(c_ij, t_shape)

    epsilon = 1e-12
    entropy = -tf.reduce_sum(c_ij * tf.math.log(c_ij + epsilon), axis=-1)
    T = 1 - entropy / -tf.math.log(1 / out_caps)
    return tf.reduce_mean(T)


def d_score(v_j):
    """ 
        See [Peer et al., 2018](https://arxiv.org/pdf/1812.09707.pdf)

        Measures how the activation of capsules adapts to the input.

        param: v_j - activations of capsules with shape (batch_size, num_capsules, dim)
    """
    v_j_norm = tf.norm(v_j, axis=-1)
    v_j_std = tf.math.reduce_std(v_j_norm, axis=0)   # Note: Calc std along the batch dimension
    return tf.reduce_max(v_j_std)    
