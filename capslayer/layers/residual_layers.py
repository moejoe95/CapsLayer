from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import capslayer as cl
from config import cfg


def residualConvs(inputs, 
                conv_params,
                padding='VALID'):
    """
    Subnetwork with 6 convolutional layers + batch normalization + 2 residual connections.
    """

    # define parameters for residual shortcuts
    conv_shortcut_params = {
                "filters": conv_params['filters'],
                "kernel_size": -1,
                "strides": 1
        }

    if padding == 'VALID':
        shape = cl.shape(inputs)
        print('shape:', shape[1])
        conv_shortcut_params['kernel_size'] = shape[1] - (shape[1] - 2 * (conv_params['kernel_size'] - 1)) + 1
    else:
        conv_shortcut_params['kernel_size'] = conv_params['kernel_size']

    # conv1
    conv1 = tf.layers.conv2d(inputs,
                            **conv_params,
                            padding=padding,
                            name="res_conv1_layer")
    conv1_batched = tf.keras.layers.BatchNormalization(name='batch_norm_1')(conv1, training=cfg.is_training)
    conv1_act = tf.nn.relu(conv1_batched)

    # conv2
    conv2 = tf.layers.conv2d(conv1_act,
                            **conv_params,
                            padding=padding,
                            name="res_conv2_layer")
    conv2_batched = tf.keras.layers.BatchNormalization(name='batch_norm_2')(conv2, training=cfg.is_training)
    conv2_act = tf.nn.relu(conv2_batched)

    # conv3
    conv3 = tf.layers.conv2d(conv2_act,
                            **conv_params,
                            padding=padding,
                            name="res_conv3_layer")
    conv3_batched = tf.keras.layers.BatchNormalization(name='batch_norm_3')(conv3, training=cfg.is_training)
    
    # resize conv1 to size of conv3 by convolution if padding is VALID
    if padding == 'VALID':
        conv1_batched = tf.layers.conv2d(conv1_batched,
                                **conv_shortcut_params,
                                padding='VALID',
                                name="res_shortcut1_layer")

    # first residual connection
    conv3 = tf.keras.layers.Add()([conv1_batched, conv3_batched])
    conv3 = tf.nn.relu(conv3)

    # conv4
    conv4 = tf.layers.conv2d(conv3,
                            **conv_params,
                            padding=padding,
                            name="res_conv4_layer")
    conv4_batched = tf.keras.layers.BatchNormalization(name='batch_norm_4')(conv4, training=cfg.is_training)
    conv4_act = tf.nn.relu(conv4_batched)

    # conv5
    conv5 = tf.layers.conv2d(conv4_act,
                            **conv_params,
                            padding=padding,
                            name="res_conv5_layer")
    conv5_batched = tf.keras.layers.BatchNormalization(name='batch_norm_5')(conv5, training=cfg.is_training)
    conv5_act = tf.nn.relu(conv5_batched)

    # conv6
    conv6 = tf.layers.conv2d(conv5_act,
                            **conv_params,
                            padding='VALID',
                            name="res_conv6_layer")
    conv6_batched = tf.keras.layers.BatchNormalization(name='batch_norm_6')(conv6, training=cfg.is_training)

    # resize conv4 to size of conv6 by convolution
    conv4 = tf.layers.conv2d(conv4_batched,
                            **conv_shortcut_params,
                            padding='VALID',
                            name="res_shortcut2_layer")

    # second residal connection
    conv6 = tf.keras.layers.Add()([conv4, conv6_batched])
    return tf.nn.relu(conv6)         


def getParamsSkip(skip_params, previous_params):
    return {
        "filters": previous_params['filters'],
        "kernel_size": skip_params['kernel_size'] - previous_params['kernel_size'] + 1,
        "strides": 1,
        "out_caps_dims": previous_params['out_caps_dims'],
        "num_iter": previous_params['num_iter'],
        "routing_method": previous_params['routing_method']
    }


def capsResidual(res_pose, res_activation, previous_pose, previous_activation, skip_params):
    """
    Adds skip connection to output of previous caps layer and performs relu activation.
    """
    skip_pose, skip_activation, _ = cl.layers.conv2d(res_pose,
                                        res_activation,
                                        **skip_params,
                                        name="ConvCaps_layer_skip")

    pose_add = tf.keras.layers.Add()([previous_pose, skip_pose])
    pose_add = tf.nn.relu(pose_add)
    activation_add = tf.keras.layers.Add()([previous_activation, skip_activation])
    activation_add = tf.nn.relu(activation_add)

    return pose_add, activation_add
        