# Copyright 2018 The CapsLayer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import capslayer as cl
import tensorflow as tf
import os

from config import cfg


class CapsNet(object):
    def __init__(self, height=28, width=28, channels=1, num_label=10):
        '''

        Args:
            height: Integer, the height of inputs.
            width: Integer, the width of inputs.
            channels: Integer, the channels of inputs.
            num_label: Integer, the category number.
        '''
        self.height = height
        self.width = width
        self.channels = channels
        self.num_label = num_label


    def create_network(self, inputs, labels, attention=True, decoder='FC', routing_method='SDARouting', vec_shape=[8, 1]):
        """ Setup a convolutional vector capsule network.

        Args:
            inputs: Tensor or array with shape [batch_size, height, width, channels] or [batch_size, height * width * channels].
            labels: Tensor or array with shape [batch_size].
            attention: enable attention mechanism.
            decoder: which reconstruction network to use, either 'FC' or 'DECONV'.

        Returns:
            poses: [batch_size, num_label, 16, 1].
            probs: Tensor with shape [batch_size, num_label], the probability of entity presence.
        """

        # set model parameters

        self.routing_method = routing_method
        self.vec_shape = vec_shape
        self.decoder = decoder
        self.attention = attention
        self.raw_imgs = inputs
        self.labels = labels

        self.conv1_params = {
            "filters": 256,
            "kernel_size": 9,
            "strides": 1
        }

        self.prim_caps_params = {
            "filters": 32,
            "kernel_size": 9,
            "strides": 2,
            "out_caps_dims": self.vec_shape
        }

        self.conv_caps_params = {
            "filters": 32,
            "kernel_size": 5,
            "strides": 1,
            "out_caps_dims": self.vec_shape,
            "num_iter": 3,
            "routing_method": self.routing_method
        }

        self.fc_caps_params = {
            "num_outputs": self.num_label,
            "out_caps_dims": [16, 1],
            "routing_method": self.routing_method
        }

        inputs = tf.reshape(self.raw_imgs, shape=[-1, self.height, self.width, self.channels])

        # first convolutional layer
        conv1 = tf.layers.conv2d(inputs,
                                 **self.conv1_params,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 name="Conv1_layer")

        # idea of attention in CapsNet is from Hoogi et al. 
        # in the paper 'Self-Attention Capsule Networks for Image Classification'
        if attention:
            conv1 = cl.layers.selfAttention(conv1, 256)

        # primary caps layer
        pose_conv, activation_conv = cl.layers.primaryCaps(conv1,
                                                **self.prim_caps_params,
                                                method="norm",
                                                name="PrimaryCaps_layer")

        # 1st convolutional capsule layer
        pose_conv, activation_conv = cl.layers.conv2d(pose_conv,
                                                activation_conv,
                                                **self.conv_caps_params,
                                                name="ConvCaps_layer1")

        # fully connected capsule layer
        with tf.variable_scope('FullyConnCaps_layer'):
            num_inputs = np.prod(cl.shape(pose_conv)[1:4])
            pose_conv = tf.reshape(pose_conv, shape=[-1, num_inputs, self.vec_shape[0], self.vec_shape[1]])
            activation_conv = tf.reshape(activation_conv, shape=[-1, num_inputs])

            self.poses, self.probs = cl.layers.dense(pose_conv,
                                                     activation_conv,
                                                     **self.fc_caps_params)

        # reconstruction network
        if decoder == 'FC':
            decoder = cl.decoders.FCDecoderNet(self.height, 
                                                self.width, 
                                                self.channels, 
                                                self.num_label, 
                                                self.labels,
                                                self.poses)
        elif decoder == 'DECONV':
            decoder = cl.decoders.DeconvDecoderNet(self.height, 
                                                    self.width, 
                                                    self.channels, 
                                                    self.num_label, 
                                                    self.labels,
                                                    self.probs,
                                                    self.vec_shape)      
        else:
            print('decoder', decoder, 'not known. Value should be either FC or DECONV.')   
            exit(-1)   

        self.recon_imgs, self.labels_one_hoted = decoder.reconstruct_image()

        self.calculate_accuracy()

        # count number of trainable parameters
        self.num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

        # log parameters into file params.conf
        self.log_params()

        return self.poses, self.probs


    def calculate_accuracy(self):
        with tf.variable_scope('accuracy'):
            logits_idx = tf.to_int32(tf.argmax(cl.softmax(self.probs, axis=1), axis=1))
            correct_prediction = tf.equal(tf.to_int32(self.labels), logits_idx)
            correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy = tf.reduce_mean(correct / tf.cast(tf.shape(self.probs)[0], tf.float32))
            cl.summary.scalar('accuracy', self.accuracy, verbose=cfg.summary_verbose)


    def _loss(self):
        with tf.variable_scope("loss"):
            # 1. Margin loss
            margin_loss = cl.losses.margin_loss(logits=self.probs,
                                                labels=tf.squeeze(self.labels_one_hoted, axis=(2, 3)))
            cl.summary.scalar('margin_loss', margin_loss, verbose=cfg.summary_verbose)

            # 2. The reconstruction loss
            origin = tf.reshape(self.raw_imgs, shape=(-1, self.height * self.width * self.channels))

            squared = tf.square(self.recon_imgs - origin)
            reconstruction_err = tf.reduce_mean(squared)
            cl.summary.scalar('reconstruction_loss', reconstruction_err, verbose=cfg.summary_verbose)

            # 3. Total loss
            # The paper uses sum of squared error as reconstruction error, but we
            # have used reduce_mean in `# 2 The reconstruction loss` to calculate
            # mean squared error. In order to keep in line with the paper,the
            # regularization scale should be 0.0005*784=0.392
            total_loss = margin_loss + cfg.regularization_scale * reconstruction_err

            cl.summary.scalar('total_loss', total_loss, verbose=cfg.summary_verbose)
            return total_loss


    def train(self, optimizer, num_gpus=1):
        self.global_step = tf.Variable(1, name='global_step', trainable=False)
        total_loss = self._loss()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_ops = optimizer.minimize(total_loss, global_step=self.global_step)
        summary_ops = tf.summary.merge_all()

        return(total_loss, train_ops, summary_ops)

    def log_params(self):

        params = os.path.join(cfg.results_dir, 'params.conf')
        fd_params = open(params, 'w')
        fd_params.write('dataset: ' + cfg.dataset + '\n')
        fd_params.write('vector shape: ' + str(self.vec_shape) + '\n')
        fd_params.write('decoder ' + self.decoder + '\n')
        fd_params.write('attention: ' + str(self.attention) + '\n')
        fd_params.write('conv1: ' + str(self.conv1_params) + '\n')
        fd_params.write('primary caps: ' + str(self.prim_caps_params) + '\n')
        fd_params.write('conv caps: ' + str(self.conv_caps_params) + '\n')
