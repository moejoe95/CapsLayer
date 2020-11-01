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

import datetime
import os
import pprint

from config import cfg


class CapsNet(object):
    def __init__(self, height=28, width=28, channels=1, num_label=10):
        '''
        CapsNet architecture for datasets like mnist or fashion mnist.
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

        self.lr = cfg.learning_rate
        self.routing_method = cfg.routing_alg
        
        self.decoder = cfg.decoder
        self.attention = cfg.attention
        self.num_iter = cfg.routing_iterations
        self.drop_ratio = cfg.drop_ratio
        self.drop_mode = cfg.drop_mode
        self.preCapsResidualNet = cfg.preResNet

        self.T = tf.zeros((), dtype=tf.float32)
        self.D = tf.zeros((), dtype=tf.float32)


    def create_network(self, inputs, labels):
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

        self.raw_imgs = inputs
        self.labels = labels

        # set model parameters
        self.vec_shape = [8, 1]

        self.layers = 3
        self.skip = [(0,3)] 
        self.make_skips = True

        self.conv1_params = {
            "filters": 64,
            "kernel_size": 9,
            "strides": 1
        }

        self.prim_caps_params = {
            "filters": 64,
            "kernel_size": 7,
            "strides": 2,
            "out_caps_dims": self.vec_shape
        }

        self.conv_caps_params = {
            "filters": 32,
            "kernel_size": 3,
            "strides": 2,
            "out_caps_dims": self.vec_shape,
            "num_iter": self.num_iter,
            "routing_method": self.routing_method,
            "padding": 'SAME'
        }
        
        self.fc_caps_params = {
            "num_outputs": self.num_label,
            "out_caps_dims": [16, 1],
            "routing_method": self.routing_method,
            "num_iter": self.num_iter,
            "coordinate_addition": False
        }

        inputs = tf.reshape(self.raw_imgs, shape=[-1, self.height, self.width, self.channels])

        if self.preCapsResidualNet: # pre-capsule residual subnetwork
            conv1 = cl.layers.residualConvs(inputs, self.conv1_params, padding='SAME')
        else: # do normal convolution before capsule network
            conv1 = tf.layers.conv2d(inputs,
                                    **self.conv1_params,
                                    padding='VALID',
                                    activation=tf.nn.relu,
                                    name="Conv1_layer")

        # idea of attention in CapsNet is from Hoogi et al. 
        # in the paper 'Self-Attention Capsule Networks for Image Classification'
        if self.attention:
            conv1 = cl.layers.selfAttention(conv1, self.conv1_params['filters'])

        # primary caps layer
        pose, activation = cl.layers.primaryCaps(conv1,
                                                **self.prim_caps_params,
                                                method="norm",
                                                name="PrimaryCaps_layer")

        # residual capsule network                                     
        pose, activation, c_1 = cl.layers.residualCapsNetwork(pose, 
                                            activation, 
                                            self.conv_caps_params, 
                                            layers=self.layers, 
                                            skip=self.skip,
                                            make_skips=self.make_skips,
                                            drop_ratio=self.drop_ratio,
                                            drop_mode=self.drop_mode) 

        # fully connected capsule layer
        with tf.compat.v1.variable_scope('FullyConnCaps_layer'):
            num_inputs = np.prod(cl.shape(pose)[1:4])
            pose = tf.reshape(pose, shape=[-1, num_inputs, self.vec_shape[0], self.vec_shape[1]])
            activation = tf.reshape(activation, shape=[-1, num_inputs])

            self.poses, self.probs, c_2 = cl.layers.dense(pose,
                                                     activation,
                                                     **self.fc_caps_params)
        

        with tf.compat.v1.variable_scope('gamma_metrics'):
            if c_1 is not None and c_2 is not None:
                self.T = (cl.ops.t_score(c_1) + cl.ops.t_score(c_2)) / 2                                
                self.D = cl.ops.d_score(activation)

        # reconstruction network
        if self.decoder == 'FC':
            decoderNet = cl.decoders.FCDecoderNet(self.height, 
                                                self.width, 
                                                self.channels, 
                                                self.num_label, 
                                                self.labels,
                                                self.poses)
            self.recon_imgs, self.labels_one_hoted = decoderNet.reconstruct_image()

        elif self.decoder == 'DECONV':
            decoderNet = cl.decoders.DeconvDecoderNet(self.height, 
                                                    self.width, 
                                                    self.channels, 
                                                    self.num_label, 
                                                    self.labels,
                                                    self.probs,
                                                    self.vec_shape)     
            self.recon_imgs, self.labels_one_hoted = decoderNet.reconstruct_image() 
        
        elif self.decoder == 'NONE':
            # use no decoder at all
            labels = tf.one_hot(self.labels, depth=self.num_label, axis=-1, dtype=tf.float32)
            self.recon_imgs, self.labels_one_hoted = None, tf.reshape(labels, (-1, self.num_label, 1, 1))

        else:
            print("decoder:", "'" + self.decoder + "'", "not known")
            exit(-1)

        self.calculate_accuracy()

        # count number of trainable parameters
        self.num_train_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])

        # log parameters into file params.conf
        self.log_params()

        return self.poses, self.probs


    def calculate_accuracy(self):
        with tf.compat.v1.variable_scope('accuracy'):
            logits_idx = tf.to_int32(tf.argmax(cl.softmax(self.probs, axis=1), axis=1))
            correct_prediction = tf.equal(tf.to_int32(self.labels), logits_idx)
            correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy = tf.reduce_mean(correct / tf.cast(tf.shape(self.probs)[0], tf.float32))
            cl.summary.scalar('accuracy', self.accuracy, verbose=cfg.summary_verbose)


    def _loss(self):
        with tf.compat.v1.variable_scope("loss"):
            # 1. Margin loss
            margin_loss = cl.losses.margin_loss(logits=self.probs,
                                                labels=tf.squeeze(self.labels_one_hoted, axis=(2, 3)))
            cl.summary.scalar('margin_loss', margin_loss, verbose=cfg.summary_verbose)

            # 2. The reconstruction loss
            if self.decoder == 'NONE':
                reconstruction_err = 0
            else:
                origin = tf.reshape(self.raw_imgs, shape=(-1, self.height * self.width * self.channels))
                squared = tf.square(self.recon_imgs - origin)
                reconstruction_err = tf.reduce_mean(squared)
                cl.summary.scalar('reconstruction_loss', reconstruction_err, verbose=cfg.summary_verbose)

            # total loss
            total_loss = margin_loss + cfg.regularization_scale * reconstruction_err

            cl.summary.scalar('total_loss', total_loss, verbose=cfg.summary_verbose)

            return total_loss


    def train(self, optimizer, num_gpus=1):
        self.global_step = tf.Variable(1, name='global_step', trainable=False)

        cl.summary.scalar('learning_rate', self.lr, verbose=cfg.summary_verbose)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)

        total_loss = self._loss()
        train_ops = optimizer.minimize(total_loss, global_step=self.global_step)
        summary_ops = tf.compat.v1.summary.merge_all()

        if self.global_step % 60000 == 0: # check if learning rate needs decrease
            if self.prev_loss is not None:
                self.lr = cl.losses.get_adaptive_lr(self.prev_loss, total_loss, self.lr)
            self.prev_loss = total_loss

        return(total_loss, train_ops, summary_ops)


    def log_params(self):
        i = 0
        dir_exists = False
        while not dir_exists:
            self.model_result_dir = os.path.join(cfg.results_dir, cfg.dataset + '_' + cfg.model + '_' + str(i))
            if not os.path.exists(self.model_result_dir):
                os.makedirs(self.model_result_dir)
                dir_exists = True
            i += 1

        params = os.path.join(self.model_result_dir, 'params.conf')

        with open(params, 'wt') as fd_params:

            fd_params.write(cfg.model + ' at ' + str(datetime.datetime.now()) + '\n\n')

            fd_params.write('dataset: ' + cfg.dataset + '\n')
            fd_params.write('batch_size: ' + str(cfg.batch_size) + '\n')
            fd_params.write('decay step: ' + str(cfg.decay_step) + '\n\n')

            # get all class variables
            settings = self.__dict__.copy()

            # remove runtime variables
            settings.pop('raw_imgs')
            settings.pop('labels')
            settings.pop('poses')
            settings.pop('probs')
            settings.pop('recon_imgs')
            settings.pop('labels_one_hoted')
            settings.pop('accuracy')
            settings.pop('T')
            settings.pop('D')
            
            # print model parameters
            pp = pprint.PrettyPrinter(indent=2, stream=fd_params)
            pp.pprint(settings)
