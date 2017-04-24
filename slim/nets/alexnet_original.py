# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a model definition for AlexNet.

This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014

Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.

Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)

@@alexnet_v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

# slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def alexnet_original_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        biases_initializer=tf.constant_initializer(0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d, conv_with_group_layer], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                return arg_sc


# def convolution(inputs,
#                 num_outputs,
#                 kernel_size,
#                 stride=1,
#                 padding='SAME',
#                 data_format=None,
#                 rate=1,
#                 activation_fn=nn.relu,
#                 normalizer_fn=None,
#                 normalizer_params=None,
#                 weights_initializer=initializers.xavier_initializer(),
#                 weights_regularizer=None,
#                 biases_initializer=init_ops.zeros_initializer(),
#                 biases_regularizer=None,
#                 reuse=None,
#                 variables_collections=None,
#                 outputs_collections=None,
#                 trainable=True,
#                 scope=None):


@slim.add_arg_scope
def conv_with_group_layer(inputs,
                          num_outputs,
                          kernel_size,
                          stride=1,
                          padding='SAME',
                          trainable=True,
                          scope=None,
                          group=1,
                          outputs_collections=None,
                          relu=True):
    c_i = inputs.get_shape()[-1].value
    c_o = num_outputs
    assert c_i % group == 0
    assert c_o % group == 0
    s_h = stride
    s_w = stride
    k_h, k_w = kernel_size

    def convolve(input_tensor, kernel_tensor):
        return tf.nn.conv2d(input_tensor, kernel_tensor, [1, s_h, s_w, 1], padding=padding)

    with tf.variable_scope(scope) as var_scope:
        from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer
        init_weights = xavier_initializer()
        init_biases = tf.constant_initializer(0.0)
        var_kernel = tf.get_variable('weights', [k_h, k_w, c_i / group, c_o], initializer=init_weights, trainable=trainable,
                                     collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES])
        var_biases = tf.get_variable('biases', [c_o], initializer=init_biases, trainable=trainable,
                                     collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES])
        if group == 1:
            conv = convolve(inputs, var_kernel)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=group, value=inputs)
            kernel_groups = tf.split(axis=3, num_or_size_splits=group, value=var_kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(axis=3, values=output_groups)
        if relu:
            bias = tf.nn.bias_add(conv, var_biases)
            outputs = tf.nn.relu(bias, name=var_scope.name)
        else:
            outputs = tf.nn.bias_add(conv, var_biases, name=var_scope.name)
        from tensorflow.contrib.layers.python.layers import utils
        return utils.collect_named_outputs(outputs_collections, var_scope.name, outputs)


def alexnet_original(inputs,
                     num_classes=1000,
                     is_training=True,
                     dropout_keep_prob=0.5,
                     spatial_squeeze=True,
                     scope='alexnet_v2',
                     ):
    """original AlexNet .
  Described in: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
  Parameters from: https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224. To use in fully
        convolutional mode, set spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
    end_points_collection = 'alexnet_original_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d, conv_with_group_layer, slim.flatten],
                        outputs_collections=[end_points_collection]):
        net = conv_with_group_layer(inputs, 96, [11, 11], 4, padding='VALID', scope='conv1')
        net = slim.max_pool2d(net, [3, 3], 2, padding='VALID', scope='pool1')
        net = conv_with_group_layer(net, 256, [5, 5], padding='SAME', scope='conv2', group=2)
        net = slim.max_pool2d(net, [3, 3], 2, padding='VALID', scope='pool2')
        net = conv_with_group_layer(net, 384, [3, 3], padding='SAME', scope='conv3')
        net = conv_with_group_layer(net, 384, [3, 3], padding='SAME', scope='conv4', group=2)
        net = conv_with_group_layer(net, 256, [3, 3], padding='SAME', scope='conv5', group=2)
        net = slim.max_pool2d(net, [3, 3], 2, padding='VALID', scope='pool5')

        # Use conv2d instead of fully_connected layers.
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=trunc_normal(0.005),
                            biases_initializer=tf.constant_initializer(0.1)):
            # net = slim.conv2d(net, 4096, [6, 6], padding='VALID', scope='fc6')
            net = slim.flatten(net, scope='flt1')
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            # net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
            # net = slim.conv2d(net, num_classes, [1, 1],
            #                   activation_fn=None,
            #                   normalizer_fn=None,
            #                   biases_initializer=tf.zeros_initializer(),
            #                   scope='fc8')
            net = slim.fully_connected(net, num_classes,
                              activation_fn=None,
                              normalizer_fn=None,
                              biases_initializer=tf.zeros_initializer(),
                              scope='fc8')

        # Convert end_points_collection into a end_point dict.
        from tensorflow.contrib.layers.python.layers.utils import convert_collection_to_dict
        end_points = convert_collection_to_dict(end_points_collection)
        # if spatial_squeeze:
        #     net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        #     end_points['fc8/squeezed'] = net
        return net, end_points


alexnet_original.default_image_size = 227
