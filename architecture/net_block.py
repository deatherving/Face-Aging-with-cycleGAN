# -*- coding: utf-8 -*-
# author: K

import tensorflow as tf

import tensorflow.contrib.slim as slim

import numpy as np


def conv2d(inputs, output_dim, kernel_size, strides, stddev = 0.02, padding = 'SAME', name = 'conv2d'):
	with tf.variable_scope(name):
		return slim.conv2d(inputs, output_dim, kernel_size, strides, padding = padding, activation_fn = None, 
					weights_initializer = tf.truncated_normal_initializer(stddev = stddev), biases_initializer = None)

# conv2d_transpose equals to deconv2d and nn.SpatialFullConvolution in torch
def conv2d_transpose(inputs, output_dim, kernel_size, strides, stddev = 0.02, padding = "SAME", name = 'conv2d_transpose'):
	with tf.variable_scope(name):
		return slim.conv2d_transpose(inputs, output_dim, kernel_size, strides, padding = padding, activation_fn = None,
					weights_initializer = tf.truncated_normal_initializer(stddev = stddev), biases_initializer = None)

# Paper: https://arxiv.org/pdf/1607.08022
# Apply instance_norm rather than batch norm, the paper indicates instance norm prevents instance-specific mean and covariance shift. In test phase, the layer should be applied, either.
# I implemented this layer according to the formula in paper, hope it works
def instance_norm(x, epsilon = 1e-5, name = "instance_norm"):
	with tf.variable_scope(name):
		# Assume HWC mode
		depth = x.get_shape()[3]

		# The paper only gives how to compute the normalized result. However, this is not a trainable layer, we should add two variables for training
		# Each chanel refers to a scale dim
		scale = tf.get_variable("scale", [depth], initializer = tf.random_normal_initializer(1.0, 0.02, dtype = tf.float32))
		offset = tf.get_variable("offset", [depth], initializer = tf.constant_initializer(0.0)) 

		# This is quite important. The axes should be [1, 2] if the image chanel is HWC, otherwise the axes has to be [2, 3]. 
		# To be noticed that some of the implementations set keep_dims to True. I don't know the difference in the image process situation. I'll test it later.
		mean, var = tf.nn.moments(x, axes = [1, 2], keep_dims = True)
		normalized = (x - mean) / tf.sqrt(var + epsilon)

	return scale * normalized + offset

# Paper: https://arxiv.org/abs/1512.03385
# By Kaiming He
def build_resnet_block(x, dim, name, kernel_size = 3, strides = 1):
	with tf.variable_scope(name):
		y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
		y = instance_norm(conv2d(y, dim, kernel_size, strides, padding = 'VALID', name = 'conv_res_0'), name = "instance_norm_res_0")
		y = tf.nn.relu(y)
		y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
		y = instance_norm(conv2d(y, dim, kernel_size, strides, padding = 'VALID', name = 'conv_res_1'), name = "instance_norm_res_1")
	return y + x

# Unlike the implementation in Keras, I used this version, which seems more stable than the version in keras.
# 
def leakyrelu(x, alpha = 0.2):
	return tf.maximum(x, alpha * x)


