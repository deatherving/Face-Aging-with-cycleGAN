# -*- coding: utf-8 -*-
# author: K


import tensorflow as tf

import tensorflow.contrib.slim as slim

import numpy as np


def fully_connected(inputs, num_outputs, stddev = 0.02, name = "fc"):
	with tf.variable_scope(name):
		stddev = np.sqrt(1.0 / (np.sqrt(inputs.get_shape()[-1].value * num_outputs)))
		return slim.fully_connected(inputs, num_outputs, activation_fn = None,
				weights_initializer = tf.truncated_normal_initializer(stddev = stddev), biases_initializer = tf.zeros_initializer())

def conv2d(inputs, output_dim, kernel_size, strides, stddev = 0.02, padding = "SAME", name = "conv2d"):
	with tf.variable_scope(name):
		stddev = np.sqrt(2.0 / (np.sqrt(inputs.get_shape()[-1].value * output_dim) * kernel_size ** 2))
		return slim.conv2d(inputs, output_dim, kernel_size, strides, padding = padding, activation_fn = None,
					weights_initializer = tf.truncated_normal_initializer(stddev = stddev), biases_initializer = tf.zeros_initializer())

def conv2d_transpose(inputs, output_dim, kernel_size, strides, stddev = 0.02, padding = "SAME", name = "conv2d_transpose"):
	with tf.variable_scope(name):
		stddev = np.sqrt(1.0 / (np.sqrt(inputs.get_shape()[-1].value * output_dim) * kernel_size ** 2))
		return slim.conv2d_transpose(inputs, output_dim, kernel_size, strides, padding = padding, activation_fn = None,
					weights_initializer = tf.truncated_normal_initializer(stddev = stddev), biases_initializer = tf.zeros_initializer())

def batch_norm(inputs, is_training = True, name = "batch_norm"):
	with tf.variable_scope(name):
		return slim.batch_norm(inputs, activation_fn = None, is_training = is_training)


def leakyrelu(x, alpha = 0.2):
	return tf.maximum(x, alpha * x)

def flatten(x):
	return slim.flatten(x)
