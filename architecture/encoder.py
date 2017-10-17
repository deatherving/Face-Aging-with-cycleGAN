# -*- coding: utf-8 -*-
# author: K


import tensorflow as tf
from net_blocks import *

import numpy as np


class Encoder:
	def __init__(self, nef, z_dim, name):
		self.nef = nef
		self.reuse = False
		self.z_dim = z_dim
		self.name = name
	def build_graph(self, inputs):
		with tf.variable_scope(self.name, reuse = self.reuse):
			c0 = tf.nn.relu(conv2d(inputs, self.nef, 5, 2, name = "conv2d_c0"))
			c1 = tf.nn.relu(conv2d(c0, self.nef * 2, 5, 2, name = "conv2d_c1"))
			c2 = tf.nn.relu(conv2d(c1, self.nef * 2 * 2, 5, 2, name = "conv2d_c2"))
			c3 = tf.nn.relu(conv2d(c2, self.nef * 2 ** 3, 5, 2, name = "conv2d_c3"))

		
			fc0 = fully_connected(flatten(c3), self.z_dim, name = "fc0")

		output = tf.nn.tanh(fc0)

		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
		return output


	def __call__(self, inputs):
		return self.build_graph(inputs)


if __name__ == '__main__':
	inputs = np.random.random([128, 128, 128, 3]).astype('float32')

	encoder = Encoder(64, 50, "Encoder")

	print encoder(inputs).get_shape()

	
