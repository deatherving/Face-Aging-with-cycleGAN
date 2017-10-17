# -*- coding: utf-8 -*-
# author: K

import tensorflow as tf
from net_blocks import *

import numpy as np


class Generator:
	def __init__(self, ngf, image_size, name):
		self.reuse = False
		self.ngf = ngf
		self.name = name
		
	def build_graph(self, inputs, ages, genders):
		with tf.variable_scope(self.name, reuse = self.reuse):
			c_inputs = tf.concat([inputs, ages], axis = 1)
			c_inputs = tf.concat([c_inputs, genders], axis = 1)

			f0 = fully_connected((c_inputs), 8 * 8 * self.ngf, name = "fc0")

			r0 = tf.nn.relu(tf.reshape(f0, [-1, 8, 8, self.ngf]))

			t0 = tf.nn.relu(conv2d_transpose(r0, self.ngf / 2, 5, 2, name = "conv2d_transpose0"))

			t1 = tf.nn.relu(conv2d_transpose(t0, self.ngf / 2**2, 5, 2, name = "conv2d_transpose1"))

			t2 = tf.nn.relu(conv2d_transpose(t1, self.ngf / 2**3, 5, 2, name = "conv2d_transpose2"))

			t3 = tf.nn.relu(conv2d_transpose(t2, self.ngf / 2**4, 5, 2, name = "conv2d_transpose3"))

			t4 = tf.nn.tanh(conv2d_transpose(t3, 3, 5, 1, name = "conv2d_tranpose4"))

		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
		return t4
		
	def __call__(self, inputs, ages, genders):
		return self.build_graph(inputs, ages, genders)


if __name__ == '__main__':
	inputs = np.random.random([128, 50]).astype('float32')
	
	ages = np.random.random([128, 10]).astype('float32')

	genders = np.random.random([128, 2])

	g = Generator(1024, 128, "Generator")

	print g(inputs, ages, genders).get_shape()
