# -*- coding: utf-8 -*-
# author K

import tensorflow as tf
from net_blocks import *

class Discriminator:
	def __init__(self, ndf, name):
		self.ndf = ndf
		self.name = name
		self.reuse = False
	def build_graph(self, inputs, ages, genders):
		with tf.variable_scope(self.name, reuse = self.reuse):
			c0 = tf.nn.relu(batch_norm(conv2d(inputs, self.ndf, 5, 2, name = "conv0"), name = "batch_norm0"))
			c_c0 = tf.concat([c0, ages], axis = 3)
			c_c1 = tf.concat([c_c0, genders], axis = 3)

			c1 = tf.nn.relu(batch_norm(conv2d(c_c1, self.ndf * 2, 5, 2, name = "conv1"), name = "batch_norm1"))

			c2 = tf.nn.relu(batch_norm(conv2d(c1, self.ndf * (2**2), 5, 2, name = "conv2"), name = "batch_norm2"))
			c3 = tf.nn.relu(batch_norm(conv2d(c2, self.ndf * (2**3), 5, 2, name = "conv3"), name = "batch_norm3"))

			f0 = leakyrelu(fully_connected(flatten(c3), 1024, name = "fc0"))

			f1 = fully_connected(f0, 1, name = "fc1")
			
		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
		
		return f1
	
	def __call__(self, inputs, ages, genders):
		return self.build_graph(inputs, ages, genders)

class DiscriminatorZ:
	def __init__(self, ndf, name):
		self.ndf = ndf
		self.name = name
		self.reuse = False
	def build_graph(self, inputs):
		with tf.variable_scope(self.name, reuse = self.reuse):
			f0 = tf.nn.relu(batch_norm(fully_connected(inputs, self.ndf, name = "fc0"), name = "batch_norm0"))
			f1 = tf.nn.relu(batch_norm(fully_connected(f0, self.ndf / 2, name = "fc1"), name = "batch_norm1"))
			f2 = tf.nn.relu(batch_norm(fully_connected(f1, self.ndf / 4, name = "fc2"), name = "batch_norm2"))
			f3 = tf.nn.relu(batch_norm(fully_connected(f2, self.ndf / 8, name = "fc3"), name = "batch_norm3"))

			f4 = fully_connected(flatten(f3), 1, name = "fc4")

		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
		
		return f4
	
	def __call__(self, inputs):
		return self.build_graph(inputs)



if __name__ == '__main__':
	inputs = np.random.random([128, 128, 128, 3]).astype('float32')
	ages = np.random.random([128, 64, 64, 10]).astype('float32')
	genders = np.random.random([128, 64, 64, 2]).astype('float32')

	d = Discriminator(16, "Discriminator")

	print d(inputs, ages, genders).get_shape()
	
