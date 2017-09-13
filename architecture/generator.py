# -*- coding: utf-8 -*-
# author: K


import tensorflow as tf
from net_block import *


# The network structure is from
# Paper: Perceptual losses for real-time style transfer and super-resolution
# By J.Johnson
class Generator:
	def __init__(self, ngf, image_size, name):
		self.reuse = False
		self.ngf = ngf
		self.name = name
		self.image_size = image_size
	def build_graph(self, inputs, ages, genders):
		# The graph consists of resnet blocks
		with tf.variable_scope(self.name, reuse = self.reuse):

			c_inputs = tf.concat([inputs, ages], 1)
			c_inputs = tf.concat([c_inputs, genders], 1)

			image_chanel = self.image_size / 4

			f0 = fully_connected(flatten(c_inputs), image_chanel * image_chanel * 256, name = "fully0")

			reshape0 = tf.reshape(f0,[-1, image_chanel, image_chanel, 256])

			c0 = tf.nn.relu(instance_norm(conv2d_transpose(reshape0, self.ngf * 2, 3, 2, name = "conv2d_transpose_c0"), name = 'instance_norm_c0'))
			c1 = tf.nn.relu(instance_norm(conv2d_transpose(c0, self.ngf, 3, 2, name = "con2d_transpose_c1"), name = 'instance_norm_c1'))
			p0 = tf.pad(c1, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
			c2 = tf.nn.tanh(conv2d(p0, 3, 7, 1, name = 'conv2d_c2', padding = 'VALID'))

			

		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
		return c2

	def __call__(self, inputs, ages, genders):
		return self.build_graph(inputs, ages, genders)

if __name__ == '__main__':
	image = np.random.random([1, 50]).astype('float32')
	ages = np.random.random([1, 10]).astype('float32')
	genders = np.random.random([1, 2]).astype('float32')

	g = Generator(64, 256, "GeneratorA")
	
	print g.build_graph(image, ages, genders).get_shape()
