# -*- coding: utf-8 -*-
# author: K

import tensorflow as tf
from net_block import *

class Discriminator:
	def __init__(self, ndf, name):
		self.reuse = False
		self.ndf = ndf
		self.name = name
	def build_graph(self, inputs, ages, genders):
		with tf.variable_scope(self.name, reuse = self.reuse):

			c0 = conv2d(inputs, self.ndf, 4, 2, name = "conv2d_c0")
			l0 = leakyrelu(c0)

			# add the conditions age and gender as a feature map rather than a original input, I saw a lot of fellows did like this so I eventually followed.
			c_l0 = tf.concat([l0, ages], axis = 3)
			c_l0 = tf.concat([c_l0, genders], axis = 3)

			c1 = conv2d(l0, self.ndf * 2, 4, 2, name = "conv2d_c1")
			l1 = leakyrelu(instance_norm(c1, name = "instance_norm_l1"))
			c2 = conv2d(l1, self.ndf * 4, 4, 2, name = 'conv2d_c2')
			l2 = leakyrelu(instance_norm(c2, name = "instance_norm_l2"))
			
			c3 = conv2d(l2, self.ndf * 3, 4, 1, name = 'conv2d_c3')
			l3 = leakyrelu(instance_norm(c3, name = 'instance_norm_l3'))
	
			c4 = conv2d(l3, 1, 4, 1, name = "conv2d_c4")

		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
		
		# Actually we could apply sigmoid layer here. However, least square GAN supports logits instead of probabilities. So, I followed it.
		# The least square GAN is implemented in another repository. You could check it.
		return c4


	def __call__(self, inputs, ages, genders):
		return self.build_graph(inputs, ages, genders)


class DiscriminatorZ:
	def __init__(self, ndf, name):
		self.ndf = ndf
		self.name = name
		self.reuse = False
	def build_graph(self, inputs):
		with tf.variable_scope(self.name, reuse = self.reuse):
			f0 = fully_connected(inputs, self.ndf * 4, name = "fully0")
			f1 = fully_connected(f0, self.ndf * 2, name = "fully1")
			f2 = fully_connected(f1, self.ndf, name = "fully2")
			
		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
		return f2
	
	def __call__(self, inputs):
		return self.build_graph(inputs)



if __name__ == '__main__':
	image = np.random.random([1, 256, 256, 3]).astype('float32')
	ages = np.random.random([1, 128, 128, 10]).astype('float32')
	genders = np.random.random([1, 128, 128, 2]).astype('float32')

	d = Discriminator(64, "DiscriminatorA")

	print d.build_graph(image, ages, genders).get_shape()

