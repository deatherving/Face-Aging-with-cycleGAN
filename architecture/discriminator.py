# -*- coding: utf-8 -*-
# author: K

import tensorflow as tf
from net_block import *

class Discriminator:
	def __init__(self, ndf, name):
		self.reuse = False
		self.ndf = ndf
		self.name = name
	def build_graph(self, inputs, meta_data):
		with tf.variable_scope(self.name, reuse = self.reuse):

			c_inputs = tf.concat([inputs, meta_data], 3)
			c0 = conv2d(c_inputs, self.ndf, 4, 2, name = "conv2d_c0")
			l0 = leakyrelu(c0)
	
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


	def __call__(self, inputs, meta):
		return self.build_graph(inputs, meta)

if __name__ == '__main__':
	image = np.random.random([1, 256, 256, 3]).astype('float32')
	meta = np.random.random([1, 256, 256, 3]).astype('float32')

	d = Discriminator(64, "DiscriminatorA")

	print d.build_graph(image, meta).get_shape()

