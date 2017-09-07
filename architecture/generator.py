# -*- coding: utf-8 -*-
# author: K


import tensorflow as tf
from net_block import *


# The network structure is from
# Paper: Perceptual losses for real-time style transfer and super-resolution
# By J.Johnson
class Generator:
	def __init__(self, ngf, name):
		self.reuse = False
		self.ngf = ngf
		self.name = name
	def build_graph(self, inputs, meta_data):
		# The graph consists of resnet blocks
		with tf.variable_scope(self.name, reuse = self.reuse):
			c_inputs = tf.concat([inputs, meta_data], 3)
			# we could concat the meta data with the inputs or with the result of first conv2d, let's try the former one
			p0 = tf.pad(c_inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
			c0 = tf.nn.relu(instance_norm(conv2d(p0, self.ngf, 7, 1,  padding = 'VALID', name = 'conv2d_c0'), name = 'instance_norm_c0'))
			c1 = tf.nn.relu(instance_norm(conv2d(c0, self.ngf * 2, 3, 2, name = 'conv2d_c1'), name = 'instance_norm_c1'))
			c2 = tf.nn.relu(instance_norm(conv2d(c1, self.ngf * 4, 3, 2, name = 'conv2d_c2'), name = 'instance_norm_c2'))

			#build resnet
			r0 = build_resnet_block(c2, self.ngf * 4, name = 'res_r0')
			r1 = build_resnet_block(r0, self.ngf * 4, name = 'res_r1')
			r2 = build_resnet_block(r1, self.ngf * 4, name = 'res_r2')
			r3 = build_resnet_block(r2, self.ngf * 4, name = 'res_r3')
			r4 = build_resnet_block(r3, self.ngf * 4, name = 'res_r4')
			r5 = build_resnet_block(r4, self.ngf * 4, name = 'res_r5')
			r6 = build_resnet_block(r5, self.ngf * 4, name = 'res_r6')
			r7 = build_resnet_block(r6, self.ngf * 4, name = 'res_r7')
			r8 = build_resnet_block(r7, self.ngf * 4, name = 'res_r8')
			
			c3 = tf.nn.relu(instance_norm(conv2d_transpose(r5, self.ngf * 2, 3, 2, name = "conv2d_transpose_c3"), name = 'instance_norm_c3'))
			c4 = tf.nn.relu(instance_norm(conv2d_transpose(c3, self.ngf, 3, 2, name = "con2d_transpose_c4"), name = 'instance_norm_c4'))
			p1 = tf.pad(c4, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
			c5 = tf.nn.tanh(conv2d(p1, 3, 7, 1, name = 'conv2d_c5', padding = 'VALID'))

		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
		return c5

	def __call__(self, inputs, meta):
		return self.build_graph(inputs, meta)

if __name__ == '__main__':
	image = np.random.random([1, 256, 256, 3]).astype('float32')
	meta = np.random.random([1, 256, 256, 3]).astype('float32')

	g = Generator(64, "GeneratorA")
	
	
	print g.build_graph(image, meta).get_shape()
