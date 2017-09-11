# -*- coding: utf-8 -*-
# author: K

import tensorflow as tf
from net_block import *


class Encoder:
	def __init__(self, nef, name):
		self.reuse = False
		self.nef = nef
		self.name = name
	def build_graph(self, inputs):
		with tf.variable_scope(self.name, reuse = self.reuse):
			# we could concat the meta data with the inputs or with the result of first conv2d, let's try the former one
                        p0 = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
                        c0 = tf.nn.relu(instance_norm(conv2d(p0, self.nef, 7, 1,  padding = 'VALID', name = 'conv2d_c0'), name = 'instance_norm_c0'))
                        c1 = tf.nn.relu(instance_norm(conv2d(c0, self.nef * 2, 3, 2, name = 'conv2d_c1'), name = 'instance_norm_c1'))
                        c2 = tf.nn.relu(instance_norm(conv2d(c1, self.nef * 4, 3, 2, name = 'conv2d_c2'), name = 'instance_norm_c2'))

                        #build resnet
                        r0 = build_resnet_block(c2, self.nef * 4, name = 'res_r0')
                        r1 = build_resnet_block(r0, self.nef * 4, name = 'res_r1')
                        r2 = build_resnet_block(r1, self.nef * 4, name = 'res_r2')
                        r3 = build_resnet_block(r2, self.nef * 4, name = 'res_r3')
                        r4 = build_resnet_block(r3, self.nef * 4, name = 'res_r4')
                        r5 = build_resnet_block(r4, self.nef * 4, name = 'res_r5')
			r6 = build_resnet_block(r5, self.nef * 4, name = 'res_r6')
                        r7 = build_resnet_block(r6, self.nef * 4, name = 'res_r7')
                        r8 = build_resnet_block(r7, self.nef * 4, name = 'res_r8')

			output = tf.nn.tanh(r8)

		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
		return output


if __name__ == '__main__':
	image = np.random.random([1, 128, 128, 3]).astype('float32')

	e = Encoder(64, "Encoder")

	print e.build_graph(image).get_shape()
