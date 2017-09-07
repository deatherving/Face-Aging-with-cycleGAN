# -*- coding: utf-8 -*-
# author: K

import tensorflow as tf
import numpy as np
from architecture.generator import Generator
from architecture.discriminator import Discriminator

class CycleGAN:
	def __init__(self, config):
		# initializing 2 generators and 2 discriminators
		# We don't use ImagePool. Perhaps later
		self.lr = float(config['lr'])
		self.generatorG = Generator(int(config['ngf']), "GeneratorG")
		self.generatorF = Generator(int(config['ngf']), "GeneratorF")
		self.discriminatorA = Discriminator(int(config['ndf']), "DiscriminatorA")
		self.discriminatorB = Discriminator(int(config['ndf']), "DiscriminatorB")
		self.is_peason_div = int(config['is_peason'])
		self.L1_lambda = float(config['L1_lambda'])
		self.width = int(config['width'])
		self.height = int(config['height'])
		self.img_A = tf.placeholder(tf.float32, [None, self.width, self.height, 3])
		self.img_B = tf.placeholder(tf.float32, [None, self.width, self.height, 3])
		self.meta_vector_A2B = tf.placeholder(tf.float32, [None, self.width, self.height, 1])
		self.meta_vector_B2A = tf.placeholder(tf.float32, [None, self.width, self.height, 1])
		self.sample_vector = tf.placeholder(tf.float32, [None, self.width, self.height, 3])
		self.sample_meta_data = tf.placeholder(tf.float32, [None, self.width, self.height, 1])
	def build_model(self):
		# The generator logits
		g_logits_fake_B = self.generatorG(self.img_A, self.meta_vector_A2B)
		g_logits_rectA = self.generatorF(g_logits_fake_B, self.meta_vector_B2A)
		g_logits_fake_A = self.generatorF(self.img_B, self.meta_vector_B2A)
		g_logits_rectB = self.generatorG(g_logits_fake_A, self.meta_vector_A2B)

		# The discriminator logits, we only need fake logits to compute the 
		d_logits_fake_B = self.discriminatorB(g_logits_fake_B, self.meta_vector_A2B)
		d_logits_fake_A = self.discriminatorA(g_logits_fake_A, self.meta_vector_B2A)
		d_logits_real_B = self.discriminatorB(self.img_B, self.meta_vector_A2B)
		d_logits_real_A = self.discriminatorA(self.img_A, self.meta_vector_B2A)

		if self.is_peason_div:
			print "[*] Initialize with Peason Divergence Loss"
			d_loss_A, lsgan_loss_A = self.build_loss(d_logits_fake_A, d_logits_real_A, -1, 1, 0)
			d_loss_B, lsgan_loss_B = self.build_loss(d_logits_fake_B, d_logits_real_B, -1, 1, 0)
		else:
			print "[*] Initialize with Common Least Square Loss"
			d_loss_A, lsgan_loss_A = self.build_loss(d_logits_fake_A, d_logits_real_A, 0, 1, 1)
			d_loss_B, lsgan_loss_B = self.build_loss(d_logits_fake_B, d_logits_real_B, 0, 1, 1)

		self.g_loss = lsgan_loss_A + lsgan_loss_B + self.build_consistency_loss(self.img_A, g_logits_rectA) + self.build_consistency_loss(self.img_B, g_logits_rectB)
		self.d_loss = d_loss_A + d_loss_B

		self.g_optim = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.g_loss, var_list = self.generatorG.variables + self.generatorF.variables)
		self.d_optim = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.d_loss, var_list = self.discriminatorA.variables + self.discriminatorB.variables)

	def build_loss(self, d_logits_fake, d_logits_real, a, b, c):
		d_loss = 0.5 * tf.reduce_mean(tf.square(d_logits_real - b)) + 0.5 * tf.reduce_mean(tf.square(d_logits_fake - a))
		lsgan_loss = 0.5 * tf.reduce_mean(tf.square(d_logits_fake - c))
		return d_loss, lsgan_loss
	def build_consistency_loss(self, img, g_logits_rect):
		return self.L1_lambda * tf.reduce_mean(tf.abs(img - g_logits_rect))
	def sample(self):
		fake_A = self.generatorG(self.sample_vector, self.sample_meta_data)
		#fake_B = self.generatorF(self.sample_vector, self.sample_meta_data)
		
		fake_A = tf.cast(tf.multiply(tf.add(fake_A, 1.0), 127.5), tf.uint8)
		#fake_B = tf.cast(tf.multiply(tf.add(fake_B, 1.0), 127.5), tf.uint8)


		return fake_A #, fake_B
	def predict(self, mode):
		if mode == 'AtoB':
			output_img = self.generatorG(self.sample_vector, self.sample_meta_data)
		elif mode == 'BtoA':
			output_img = self.generatorF(self.sample_vector, self.sample_meta_data)
		else:
			raise ValueError("Mode Error")

		output_img = tf.cast(tf.multiply(tf.add(output_img, 1.0), 127.5), tf.uint8)

		return output_img
		
