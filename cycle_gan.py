# -*- coding: utf-8 -*-
# author: K

import tensorflow as tf
import numpy as np
from architecture.generator import Generator
from architecture.discriminator import Discriminator, DiscriminatorZ
from architecture.encoder import Encoder


class CycleGAN:
	def __init__(self, config):
		# initializing 2 generators and 2 discriminators
		# We don't use ImagePool. Perhaps later
		self.lr = float(config['lr'])
		self.image_size = int(config['image_align_size'])
		self.beta1 = float(config['beta1'])		

		self.encoder = Encoder(int(config['nef']), int(config['z_dim']), "Encoder")
		self.generator = Generator(int(config['ngf']), self.image_size, "Generator")
		self.discriminator_z = DiscriminatorZ(int(config['ndzf']), "DiscriminatorZ")
		self.discriminator = Discriminator(int(config['ndf']), "Discriminator")
		
		self.width = int(config['width'])
		self.height = int(config['height'])
		self.decay_rate = float(config['decay_rate'])


		self.batch_size = tf.placeholder(tf.int32)
		self.img_batch = tf.placeholder(tf.float32, [None, self.width, self.height, 3])
		self.age_batch = tf.placeholder(tf.float32, [None, int(config['age_segment'])])
		self.gender_batch = tf.placeholder(tf.float32, [None, int(config['gender_segment'])])
		self.prior = tf.placeholder(tf.float32, [None, int(config['z_dim'])])	
		self.sample_batch = tf.placeholder(tf.float32, [None, self.width, self.height, 3])


		self.is_peason_div = int(config['is_peason'])
		self.loss_control = float(config['loss_control'])

	def build_model(self):
		# The encoder logits
		z_logits = self.encoder(self.img_batch)

		# The generator logits
		g_logits = self.generator(z_logits, self.age_batch, self.gender_batch)

		# The discriminator_z logits
		dz_logits_fake = self.discriminator_z(z_logits)
		dz_logits_real = self.discriminator_z(self.prior)
		
		# The discriminator logits
		d_logits_fake = self.discriminator(g_logits, self.convert(self.age_batch), self.convert(self.gender_batch))
		d_logits_real = self.discriminator(self.img_batch, self.convert(self.age_batch), self.convert(self.gender_batch))

		# build loss
		'''
		dz_loss_prior = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dz_logits_real, labels = tf.ones_like(dz_logits_real)))
		dz_loss_z = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dz_logits_fake, labels = tf.zeros_like(dz_logits_fake)))
		ez_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dz_logits_fake, labels = tf.ones_like(dz_logits_fake)))

		d_loss_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_real, labels = tf.ones_like(d_logits_real)))
		d_loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake, labels = tf.zeros_like(d_logits_fake)))
		g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake, labels = tf.ones_like(d_logits_fake)))
		'''

		if self.is_peason_div:
			dz_loss, zlsgan_loss = self.build_loss(dz_logits_fake, dz_logits_real, -1, 1, 0)
			d_loss, lsgan_loss = self.build_loss(d_logits_fake, d_logits_real, -1, 1, 0)
		else:
			dz_loss, zlsgan_loss = self.build_loss(dz_logits_fake, dz_logits_real, 0, 1, 1)
			d_loss, lsgan_loss = self.build_loss(d_logits_fake, d_logits_real, 0, 1, 1)
		# The L1 Loss
		l1_loss = self.build_l1_loss(self.img_batch, g_logits)

		# The Total Variation of Image, to denoising the generated image, https://www.tensorflow.org/api_docs/python/tf/image/total_variation
		tv_loss = tf.reduce_sum(tf.image.total_variation(g_logits))

		# The loss is slightly different from the Paper cause the TV loss is too large and forces the generated image to a one-channel-like picture
		self.eg_loss = l1_loss + self.loss_control * lsgan_loss #+  self.loss_control * zlsgan_loss + self.loss_control * tv_loss
		self.d_loss = d_loss
		self.dz_loss = dz_loss

		# learning rate decay
		#self.global_step = tf.Variable(0, trainable = False)
		#learning_rate = tf.train.exponential_decay(self.lr, self.global_step, decay_steps = 256, decay_rate = self.decay_rate, staircase = True)

		# Still hesitate

		self.eg_optim = tf.train.AdamOptimizer(
			learning_rate = self.lr, beta1 = self.beta1).minimize(self.eg_loss, var_list = self.encoder.variables + self.generator.variables)

		self.d_optim = tf.train.AdamOptimizer(
			learning_rate = self.lr, beta1 = self.beta1).minimize(self.d_loss, var_list = self.discriminator.variables)

		self.dz_optim = tf.train.AdamOptimizer(
			learning_rate = self.lr, beta1 = self.beta1).minimize(self.dz_loss, var_list = self.discriminator_z.variables)


	def build_loss(self, d_logits_fake, d_logits_real, a, b, c):
		d_loss = 0.5 * tf.reduce_mean(tf.square(d_logits_real - b)) + 0.5 * tf.reduce_mean(tf.square(d_logits_fake - a))
		lsgan_loss = 0.5 * tf.reduce_mean(tf.square(d_logits_fake - c))
		return d_loss, lsgan_loss

	def build_l1_loss(self, img, g_logits):
		return tf.reduce_mean(tf.abs(img - g_logits))

	def convert(self, inputs):
		inputs_shape = inputs.get_shape().as_list()
		r_inputs = tf.reshape(inputs, [self.batch_size, 1, 1, inputs_shape[-1]])
		base = tf.ones([self.batch_size, self.image_size / 2, self.image_size / 2, inputs_shape[-1]])
		return r_inputs * base

	def predict(self):
		z_logits = self.encoder(self.sample_batch)

		fake_img = self.generator(z_logits, self.age_batch, self.gender_batch)

		fake_img = tf.cast(tf.multiply(tf.add(fake_img, 1.0), 127.5), tf.uint8)

		return fake_img
		
		
		

