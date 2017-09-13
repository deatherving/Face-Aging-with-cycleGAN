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
		self.batch_size = int(config['batch_size'])
		
		self.encoder = Encoder(int(config['nef']), int(config['z_dim']), "Encoder")
		self.generator = Generator(int(config['ngf']), self.image_size, "Generator")
		self.discriminator_z = DiscriminatorZ(int(config['ndzf']), "DiscriminatorZ")
		self.discriminator = Discriminator(int(config['ndf']), "Discriminator")
		
		self.width = int(config['width'])
		self.height = int(config['height'])

		self.img_batch = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height, 3])
		self.age_batch = tf.placeholder(tf.float32, [self.batch_size, int(config['age_segment'])])
		self.gender_batch = tf.placeholder(tf.float32, [self.batch_size, int(config['gender_segment'])])
		self.prior = tf.placeholder(tf.float32, [self.batch_size, int(config['z_dim'])])	

		self.is_peason_div = int(config['is_peason'])

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
		if self.is_peason_div:
			pass
		else:
			pass

	def build_loss(self, d_logits_fake, d_logits_real, a, b, c):
		d_loss = 0.5 * tf.reduce_mean(tf.square(d_logits_real - b)) + 0.5 * tf.reduce_mean(tf.square(d_logits_fake - a))
		lsgan_loss = 0.5 * tf.reduce_mean(tf.square(d_logits_fake - c))
		return d_loss, lsgan_loss
	def build_consistency_loss(self, img, g_logits_rect):
		return self.L1_lambda * tf.reduce_mean(tf.abs(img - g_logits_rect))
	def convert(self, inputs):
		inputs_shape = inputs.get_shape().as_list()
		base = tf.ones([inputs_shape[0], self.image_size / 2, self.image_size / 2, inputs_shape[-1]])
		return inputs * base


