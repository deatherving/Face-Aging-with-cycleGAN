# -*- coding: utf-8 -*-
# author: K



import tensorflow as tf
import numpy as np
from os.path import join
from utils.read_config import read_config
from utils.read_data import read_data_name, read_data_batch
from utils.meta_initializer import meta_initializer
from cycle_gan import CycleGAN
from utils.ops import get_keys, dic_key_invert, get_class_len

flags = tf.app.flags
flags.DEFINE_string('config_path', '' , 'The path of the config file')
flags.DEFINE_string('data_dir', '', 'The directory of image data')
flags.DEFINE_string('sample_dir', '', 'The directory for sampling output')
flags.DEFINE_string('model_dir', '','The directory of models')
FLAGS = flags.FLAGS



def save_model(saver, sess, step, model_dir):
	model_name = "aging-cyclegan"
	saver.save(sess, join(model_dir, model_name), global_step = step)
	print "Model and Step {}".format(step), "Saved"


def train():
	train_configs = read_config(FLAGS.config_path)
	face_names = read_data_name(FLAGS.data_dir)

	cyc_GAN = CycleGAN(train_configs)
	cyc_GAN.build_model()

	saver = tf.train.Saver()
	
	# run training
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		batch_size = int(train_configs['batch_size'])

		num_batches = len(face_names) / batch_size

		for epoch in range(int(train_configs['epochs'])):

			for step in range(num_batches):
				# Perhaps random shuffle is more suitable, I'll test it later
				img_batch_names = np.random.choice(face_names, batch_size)

				img_batch, age_batch, gender_batch = read_data_batch(img_batch_names, int(train_configs['width']), int(train_configs['height']))

				prior_batch = np.random.uniform(-1, 1, [batch_size, int(train_configs['z_dim'])]	

				sess.run(cyc_GAN.eg_optim, feed_dict = {cyc_GAN.img_batch: img_batch, cyc_GAN.age_batch: age_batch, cyc_GAN.gender_batch: gender_batch, 
						cyc_GAN.prior: prior_batch})

				sess.run(cyc_GAN.dz_optim, feed_dict = {cyc_GAN.img_batch: img_batch, cyc_GAN.age_batch: age_batch, cyc_GAN.gender_batch: gender_batch, 
						cyc_GAN.prior: prior_batch})

				sess.run(cyc_GAN.d_optim, feed_dict = {cyc_GAN.img_batch: img_batch, cyc_GAN.age_batch: age_batch, cyc_GAN.gender_batch: gender_batch, 
						cyc_GAN.prior: prior_batch})

				eg_loss = sess.run(cyc_GAN.eg_loss, feed_dict = {cyc_GAN.img_batch: img_batch, cyc_GAN.age_batch: age_batch, cyc_GAN.gender_batch: gender_batch, 
						cyc_GAN.prior: prior_batch})
						
				print "Epoch:", epoch, "EG Loss:", eg_loss

			if epoch % sample_epoch == 0 and epoch != 0:



				
				sample_img = sess.run(cyc_GAN.predict(), feed_dict = {}



def main(_):
	train()

if __name__ == '__main__':
	tf.app.run()
