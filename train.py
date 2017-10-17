# -*- coding: utf-8 -*-
# author: K


import tensorflow as tf
import numpy as np
from os.path import join, basename
from utils.read_config import read_config
from utils.read_data import read_data_shuffle, read_data_batch
from utils.meta_initializer import meta_initializer
from cycle_gan import CycleGAN
from utils.ops import get_keys, dic_key_invert, get_class_len
from os import environ


#environ["CUDA_VISIBLE_DEVICES"] = "1"

flags = tf.app.flags
flags.DEFINE_string('config_path', '' , 'The path of the config file')
flags.DEFINE_string('data_dir', '', 'The directory of image data')
flags.DEFINE_string('sample_dir', '', 'The directory for sampling test')
flags.DEFINE_string('model_dir', '','The directory of models')
flags.DEFINE_string('output_dir', '', 'The output directory')
FLAGS = flags.FLAGS


def save_model(saver, sess, step, model_dir):
	model_name = "aging-cyclegan"
	saver.save(sess, join(model_dir, model_name), global_step = step)
	print "Model and Step {}".format(step), "Saved"


def train():
	train_configs = read_config(FLAGS.config_path)

	print "[*] Start reading data and shuffling"

	face_names, face_data = read_data_shuffle(FLAGS.data_dir, int(train_configs['width']), int(train_configs['height']))
	test_names, test_data = read_data_shuffle(FLAGS.sample_dir, int(train_configs['width']), int(train_configs['height']))

	print "[*] training set size: ", len(face_names)
	print "[*] validation set size: ", len(test_names)

	print "[*] Start building model"

	cyc_GAN = CycleGAN(train_configs)
	cyc_GAN.build_model()

	saver = tf.train.Saver()

	# run training
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		batch_idx = 0

		for epoch in range(int(train_configs['epochs'])):
			
			batch_size = int(train_configs['batch_size'])

			num_batches = len(face_names) / batch_size

			for step in range(num_batches):
				# Perhaps random shuffle is more suitable, I'll test it later

				if (step + 1) * batch_size > len(face_names):
					img_batch_names = face_names[step * batch_size:]
					img_batch = np.asarray(face_data[step * batch_size:])
					batch_size = len(face_names) - step * batch_size

				else:
					img_batch_names = face_names[step * batch_size : (step + 1) * batch_size]
					img_batch = np.asarray(face_data[step * batch_size : (step + 1) * batch_size])

				age_batch, gender_batch = read_data_batch(img_batch_names)

				prior_batch = np.random.uniform(-1.0, 1.0, [batch_size, int(train_configs['z_dim'])])	

				sess.run(cyc_GAN.eg_optim, feed_dict = {cyc_GAN.img_batch: img_batch, cyc_GAN.age_batch: age_batch, cyc_GAN.gender_batch: gender_batch, 
						cyc_GAN.prior: prior_batch, cyc_GAN.batch_size: batch_size})

				# remove the effect of the discriminator_Z
				#sess.run(cyc_GAN.dz_optim, feed_dict = {cyc_GAN.img_batch: img_batch, cyc_GAN.age_batch: age_batch, cyc_GAN.gender_batch: gender_batch, 
						#cyc_GAN.prior: prior_batch, cyc_GAN.batch_size: batch_size})

				sess.run(cyc_GAN.d_optim, feed_dict = {cyc_GAN.img_batch: img_batch, cyc_GAN.age_batch: age_batch, cyc_GAN.gender_batch: gender_batch, 
						cyc_GAN.prior: prior_batch, cyc_GAN.batch_size: batch_size})

				eg_loss = sess.run(cyc_GAN.eg_loss, feed_dict = {cyc_GAN.img_batch: img_batch, cyc_GAN.age_batch: age_batch, cyc_GAN.gender_batch: gender_batch, 
						cyc_GAN.prior: prior_batch, cyc_GAN.batch_size: batch_size})

			
				progress_rate = float(step) / float(num_batches) * 100.0
	
				print "Epoch:", epoch, "EG Loss:", eg_loss, "Progress: {}%".format(progress_rate)

			if epoch % int(train_configs['save_epoch']) == 0:
				save_model(saver, sess, epoch, FLAGS.model_dir)

			if epoch % int(train_configs['sample_epoch']) == 0:

				age_batch, gender_batch = read_data_batch(test_names)

				fake_imgs = sess.run(cyc_GAN.predict(), feed_dict = {cyc_GAN.sample_batch: test_data, cyc_GAN.age_batch: age_batch, cyc_GAN.gender_batch: gender_batch})

				for i in range(len(test_names)):

					sample_img = sess.run(tf.image.encode_jpeg(tf.squeeze(fake_imgs[i])))

					with open(join(FLAGS.output_dir, "epoch_" + str(epoch) + "_" + basename(test_names[i])), "wb") as fw:
						fw.write(sample_img)

				print "Images Sampled"

def main(_):
	train()

if __name__ == '__main__':
	tf.app.run()
