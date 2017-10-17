# -*- coding: utf-8 -*-
# author: K

import tensorflow as tf
import numpy as np
from os import environ
from os.path import join, basename
from utils.read_config import read_config
from utils.read_data import read_data_shuffle, read_data_batch
from utils.meta_initializer import meta_initializer
from cycle_gan import CycleGAN

environ['CUDA_VISIBLE_DEVICES'] = '1'

flags = tf.app.flags
flags.DEFINE_string('config_path', '', 'The path of the config file')
flags.DEFINE_string('data_dir', '' ,'The image data directory')
flags.DEFINE_string('model_path', '', 'The path of the model')
flags.DEFINE_string('output_dir', '', 'The output directory')
FLAGS = flags.FLAGS


def inference():
	configs = read_config(FLAGS.config_path)

	face_names, face_data = read_data_shuffle(FLAGS.data_dir, int(configs['width']), int(configs['height']))

	cyc_GAN = CycleGAN(configs)
	cyc_GAN.build_model()

	# You must define saver after you build a tensorflow graph
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, FLAGS.model_path)
		
		age_batch, gender_batch = read_data_batch(face_names)
		
		fake_imgs = sess.run(cyc_GAN.predict(), feed_dict = {cyc_GAN.sample_batch: face_data, cyc_GAN.age_batch: age_batch, cyc_GAN.gender_batch: gender_batch})

		for i in range(len(face_names)):
			pred_img = sess.run(tf.image.encode_jpeg(fake_imgs[i]))
			
			with open(join(FLAGS.output_dir, basename(face_names[i])), "wb") as fw:
				fw.write(pred_img)

def main(_):
	inference()

if __name__ == '__main__':
	tf.app.run()
