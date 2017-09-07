# -*- coding: utf-8 -*-
# author: K

import tensorflow as tf
import numpy as np
from os.path import join
from utils.read_config import read_config
from utils.read_data import read_data
from utils.meta_initializer import meta_initializer
from cycle_gan import CycleGAN

flags = tf.app.flags
flags.DEFINE_string('config_path', '', 'The path of the config file')
flags.DEFINE_string('class_config', '', 'The path of the class config')
flags.DEFINE_string('data_dir', '' ,'The image data directory')
flags.DEFINE_string('f', '', 'The condition from class')
flags.DEFINE_string('t', '', 'The condition to class')
flags.DEFINE_string('model_path', '', 'The path of the model')
flags.DEFINE_string('output_dir', '', 'The output directory')
FLAGS = flags.FLAGS


def inference():
	configs = read_config(FLAGS.config_path)
	class_ids = read_config(FLAGS.class_config)

	img_data = read_data(FLAGS.data_dir, int(configs['width']), int(configs['height']))

	cyc_GAN = CycleGAN(configs)
	cyc_GAN.build_model()

	# You must define saver after you build a tensorflow graph
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, FLAGS.model_path)

		meta_data = meta_initializer(int(class_ids[FLAGS.t]), [int(configs['width']), int(configs['height'])])

		idx = 0


		for img in img_data:

			img_vec = np.expand_dims(img, axis = 0)

			if int(class_ids[FLAGS.f]) < int(class_ids[FLAGS.t]):
				mode = "AtoB"
			else:
				mode = "BtoA"

			output_img = sess.run(cyc_GAN.predict(mode), feed_dict = {cyc_GAN.sample_vector: img_vec, cyc_GAN.sample_meta_data: meta_data})
	
			output_img = sess.run(tf.image.encode_jpeg(tf.squeeze(output_img)))

			with open(join(FLAGS.output_dir, str(idx) + '.jpg'), 'wb') as f:
				f.write(output_img)

			idx += 1

def main(_):
	inference()

if __name__ == '__main__':
	tf.app.run()
