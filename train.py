# -*- coding: utf-8 -*-
# author: K



import tensorflow as tf
import numpy as np
from os.path import join
from utils.read_config import read_config
from utils.read_data import read_data_corpus, load_img, img_to_array, pixel_normalize
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

	cyc_GAN = CycleGAN(train_configs)
	cyc_GAN.build_model()


	saver = tf.train.Saver()
	
	# run training
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
				
		# We pick two class and samples within randomly


def main(_):
	train()

if __name__ == '__main__':
	tf.app.run()
