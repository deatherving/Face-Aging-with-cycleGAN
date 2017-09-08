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
flags.DEFINE_string('class_config', '', 'The path of the class config')
flags.DEFINE_string('data_dir', '', 'The directory of image data')
flags.DEFINE_string('sample_dir', '', 'The directory for sampling output')
flags.DEFINE_string('model_dir', '','The directory of models')
FLAGS = flags.FLAGS



def save_model(saver, sess, step, model_dir):
	model_name = "cyclegan"
	saver.save(sess, join(model_dir, model_name), global_step = step)
	print "Model and Step {}".format(step), "Saved"


def train():
	train_configs = read_config(FLAGS.config_path)
	class_index = dic_key_invert(read_config(FLAGS.class_config))

	# Read Data
	img_corpus = read_data_corpus(FLAGS.data_dir, int(train_configs['width']), int(train_configs['height']))
	class_len = get_class_len(img_corpus)

	# Init cycle GAN
	cyc_GAN = CycleGAN(train_configs)
	cyc_GAN.build_model()


	saver = tf.train.Saver()
	
	# run training
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
				
		# We pick two class and samples within randomly

		for step in range(int(train_configs['train_steps_total'])):

			img_list = []
			
			class_range = get_keys(class_index)
			
			class_ids = np.random.choice(class_range, 2, replace = False)

			class_ids = np.sort(class_ids)

			meta_vector_A2B = meta_initializer(class_ids[1], [int(train_configs['width']), int(train_configs['height'])])
			
			meta_vector_B2A = meta_initializer(class_ids[0], [int(train_configs['width']), int(train_configs['height'])])

			for class_id in class_ids:

				random_ids = np.random.choice(class_len[class_index[class_id]], int(train_configs['batch_size']), replace = False)
				
				ims = []

				for r_id in random_ids:
					im = img_corpus[class_index[class_id]][r_id]
					ims.append(im)

				img_list.append(np.asarray(ims))

			sess.run(cyc_GAN.g_optim, 
				feed_dict = {cyc_GAN.img_A: img_list[0], cyc_GAN.img_B: img_list[1], cyc_GAN.meta_vector_A2B: meta_vector_A2B, cyc_GAN.meta_vector_B2A: meta_vector_B2A})
				
			sess.run(cyc_GAN.d_optim,
				feed_dict = {cyc_GAN.img_A: img_list[0], cyc_GAN.img_B: img_list[1], cyc_GAN.meta_vector_A2B: meta_vector_A2B, cyc_GAN.meta_vector_B2A: meta_vector_B2A})

			g_loss, d_loss = sess.run([cyc_GAN.g_loss, cyc_GAN.d_loss], 
				feed_dict = {cyc_GAN.img_A: img_list[0], cyc_GAN.img_B: img_list[1], cyc_GAN.meta_vector_A2B: meta_vector_A2B, cyc_GAN.meta_vector_B2A: meta_vector_B2A})

			print "Steps: ", step, "gloss", g_loss, "dloss", d_loss

			
			if step % int(train_configs['sample_step']) == 0 and step != 0:
				fake_A = sess.run(cyc_GAN.sample(), feed_dict = {cyc_GAN.sample_vector: img_list[0], cyc_GAN.sample_meta_data: meta_vector_A2B})
				
				image_A = sess.run(tf.image.encode_jpeg(tf.squeeze(fake_A)))

				with open(FLAGS.sample_dir + "/sample%d_A.jpeg" % step, "wb") as f:
					f.write(image_A)
				#with open(FLAGS.sample_dir + "/sample%d_B.jpeg" % step, "wb") as f:
					#f.write(image_B)
			if step % int(train_configs['save_step']) == 0 and step != 0:
					
				save_model(saver, sess, step, FLAGS.model_dir)

def main(_):
	train()

if __name__ == '__main__':
	tf.app.run()
