# -*- coding: utf-8 -*-
# author: K

import tensorflow as tf
import numpy as np
from os.path import join
from utils.read_config import read_config
from utils.meta_initializer import meta_initializer
from cycle_gan import CycleGAN

flags = tf.app.flags
flags.DEFINE_string('config_path', '', 'The path of the config file')
flags.DEFINE_string('model_path', '', 'The path of the model')
flags.DEFINE_string('output_dir', '', 'The output directory')
FLAGS = flags.FLAGS

def main(_):
	test()

if __name__ == '__main__':
	tf.app.run()
