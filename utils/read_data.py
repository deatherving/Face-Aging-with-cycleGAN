# -*- coding: utf-8 -*-
# author: K

import numpy as np
from os.path import basename, join
from os import walk
from PIL import Image


def load_img(path, grayscale = False, target_size = None):
        img = Image.open(path)

        if grayscale:
                if img.mode != 'L':
                        img = img.convert('L')
        else:
                if img.mode != 'RGB':
                        img = img.convert('RGB')

        if target_size:
                wh_tuple = (target_size[1], target_size[0])
                if img.size != wh_tuple:
                        img = img.resize(wh_tuple)
        return img

def img_to_array(img):
	x = np.asarray(img, dtype = np.float32)
	return x

def pixel_normalize(img_array):
	return img_array / 127.5 - 1

def read_data_corpus(path, width, height):
	img_corpus = {}
	for root, subs, files in walk(path):
		if len(subs) != 0:
			for sub in subs:
				img_corpus[sub] = []
		for key in img_corpus:
			for f in files:
				tmpstr = join(root, f)
				if key in tmpstr:
					img_corpus[key].append(pixel_normalize(img_to_array(load_img(tmpstr, target_size = (width, height)))))
	# simple preprocessing. We could apply flip right here

	return img_corpus


def get_class_index(img_corpus):
	idx = 0
	
	class_map = {}
	
	for key in img_corpus:
		class_map[idx] = key
		idx += 1
	return class_map

def get_class_len(img_corpus):
	class_len = {}

	for key in img_corpus:
		class_len[key] = len(img_corpus[key])

	return class_len
