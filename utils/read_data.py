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

def read_data_corpus(directory, width, height):
	img_corpus = {}
	for root, subs, files in walk(directory):
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

def read_data(directory, width, height):
	img_data = []

	for root, subs, files in walk(directory):
		for f in files:
			tmpstr = join(root, f)
			img_data.append(pixel_normalize(img_to_array(load_img(tmpstr, target_size = (width, height)))))

	return np.asarray(img_data)

