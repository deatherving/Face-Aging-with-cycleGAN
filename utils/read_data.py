# -*- coding: utf-8 -*-
# author: K

import numpy as np
from os.path import basename, join, splitext
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
	return img_array / 127.5 - 1.0

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

def read_data_name(directory):
	img_names = []

	for root, subs, files in walk(directory):
		for f in files:
			tmpstr = join(root, f)
			img_names.append(tmpstr)

	return img_names

def make_one_hot(label, segment_num, range_segment):
	if len(range_segment) != segment_num:
		raise "Segment num is not matching segment range"

	shape = [segment_num]

	res = np.full(shape, -1, dtype = np.float32)

	idx = 0
	for r in range_segment:
		if r[0] >= r[1]:
			raise "Segment Range Error"
		if r[0] <= label < r[1]:
			res[idx] = 1
		idx += 1

	return res

def read_data_batch(names, width, height, age_segment_num = 10, gender_segment_num = 2):
	img_data = []
	age_data = []
	gender_data = []
	for item in names:
			bname = basename(item)
			age = int(splitext(bname)[0].split('_')[0])
			gender = int(splitext(bname)[0].split('_')[1])
			# reserve race for future use
			race = int(splitext(bname)[0].split('_')[2])

			age_seg = make_one_hot(age, age_segment_num, 
					[(0, 6),
					 (6, 11),						
					 (11, 16),
					 (16, 21),
					 (21, 31),
					 (31, 41),
					 (41, 51),
					 (51, 61),
					 (61, 71),
					 (71, 81),
					])
			gender_seg = make_one_hot(gender, gender_segment_num,
					[(0, 1),
					 (1, 2)
					])
			

			age_data.append(age_seg)
			gender_data.append(gender_seg)
			img_data.append(pixel_normalize(img_to_array(load_img(item, target_size = (width, height)))))
				
	# return the image batch, one-hot age batch and one-hot gender batch
	return np.asarray(img_data), np.asarray(age_data), np.asarray(gender_data)


