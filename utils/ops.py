# -*- coding: utf-8 -*-
# author: K


def dic_key_invert(dic):
	new_dic = {}
	for key in dic:
		new_dic[dic[key]] = key

	return new_dic
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

def get_keys(dic):
	keys = []
	for key in dic:
		keys.append(int(key))

	return keys	
