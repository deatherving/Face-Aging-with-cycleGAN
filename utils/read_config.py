# -*- coding: utf-8 -*-
# author: K

from yaml import safe_load, dump

def read_config(config_path):
	with open(config_path, "r") as fread:
		config = safe_load(fread)
	return config

	
