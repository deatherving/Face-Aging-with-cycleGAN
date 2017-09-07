# -*- coding: utf-8 -*-
# author: K

import numpy as np


def meta_initializer(meta_value, shape):
	shape = [1] + shape + [1]
	return np.full(shape, meta_value / 10.0, dtype = np.float32)
