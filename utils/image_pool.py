# -*- coding: utf-8 -*-
# author: K

import numpy as np

# The researcher of Apple found out without memory buffer for training, the gan could not be stablized.
# Paper: https://arxiv.org/pdf/1612.07828
class ImagePool:
	def __init__(self, pool_size):
		self.pool_size = pool_size
		self.num_imgs = 0
		self.images = []
	def query(self, image):
		if self.pool_size == 0:
			return image
		if self.num_imgs < self.pool_size:
			self.images.append(image)
			self.num_imgs += 1
			return image
		if np.random.rand() > 0.5:
			idx = np.random.randint(self.pool_size)
			# You don't have use copy function here unless there is a list copy around
			memory_image = self.images[idx]
			self.images[idx] = image
			return memoery_image
		else:
			return image
		
