# -*- coding: utf-8 -*-
# author: K

from sys import argv
from os import walk
from os.path import join, basename
import TencentYoutuyun
from PIL import Image

def init_face_detect():
	appid = '10077717'
	secret_id = 'AKIDlY6PSEgEQM2RbU8SYNvPffOr38wUNMO9'
	secret_key = 'nMn1yf9NVNVdPwDQZT7Um0zwo44tRq89'
	userid = '774541291@qq.com'
	end_point = TencentYoutuyun.conf.API_YOUTU_END_POINT
	return TencentYoutuyun.YouTu(appid, secret_id, secret_key, userid, end_point)

def detect_face(youtu, path):

	ret = youtu.DetectFace(path)

	return ret[u'face']


def crop_head(faces, f, target_dir):

	idx = 0

	for each in faces:
		if u'x' not in each or u'y' not in each:
			continue

		x_axis = each[u'x']
		y_axis = each[u'y']

		square = each[u'height']

		image = Image.open(f)

		extracted = image.crop((x_axis - 5, y_axis - 5, x_axis + square + 5, y_axis + square + 5))

		extracted.save(join(target_dir, str(idx) + "_" + basename(f)))

		idx += 1


if __name__ == '__main__':
	source_dir = argv[1]
	target_dir = argv[2]

	youtu = init_face_detect()

	filenames = []
	for root, dirs, files in walk(source_dir):
		for f in files:
			if f.endswith("jpeg") or f.endswith("jpg") or f.endswith("png"):
				filenames.append(join(root, f))
	
	for f in filenames:
		faces = detect_face(youtu, f)

		crop_head(faces, f, target_dir)
