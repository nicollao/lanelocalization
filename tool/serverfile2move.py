import random
import shutil
import os
import threading
import math
import os


def checktagging(tag):
	d = {
		# "0_00_01_01_00":True,
		# "0_00_01_02_00":True,
		# "1_00_01_02_00":True,
		# "0_00_01_x3_00":True,
		# "0_00_02_03_00":True,
		# "1_00_01_x3_00":True,
		# "0_00_02_x4_00":True,
		"1_00_02_x4_00":True,
		# "0_00_03_x5_00":True,
		# "1_00_03_x5_00":True,
	}

	for i, (key, value) in enumerate(d.items()):
		# print(i, key, value)
		if tag == key:
			return True
	return False


def search(dirname):
	filenames = os.listdir(dirname)
	for filename in filenames:
		full_filename = os.path.join(dirname, filename)
		if os.path.isdir(full_filename):
			search(full_filename)
		else:
			return full_filename

def copysrc2dst(data, start, end):
	for i in range(start, end):
		line = data[i]		
		temp = line.split(',')

		if checktagging(temp[1].rstrip('\n')):
			src = temp[0]
			# dst = temp[0].replace("crop-300", "traindata-20190418-type2")
			dst_tmp = temp[0].split('/')
			dst = "/home/data/highway_lane/newData/traindata-20160418-type2/" + dst_tmp[7] + "/" + dst_tmp[8] + "/" + dst_tmp[9] + "/" + dst_tmp[10]

			# dstdir = os.path.dirname(dst) + '/'
			# print 'src : %s, dstdir : %s' % (src, dstdir)

			if os.path.exists(os.path.dirname(os.path.abspath(dst))) == False:
				os.makedirs(os.path.dirname(os.path.abspath(dst)))
			shutil.copy2(src, dst)


def movefile(path, output):
	# path = '/home/data/highway_lane/20190411-1/2/'
	# output = '/home/data/highway_lane/20190411-1/'

	directories = os.listdir(path)
	for dir in directories:
		full_path = os.path.join(path, dir)
		sub_directories = os.listdir(full_path)

		for filename in sub_directories:
			tmp = filename.split('_')

			src = os.path.join(full_path, filename)
			dst = os.path.join(output, tmp[0], tmp[1], tmp[2], tmp[3])

			if os.path.exists(os.path.dirname(os.path.abspath(dst))) == False:
				os.makedirs(os.path.dirname(os.path.abspath(dst)))
			if os.path.getsize(src) > 0:
				os.rename(src, dst)
			else:
				os.remove(src)

if __name__ == '__main__':

	# path2 = '/home/data/highway_lane/20190411-1/2'
	path3 = '/home/data/highway_lane/20190411-1/3'
	path4 = '/home/data/highway_lane/20190411-1/4'
	path5 = '/home/data/highway_lane/20190411-1/5'
	path6 = '/home/data/highway_lane/20190411-1/6'
	path7 = '/home/data/highway_lane/20190411-1/7'
	path8 = '/home/data/highway_lane/20190411-1/8'

	output = '/home/data/highway_lane/20190411-1/'

	# th2 = threading.Thread(target=movefile, args=(path2, output))
	# th2.start()
	th3 = threading.Thread(target=movefile, args=(path3, output))
	th3.start()
	th4 = threading.Thread(target=movefile, args=(path4, output))
	th4.start()
	th5 = threading.Thread(target=movefile, args=(path5, output))
	th5.start()
	th6 = threading.Thread(target=movefile, args=(path6, output))
	th6.start()
	th7 = threading.Thread(target=movefile, args=(path7, output))
	th7.start()
	th8 = threading.Thread(target=movefile, args=(path8, output))
	th8.start()

	# directories = os.listdir(path)
	# for dir in directories:
	# 	full_path = os.path.join(path, dir)
	#
	# 	thread_id[i] = threading.Thread(target=copysrc2dst, args=(full_path, output))
	# 	thread_id[i].start()

	# 	sub_directories = os.listdir(full_path)
	#
	# 	for filename in sub_directories:
	# 		tmp = filename.split('_')
	#
	# 		src = os.path.join(full_path, filename)
	# 		dst = os.path.join(output, tmp[0], tmp[1], tmp[2], tmp[3])
	#
	# 		if os.path.exists(os.path.dirname(os.path.abspath(dst))) == False:
	# 			os.makedirs(os.path.dirname(os.path.abspath(dst)))
	# 		os.rename(src, dst)
	# 		shutil.move(src, dst)
	#
	# test = open('/home/data/highway_lane/newData/crop-300/labels_total_20190418.txt', 'r')
	# lines = test.readlines()
	# /home/data/highway_lane/20190411-1

	# num_lines = open('/home/data/highway_lane/newData/crop-300/labels_total_20190418.txt').read().count('\n')
	# offset = int(num_lines / 8)
	# thread_id = {}
	#
	# for i in range(0, 8):
	# 	start_idx = (offset * i)
	# 	end_idx = (offset * i) + offset
	# 	print(start_idx, end_idx)
	#
	# 	thread_id[i] = threading.Thread(target=copysrc2dst, args=(lines, start_idx, end_idx))
	# 	thread_id[i].start()


		# thread_id[i] = threading.Thread(target=mdbM.save_result_from_smb, args=(data, path, start_idx, end_idx))
		# thread_id[i].start()
