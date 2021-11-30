import random
import shutil
import os
import threading
import math
from datetime import datetime


def checktagging(tag):
	d = {
		# "0_00_01_01_00":True,
		# "0_00_01_02_00":True,
		# "1_00_01_02_00":True,
		# "0_00_01_x3_00":True,
		# "0_00_02_03_00":True,
		# "1_00_01_x3_00":True,
		# "0_00_02_x4_00":True,
		"0_00_03_04_00":True,
		# "1_00_02_x4_00":True,
		# "0_00_03_x5_00":True,
		# "1_00_03_x5_00":True,
	}

	for i, (key, value) in enumerate(d.items()):
		# print(i, key, value)
		if tag == key:
			return True
	return False

def copysrc2dst(data, start, end, output):
	for i in range(start, end):

		# if i % 5000 == 0:
		# 	print (datetime.now())

		line = data[i]
		temp = line.split(',')

		# if checktagging(temp[1].rstrip('\n')) == False:
		# 	continue
		# if temp[0].find('traindata-20190424-type2') > 0:
		# 	print (temp[0])

		src = temp[0]
		# if len(temp[0].split('/')) == 10:
		dst_tmp = temp[0].split('/')
			# dst = "/home/data/highway_lane/newData/"
		dst = os.path.join(output, dst_tmp[8], dst_tmp[9], dst_tmp[10])
		# print ('dst : %s', dst)
			# dst = "/home/data/highway_lane/newData/01_TrainData" + output + dst_tmp[7] + "/" + dst_tmp[8] + "/" + dst_tmp[9]
		# else:
		# 	dst = temp[0].replace("/crop-300/", output)

		# if temp[0].find("/crop-300") > 0:
			# dst = temp[0].replace("/crop-300/", output)
		# else:
			# print ("error !!!!!")

		# if temp[0].find("/crop-300-type2/") > 0:
			# dst = temp[0].replace("/crop-300-type2/", output)

		# dst_tmp = temp[0].split('/')
		# dst = os.path.join(output, dst_tmp[6], dst_tmp[7], dst_tmp[8], dst_tmp[9], dst_tmp[10])
		# dst = os.path.join(dstdir, dst_tmp[9])
		# dst = output + dst_tmp[7] + "/" + dst_tmp[8] + "/" + dst_tmp[9] + "/" + dst_tmp[10]

		# dstdir = os.path.dirname(dst) + '/'
		# print 'src : %s, dstdir : %s' % (src, dstdir)

		if os.path.exists(os.path.dirname(os.path.abspath(dst))) == False:
			os.makedirs(os.path.dirname(os.path.abspath(dst)))
		try:
			shutil.copy2(src, dst)
		except FileNotFoundError:
			print (src)

if __name__ == '__main__':

	filename = '../checkpoint/20190508-no-8-best/labels-valSet-201905221936.txt'
	output = '/home/data/highway_lane/newData/01_TrainData/012-valSet-20190522-best/'

	# filename = '../labels-trainSet-201904301840.txt'
	# output = '/home/data/highway_lane/newData/01_TrainData/007-trainSet-class03-01-20190430/'
	# filename = '../labels-valSet-201904301840.txt'
	# output = '/home/data/highway_lane/newData/01_TrainData/007-valSet-class03-01-20190430/'
	test = open(filename, 'r')
	lines = test.readlines()

	num_lines = open(filename).read().count('\n')
	offset = int(num_lines / 8)
	thread_id = {}

	# thread_id = threading.Thread(target=copysrc2dst, args=(lines, 1, num_lines, output))
	# thread_id = threading.Thread(target=copysrc2dst, args=(lines, 0, num_lines, output))
	# thread_id.start()

	for i in range(0, 8):
		start_idx = (offset * i)

		if start_idx == 0:
			start_idx = 1

		end_idx = (offset * i) + offset
		print(start_idx, end_idx)

		thread_id[i] = threading.Thread(target=copysrc2dst, args=(lines, start_idx, end_idx, output))
		thread_id[i].start()

		# thread_id[i] = threading.Thread(target=mdbM.save_result_from_smb, args=(data, path, start_idx, end_idx))
		# thread_id[i].start()


	# for line in train:
	# 	temp = line.split(',')
	# 	src = temp[0]
	# 	dst = temp[0].replace("trainSet_20190403", "trainSet_20190404")
	# 	# dst.replace("trainSet_20190403", "trainSet_20190404")
	# 	dstdir = os.path.dirname(dst)
	# 	print 'src : %s, dstdir : %s' % (src, dstdir)

	# 	if os.path.exists(os.path.dirname(os.path.abspath(dstdir))) == False:
	# 	    os.makedirs(os.path.dirname(os.path.abspath(dstdir)))

	# 	# i = i + 1
	# 	# if i > 10:
	# 	# 	break;
	# 	shutil.copy2('/dir/file.ext', '/new/dir/newname.ext')


# lines = open('0_00_01_01_00.txt').readlines()
# random.shuffle(lines)

# train.writelines(lines[:5000])
# validation.writelines(lines[5000:5500])

# lines = open('0_00_01_02_00.txt').readlines()
# random.shuffle(lines)

# train.writelines(lines[:10000])
# validation.writelines(lines[10000:11000])

# lines = open('1_00_01_02_00.txt').readlines()
# random.shuffle(lines)

# train.writelines(lines[:10000])
# validation.writelines(lines[10000:11000])

# lines = open('0_00_02_03_00.txt').readlines()
# random.shuffle(lines)

# train.writelines(lines[:20000])
# validation.writelines(lines[20000:22000])

# lines = open('0_00_01_x3_00.txt').readlines()
# random.shuffle(lines)

# train.writelines(lines[:10000])
# validation.writelines(lines[10000:11000])

# lines = open('1_00_01_x3_00.txt').readlines()
# random.shuffle(lines)

# train.writelines(lines[:20000])
# validation.writelines(lines[20000:22000])

# lines = open('0_00_02_x4_00.txt').readlines()
# random.shuffle(lines)

# train.writelines(lines[:20000])
# validation.writelines(lines[20000:22000])

# lines = open('1_00_02_x4_00.txt').readlines()
# random.shuffle(lines)

# train.writelines(lines[:30000])
# validation.writelines(lines[30000:33000])

# lines = open('0_00_03_x5_00.txt').readlines()
# random.shuffle(lines)

# train.writelines(lines[:30000])
# validation.writelines(lines[30000:33000])

# lines = open('1_00_03_x5_00.txt').readlines()
# random.shuffle(lines)

# train.writelines(lines[:5000])
# validation.writelines(lines[5000:5500])