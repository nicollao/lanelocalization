import os
import pandas as pd
from sklearn.utils import shuffle

import cv2
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import classdefine

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from define import classdefine

# def trainvalidation(dataPath, step, t=1.0, v=1.1):
def trainvalidation(filepath, filename, step=1, t=0.9, v=1.0):
	''' #앞에서 만든 labels.txt file read '''
	data = pd.read_csv(filepath+filename)
	trainData = []
	valData = []

	''' 
		#define된 classes 정보를 하나씩 가져온다. 
		#i : 0_00_01_01_00
	'''
	for i in to_classnum:

		''' #앞에서 만든 텍스트 파일의 첫행의 Field "lane"의 데이터와 class define값과 비교해서 같은 값이면 '''
		classdata = data.loc[data['lane'] == i]

		''' #class data list를 섞는다. '''
		classdata = shuffle(classdata)

		''' 
			#class data
			#여러개 
			#['/home/shlee/data/01.image/1_x2_02_02_00/20191113/MAH06571_00870.jpg', '1_x2_02_02_00'] 
		'''
		tmp = classdata.get_values()
		''' #classes 마다의 Count를 센다. '''
		count = len(tmp)

		''' 
			#해당 classes리스트의 개수가 500보다 작으면 skip
			#해당 classes가 10000개 이상이면 10000개만  
		'''
		if count < 500:
			continue
		elif count > 10000:
			count = 10000

		print("count2 : ", count)
		''' 
			#10자리에서 올림
			#ex> 
			#2079 => 2100
			#777  => 800
		'''
		count = round(count, -2)

		print('count :: ', count)
		print("count*t : ", count*t)

		''' 
			#trainData, valData 생성..
			#tmp : classes 마다의 list
			#['/home/shlee/data/01.image/1_x2_02_02_00/20191113/MAH06571_00870.jpg', '1_x2_02_02_00']
			#['/home/shlee/data/01.image/1_x2_02_02_00/20191113/MAH06571_00870.jpg', '1_x2_02_02_00'] 
			#trainData에 적재
			#valData에 적재 
			#t : 0.9
			#v : 1.0
			#즉, 셔플 된 데이터를 전체를 다 쓰는게 아니라 이중에서 train set은 90%, val set은 10%만 사용
		'''
		trainData.insert(0, tmp[:int(count*t)])
		valData.insert(0, tmp[int(count*t):int(count*v)])

	result_a = []
	for item in trainData:
		''' #result_a의 같은 리스트에 trainData 추가 '''
		result_a.extend(item)

	result_b = []
	for item in valData:
		''' #result_b의 같은 리스트에 trainData 추가 '''
		result_b.extend(item)

	''' #shuffle한 train set, val set 텍스트 파일 저장 '''
	train_data_path = (filepath + 'train.txt')
	val_data_path = (filepath+'val.txt')

	''' 
		#텍스트 파일에 field name train Text 추가
		#shuffle한 train list train.txt파일에 write
	'''
	train = open(train_data_path, 'w')
	train.write("image,lane\n")
	for item in result_a:
		train.write("%s,%s\n" % (item[0], item[1]))
	train.close()

	''' 
		#텍스트 파일에 field name val Text 추가
		#shuffle한 val list val.txt파일에 write
	'''
	validation = open(val_data_path, 'w')
	validation.write("image,lane\n")
	for item in result_b:
		validation.write("%s,%s\n" % (item[0], item[1]))
	validation.close()

	return result_a, result_b

def maketrain(filepath, filename, count=0.9):

	data = pd.read_csv(filepath+filename)
	trainData = []
	valData = []

	for i in classdefine.to_classnum:
		classdata = data.loc[data['label'] == i]
		classdata = shuffle(classdata)
		# tmp = classdata.image.values
		tmp = classdata.values

		totallength = len(tmp)
		# count = round(count, -2)
		trainData.insert(0, tmp[:int(totallength*count)])
		valData.insert(0, tmp[int(totallength*count):])

	result_a = []
	for item in trainData:
		''' #result_a의 같은 리스트에 trainData 추가 '''
		result_a.extend(item)

	train_data_path = (filepath + 'train.txt')
	train = open(train_data_path, 'w')
	train.write("image,label\n")
	for item in result_a:
		train.write("%s,%s\n" % (item[0], item[1]))
	train.close()

	result_b = []
	for item in valData:
		''' #result_a의 같은 리스트에 trainData 추가 '''
		result_b.extend(item)

	val_data_path = (filepath + 'val.txt')
	f_val = open(val_data_path, 'w')
	f_val.write("image,label\n")
	for item in result_b:
		f_val.write("%s,%s\n" % (item[0], item[1]))
	f_val.close()

	return result_a


def maketrainDummy(filepath, filename, count=100):
	data = pd.read_csv(filepath + filename)
	trainData = []
	traindummyData = []

	for i in classdefine.to_classnum:
		classdata = data.loc[data['label'] == i]
		# classdata = shuffle(classdata)
		# tmp = classdata.image.values
		tmp = classdata.values

		# count = round(count, -2)
		trainData.insert(0, tmp[:int(count)])
		traindummyData.insert(0, tmp[int(count):])

	result_a = []
	for item in trainData:
		''' #result_a의 같은 리스트에 trainData 추가 '''
		result_a.extend(item)

	result_b = []
	for item in traindummyData:
		''' #result_a의 같은 리스트에 trainData 추가 '''
		result_b.extend(item)

	train_data_path = (filepath + 'train.txt')
	train = open(train_data_path, 'w')
	train.write("image,label\n")
	for item in result_a:
		train.write("%s,%s\n" % (item[0], item[1]))
	train.close()

	train_dummydata_path = (filepath + 'train_dummy.txt')
	trainDummy = open(train_dummydata_path, 'w')
	trainDummy.write("image,label\n")
	result_b = shuffle(result_b)
	for item in result_b:
		trainDummy.write("%s,%s\n" % (item[0], item[1]))
	trainDummy.close()

	return result_a, result_b

def makelist(filepath, filename):

	data = pd.read_csv(filepath+filename)
	trainData = []

	for i in classdefine.to_classnum:
		classdata = data.loc[data['label'] == i]
		classdata = shuffle(classdata)
		# tmp = classdata.image.values
		tmp = classdata.values

		# count = round(count, -2)
		trainData.insert(0, tmp)

	result_a = []
	for item in trainData:
		''' #result_a의 같은 리스트에 trainData 추가 '''
		result_a.extend(item)

	train_data_path = (filepath + 'train.txt')
	train = open(train_data_path, 'w')
	train.write("image\n")
	for item in result_a:
		train.write("%s,%s\n" % (item[0], item[1]))
	train.close()

	return result_a

def search(dirname, file_label):
	try:
		''' #/home/shlee/data/01.image/ 경로의 sub directory의 리스트를 만든다.'''
		filenames = os.listdir(dirname)

		''' #sub directory를 하나 씩 가져 온다.'''
		for filename in filenames:

			''' 
				#os.path.join : 경로를 병합하여 새 경로 생성
				#dir name과 filename을 합쳐서 full path 생성
				#dirname : /home/shlee/data/01.image/1_00_01_x3_x2/20191113
				#filename: MAH06576_01980.jpg
				#full_filename : /home/shlee/data/01.image/1_00_01_x3_x2/20191113/MAH06576_01980.jpg

			'''
			full_filename = os.path.join(dirname, filename)

			''' #파일이 directory이면 다시 search 아니면 전체 label 파일에 경로를 write 한다. '''
			if os.path.isdir(full_filename):
				''' #해당 파일이 dir이면 재귀로 search 호출, sub dir이 다 찾아 진다.'''
				search(full_filename, file_label)
			else:
				''' #파일이면 확장자를 구한다.'''
				ext = os.path.splitext(full_filename)[-1]

				''' #확장자 .jpg이면, 이미지이면 '''
				if ext == '.jpg':
					# Image width size가 50 이상인 이미지만 학습 데이터로 사용
					# im = cv2.imread(full_filename)

					# if im.shape[1] < 50:
					# 	# os.remove(full_filename)
					# 	continue
					''' 
						#폴더 중의 classes의 명칭인 폴더의 이름을 가져와서 label을 만든다.
						#['', 'home', 'shlee', 'data', '01.image', '1_00_01_x3_x2', '20191113', 'MAH06576_01980.jpg']
						# -8    -7      -6       -5      -4           -3               -2           -1
						# format 함수로 뒤에 ,1_00_01_x3_x2 붙인다.
					'''
					strInput = ('{},{}\n'.format(full_filename, full_filename.split('/')[-2]))
					# strInput = ('{},{}\n'.format(full_filename,full_filename.split('/')[-2]))

					''' #strInput을 파일에 쓴다.'''
					file_label.writelines(strInput)

	except PermissionError:
		pass

if __name__ == '__main__':

	''' #동영상에서 추출한 이미지의 경로 '''
	input_path = '/home/hclee/data/Data/localization/'

	'''	#makepath.sh에서 생성하지 못한 폴더 전체의 label파일을 생성 한다. '''
	lanePath = os.path.join(input_path, 'labels.txt')
	if not os.path.exists(lanePath):

		file_label = open(lanePath, 'w')
		''' #label 파일을 만들 때, image, lane field 명을 같이 입력 한다. '''
		file_label.write('image,label\n')

		''' 
			#label file을 만들기 위해서 파일을 search해서 sub directory에 label 텍스트를 만드는 함수
			#labels.txt 생성 
		'''
		search(input_path, file_label)
		file_label.close()
	
	''' #train set, validation set shuffle 하는 함수 '''
	maketrain(input_path, 'labels.txt')
