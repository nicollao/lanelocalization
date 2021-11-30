from datetime import datetime

import pandas as pd
from flask import Flask

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

import myconfig
import imagedefine

import os
import threading

app = Flask(__name__)
model = None
prev_image_array = None

def crop(image, top_percent, bottom_percent):
    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
    return image[top:bottom, :]

def resize(image, new_dim):
    return scipy.misc.imresize(image, new_dim)


driver = {
    # '홍성진': (0, 470, 950),
    # '고광용': (0, 600, 980),
    # '서성훈': (0, 600, 1050),
    # '정철훈': (0, 540, 1000),
    # '이은태': (0, 500, 1010),
    # '윤영래': (0, 460, 950),
    # '김숭섭': (0, 580, 1030),
    # '오주인': (0, 600, 1030),
    # '김원석': (0, 610, 1080),
    '이장욱': (0, 600, 1040)
}

def classnum_to_type(class_num) :

    for key, value in to_classnum.items() :
        if value[myconfig.class_type] == class_num :
            return  key, value[1], value[2], value[3], value[4]
    return "", "", "", ""

def imagecreate(data, path, start, end):

    starttime = datetime.now()
    print("start time : ")
    print(starttime)

    high = center = low = -1
    top = bottom = 0.0
    for i in range(start, end):
        img = data.iloc[i]['image'].strip()
        # lane = data.iloc[i]['lane']
        img_split = img.split('/')
        # index = img_split[-1].find('.'
        # newName = img_split[-1][:index] + '.jpg'
        # find_index = img.find('Source') + len('Source')
        # newOrigin = img.replace("Source", "Origin")
        newCrop = img.replace("Origin", "crop-300")

        raw_image = plt.imread(img)

        for name in driver:
            top = bottom = 0.0
            if img.find(name) != -1:
                high, center, low = driver.get(name)
                h_offset = center - high
                l_offset = low - center

                if h_offset > l_offset:
                    # top = bottom = l_offset / raw_image.shape[0]
                    top = int((raw_image.shape[0] / 2) - l_offset)
                    bottom = int((raw_image.shape[0] / 2) + l_offset)
                    # bottom = l_offset
                else:
                    # top = bottom = h_offset / raw_image.shape[0]
                    # top = int((raw_image.shape[0] / 2) - h_offset)
                    # bottom = int((raw_image.shape[0] / 2) + h_offset)
                    top = int(center - h_offset)
                    bottom = int(center + h_offset)
                break

        if top == 0.0 and bottom == 0.0:
            # print(img)
            continue

        # 20190110-sony1
        # new_image = crop(raw_image, 0.0, 0.1852)
        # 20190110-sony2
        # new_image = crop(raw_image, top, bottom)
        new_image = raw_image[top:bottom, :]

        # if os.path.exists(os.path.dirname(os.path.abspath(newOrigin))) == False:
        #     os.makedirs(os.path.dirname(os.path.abspath(newOrigin)))
        # mpimg.imsave(newOrigin, new_image)

        ratio = 300 / new_image.shape[1]
        crop_height = int(new_image.shape[0] * ratio)

        crop_image = resize(new_image, (crop_height, 300))
        if os.path.exists(os.path.dirname(os.path.abspath(newCrop))) == False:
            os.makedirs(os.path.dirname(os.path.abspath(newCrop)))
        mpimg.imsave(newCrop, crop_image)

    endtime = datetime.now()
    print("end time : ", endtime-starttime)
    print('End Thread - ' + path)


if __name__ == '__main__':

    # data = pd.read_csv('validation_data_labels.txt')
    # data = pd.read_csv('/home/hclee/workspace/git_adev2/lane_localization/video/image/03/labels.txt')
    data = pd.read_csv('/home/data/highway_lane/newData/Origin/2019_fh_highway_inout/labels.txt')
    num_of_img = len(data)
    # step = num_of_img / 8

    # path = "/home/data/highway_lane/crop/testSet/"
    # path = "/home/hclee/workspace/git_adev2/lane_localization/video/image/crop/03-300x100"
    path = "/home/data/highway_lane/newData/"
    # path = "./tool/"

    # t1 = threading.Thread(target=imagecreate, args=(data, path, 1, num_of_img))
    # t1 = threading.Thread(target=imagecreate, args=(data, path, 1, 2))
    # t1.start()

    # t1 = threading.Thread(target=imagecreate, args=(data, path,      1,          step))
    # t2 = threading.Thread(target=imagecreate, args=(data, path,  step+1,       step*2))
    # t3 = threading.Thread(target=imagecreate, args=(data, path,  (step*2)+1,   step*3))
    # t4 = threading.Thread(target=imagecreate, args=(data, path,  (step*3)+1,   step*4))
    # t5 = threading.Thread(target=imagecreate, args=(data, path,  (step*4)+1,   step*5))
    # t6 = threading.Thread(target=imagecreate, args=(data, path,  (step*5)+1,   step*6))
    # t7 = threading.Thread(target=imagecreate, args=(data, path,  (step*6)+1,   step*7))
    # t8 = threading.Thread(target=imagecreate, args=(data, path,  (step*7)+1,   num_of_img))

    t1 = threading.Thread(target=imagecreate, args=(data, path,      1,       20000))
    # t2 = threading.Thread(target=imagecreate, args=(data, path,  20001,       40000))
    # t3 = threading.Thread(target=imagecreate, args=(data, path,  40001,       60000))
    # t4 = threading.Thread(target=imagecreate, args=(data, path,  60001,       80000))
    # t5 = threading.Thread(target=imagecreate, args=(data, path,  80001,      100000))
    # t6 = threading.Thread(target=imagecreate, args=(data, path, 100001,      120000))
    # t7 = threading.Thread(target=imagecreate, args=(data, path, 120001,      140000))
    # t8 = threading.Thread(target=imagecreate, args=(data, path, 140001,  num_of_img))

    t1.start()
    # t2.start()
    # t3.start()
    # t4.start()
    # t5.start()
    # t6.start()
    # t7.start()
    # t8.start()

    # starttime = datetime.now()
    # print("start time : ")
    # print(starttime)
    #
    # step = 0
    # for i in range(num_of_img):
    #     img = data.iloc[i]['image'].strip()
    #     lable = data.iloc[i]['lane']
    #
    #     index=img.find('highway_lane')
    #     newpath = img[index:]
    #
    #     raw_image = plt.imread(img)
    #     new_image, laneIndex = imagedefine.generate_new_image(raw_image, lable)
    #     newPath = img_output + newpath
    #     if os.path.exists(os.path.dirname(os.path.abspath(newPath))) == False:
    #         os.makedirs(os.path.dirname(os.path.abspath(newPath)))
    #     mpimg.imsave(newPath, new_image)
    #
    #     if (i % 1000) == 0:
    #         step = step + 1
    #         via_time = datetime.now()
    #         print("step %d time : %s" % (step, via_time-starttime))
    #
    # endtime = datetime.now()
    # print("end time : ", endtime-starttime)