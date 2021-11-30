from datetime import datetime

import pandas as pd
from flask import Flask

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import myconfig
import imagedefine

import os
import threading

app = Flask(__name__)
model = None
prev_image_array = None

def imagecreate(data, path, start, end):

    for i in range(start, end):
        img = data.iloc[i]['image'].strip()
        lable = data.iloc[i]['lane']

        img_split = img.split('/')
        index = img_split[-1].find('.')
        newName = img_split[-1][:index] + '.jpg'

        # index=img.find('highway_lane')
        # index=img.find('image')
        # newpath = img[index:]
        # newName = "%d.jpg" % i

        raw_image = plt.imread(img)
        new_image, laneIndex = imagedefine.generate_new_image(raw_image, lable, 0.00, 0.1, (100,300))
        newPath = path + newName
        if os.path.exists(os.path.dirname(os.path.abspath(newPath))) == False:
            os.makedirs(os.path.dirname(os.path.abspath(newPath)))
        mpimg.imsave(newPath, new_image)

    print('End Thread - ' + path)


if __name__ == '__main__':

    # data = pd.read_csv('validation_data_labels.txt')
    # data = pd.read_csv('/home/hclee/workspace/git_adev2/lane_localization/video/image/03/labels.txt')
    data = pd.read_csv('./tool/test.txt')
    num_of_img = len(data)

    # path = "/home/data/highway_lane/crop/testSet/"
    # path = "/home/hclee/workspace/git_adev2/lane_localization/video/image/crop/03-300x100"
    path = "./tool"

    t1 = threading.Thread(target=imagecreate, args=(data, path, 1, num_of_img))
    t1.start()

    # t1 = threading.Thread(target=imagecreate, args=(data, path, 1, 10000))
    # t2 = threading.Thread(target=imagecreate, args=(data, path, 10001, 20000))
    # t3 = threading.Thread(target=imagecreate, args=(data, path, 20001, 30000))
    # t4 = threading.Thread(target=imagecreate, args=(data, path, 30001, 40000))
    # t5 = threading.Thread(target=imagecreate, args=(data, path, 10001, 20000))
    # t6 = threading.Thread(target=imagecreate, args=(data, path, 10001, 20000))
    # t1.start()
    # t2.start()
    # t3.start()
    # t4.start()
    # t5.start()
    # t6.start()

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
