import tensorflow as tf
from tensorflow import keras

import numpy as np
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from numpy import asarray

import scipy.misc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.utils import shuffle
import myconfig
import classdefine
from random import *

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, batch_size=64, dim=(200,66,3), n_channels=3, n_classes=5, shuffle=False, top_percent = 0.35, bottom_percent = 0.0):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

        self.step = 0
        self.image_files_and_index = []

        self.top_percent = top_percent
        self.bottom_percent = bottom_percent

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # print("__getitem :: ", index)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # X, y = self.__data_generation(list_IDs_temp)
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes, mode = 0):

        X_batch = []
        y_batch = []
        # print("generate_next_batch 11112")

        # images = self.get_next_image_files(indexes, self.batch_size)

        for index in indexes:
            # print("train index : ", index)
            # img = self.list_IDs[index][0]
            # lane = self.list_IDs[index][1]
            # img = temp[0]
            # lane = temp[1]
            img = self.list_IDs.iloc[index]['image'].strip()
            lane = self.list_IDs.iloc[index]['lane']

            raw_image = Image.open(img)
            raw_laneidx = classdefine.type_to_classnum(lane)[myconfig.class_type]
            # print('raw_laneidx : {}'.format(raw_laneidx))

            new_image, new_laneidx = self.generate_new_image(raw_image, raw_laneidx)

            X_batch.append(new_image)
            y_batch.append(new_laneidx)

        return np.array(X_batch), np_utils.to_categorical(np.array(y_batch), self.n_classes)


    def generate_new_image(self, image, laneidx):

        left = 0
        top = int(image.size[1] * self.top_percent)
        right = image.size[0]
        bottom = int(image.size[1] * self.bottom_percent)

        img_crop = image.crop((left, top, right, bottom))
        result = asarray(img_crop.resize(self.dim, Image.ANTIALIAS))

        return result, laneidx

    def generate_next_batch(batch_size=64, mode=0, val_data=0):
        # index=0
        while 1:
            X_batch = []
            y_batch = []
            images = get_next_image_files(batch_size, val_data)
            for img_file, laneidx in images:
                raw_image = plt.imread(img_file)
                raw_laneidx = laneidx

                new_image = generate_new_image(raw_image)

                X_batch.append(new_image)
                y_batch.append(raw_laneidx)

            yield np.array(X_batch), np_utils.to_categorical(np.array(y_batch), myconfig.num_classes)

    def get_next_image_files(self, indexes, batch_size=64, val_data=0):

        image_files_and_index = []

        for index in indexes:
            print("train index : ", index)
            img = self.list_IDs.iloc[index]['image'].strip()
            lane = self.list_IDs.iloc[index]['lane']
            image_files_and_index.append(img, lane)

        return image_files_and_index

    def init_imagedefine():

        global train_file_data, train_file_indicator, train_file_total_count
        global val_file_data, val_file_indicator, val_file_total_count

        data = pd.read_csv(myconfig.train_datapath)
        train_file_total_count = len(data)
        train_file_data = shuffle(data)

        data1 = pd.read_csv(myconfig.val_datapath)
        val_file_total_count = len(data1)
        val_file_data = shuffle(data1)

    def random_shear(image, steering_angle, shear_range=200):
        rows, cols, ch = image.shape
        dx = np.random.randint(-shear_range, shear_range + 1)
        random_point = [cols / 2 + dx, rows / 2]
        pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
        pts2 = np.float32([[0, rows], [cols, rows], random_point])
        dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
        steering_angle += dsteering
        return image, steering_angle

    def crop(self, image, top_percent=0.35, bottom_percent=0.1):
        top = int(np.ceil(image.shape[0] * top_percent))
        bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
        return image[top:bottom, :]

    def random_gamma(self, image):
        gamma = np.random.uniform(0.4, 1.5)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # LUT : look up table
        return cv2.LUT(image, table)

    def resize(self, image, new_dim):
        return scipy.misc.imresize(image, new_dim)

    def gaussian_blur(img, kernel=5):
        # Apply Gaussian Blur
        blur = cv2.GaussianBlur(img, (kernel, kernel), 0)
        return blur

    def abs_sobel(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        if orient == 'x':
            img_s = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            img_s = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        img_abs = np.absolute(img_s)
        img_sobel = np.uint8(255 * img_abs / np.max(img_abs))

        binary_output = 0 * img_sobel
        binary_output[(img_sobel >= thresh[0]) & (img_sobel <= thresh[1])] = 1
        return binary_output

    def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        # Apply threshold
        if orient == 'x':
            img_s = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        else:
            img_s = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        img_abs = np.absolute(img_s)
        img_sobel = np.uint8(255 * img_abs / np.max(img_abs))

        binary_output = 0 * img_sobel
        binary_output[(img_sobel >= thresh[0]) & (img_sobel <= thresh[1])] = 1
        return binary_output

    def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
        # Calculate gradient magnitude
        # Apply threshold
        img_sx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        img_sy = cv2.Sobel(img, cv2.CV_64F, 0, 1)

        img_s = np.sqrt(img_sx ** 2 + img_sy ** 2)
        img_s = np.uint8(img_s * 255 / np.max(img_s))
        binary_output = 0 * img_s
        binary_output[(img_s >= thresh[0]) & (img_s <= thresh[1])] = 1
        return binary_output

    def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Calculate gradient direction
        # Apply threshold
        img_sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        img_sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        grad_s = np.arctan2(np.absolute(img_sy), np.absolute(img_sx))

        binary_output = 0 * grad_s  # Remove this line
        binary_output[(grad_s >= thresh[0]) & (grad_s <= thresh[1])] = 1
        return binary_output

    def sobel_combined(image):
        # Apply combined sobel filter
        img_g_mag = mag_thresh(image, 3, (20, 150))
        img_d_mag = dir_threshold(image, 3, (.6, 1.1))
        img_abs_x = abs_sobel_thresh(image, 'x', 5, (50, 200))
        img_abs_y = abs_sobel_thresh(image, 'y', 5, (50, 200))
        sobel_combined = np.zeros_like(img_d_mag)
        sobel_combined[((img_abs_x == 1) & (img_abs_y == 1)) | ((img_g_mag == 1) & (img_d_mag == 1))] = 1
        return sobel_combined

    def apply_color_mask(hsv, img, low, high):
        # Apply color mask to image
        mask = cv2.inRange(hsv, low, high)
        res = cv2.bitwise_and(img, img, mask=mask)
        return (mask, res)

    def imagecombine(image, index=0, kernel_size=9):

        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Define threshholds for white line
        white_hsv_low = np.array([0, 0, 250])
        white_hsv_high = np.array([255, 80, 255])

        # Define threshholds for yellow line
        yellow_hsv_low = np.array([0, 100, 100])
        yellow_hsv_high = np.array([80, 255, 255])

        img_mask_white, img_white = apply_color_mask(img_hsv, image, white_hsv_low, white_hsv_high)
        img_mask_yellow, img_yellow = apply_color_mask(img_hsv, image, yellow_hsv_low, yellow_hsv_high)
        img_mask = cv2.bitwise_or(img_mask_white, img_mask_yellow)
        # print ('img_white : ',  np.shape(img_white))
        # print ('img_mask_white : ', np.shape(img_mask_white))

        img_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

        img_gs = img_hls[:, :, 1]
        img_abs_x = abs_sobel(img_gs, 'x', 5, (50, 225))
        img_abs_y = abs_sobel(img_gs, 'y', 5, (50, 225))

        img_sobel = np.copy(cv2.bitwise_or(img_abs_x, img_abs_y))

        image_combined = np.zeros_like(img_sobel)
        image_combined[(img_mask >= .5) | (img_sobel >= .5)] = 1
        # image_combined[(img_sobel >= .5)] = 1
        image_blur = gaussian_blur(image_combined, 5)

        # image_output = cv2.merge((image_combined, image_combined, image_combined))
        # image_output = cv2.cvtColor(image_combined, cv2.COLOR_GRAY2RGB)
        image_output = np.zeros_like(image)
        # image_output.fill(50)
        # image_output = cv2.cvtColor(image_output, COLOR_WHITE)
        image_output[:, :, 0] = image_blur * 250
        image_output[:, :, 1] = image_blur * 250
        image_output[:, :, 2] = image_blur * 250

        # savename1 = "imgGamma_%d_3-1_mask" % index
        # savename2 = "imgGamma_%d_3-2_sobel" % index
        # savename3 = "imgGamma_%d_3-3_white" % index
        # savename4 = "imgGamma_%d_3-4_white_mask" % index
        # savename5 = "imgGamma_%d_3-5_yellow" % index
        # savename6 = "imgGamma_%d_3-6_yellow_mask" % index
        # savename7 = "imgGamma_%d_3-7_hsv" % index
        # savename8 = "imgGamma_%d_3-8_hls" % index
        # savename9 = "imgGamma_%d_3-9_combined" % index
        # savename10 = "imgGamma_%d_3-10_blur" % index
        # savename11 = "imgGamma_%d_3-11_output" % index

        # mpimg.imsave(savename1, img_mask, cmap='gray')
        # mpimg.imsave(savename2, img_sobel)
        # mpimg.imsave(savename3, img_white)
        # mpimg.imsave(savename4, img_mask_white, cmap='gray')
        # mpimg.imsave(savename5, img_yellow)
        # mpimg.imsave(savename6, img_mask_yellow, cmap='gray')
        # mpimg.imsave(savename7, img_hsv)
        # mpimg.imsave(savename8, img_hls)
        # mpimg.imsave(savename9, image_combined)
        # mpimg.imsave(savename10, image_blur)
        # mpimg.imsave(savename11, image_output)

        return image_output

    warp_src = []
    warp_dst = []

    def warp_initial(img, top_crop_percent=0.35, bottom_crop_percent=0.1):
        # Apply perspective transform
        img_size = np.shape(img)
        # ht_window = np.uint(img_size[0] * top_crop_percent)
        ht_window = np.uint(img_size[0] * 0.5)
        hb_window = np.uint(img_size[0] * (1.0 - bottom_crop_percent))
        c_window = np.uint(img_size[1] / 2)

        ctl_window = c_window - .7 * np.uint(img_size[1] / 2)
        ctr_window = c_window + .7 * np.uint(img_size[1] / 2)
        cbl_window = c_window - .9 * np.uint(img_size[1] / 2)
        cbr_window = c_window + .9 * np.uint(img_size[1] / 2)

        global warp_src, warp_dst
        warp_src = np.float32(
            [[cbl_window, hb_window], [cbr_window, hb_window], [ctr_window, ht_window], [ctl_window, ht_window]])
        warp_dst = np.float32([[0, img_size[0]], [img_size[1], img_size[0]], [img_size[1], 0], [0, 0]])

    def warp_image(img, src, dst):
        # # Apply perspective transform
        img_size = np.shape(img)
        # ctl_window = c_window - .7*np.uint(img_size[1]/2)
        # ctr_window = c_window + .7*np.uint(img_size[1]/2)
        # cbl_window = c_window - .9*np.uint(img_size[1]/2)
        # cbr_window = c_window + .9*np.uint(img_size[1]/2)
        #
        # src = np.float32([ [cbl_window, hb_window], [cbr_window, hb_window], [ctr_window, ht_window], [ctl_window, ht_window] ])
        # dst = np.float32([ [0, img_size[0]], [img_size[1], img_size[0]], [img_size[1], 0], [0, 0] ])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)
        # Minv = cv2.getPerspectiveTransform(dst, src)
        return warped

    def contrast_image(image):
        # add contrast effect
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])

        cdf = hist.cumsum()

        # cdf의 값이 0인 경우는 mask처리를 하여 계산에서 제외
        # mask처리가 되면 Numpy 계산에서 제외가 됨
        # 아래는 cdf array에서 값이 0인 부분을 mask처리함
        cdf_m = np.ma.masked_equal(cdf, 0)

        # History Equalization 공식
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

        # Mask처리를 했던 부분을 다시 0으로 변환
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        image = cdf[image]

        return image
