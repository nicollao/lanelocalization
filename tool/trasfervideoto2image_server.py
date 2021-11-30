import cv2
import IPython
import matplotlib.image as mpimg
import os
import threading
import numpy as np
import scipy.misc
import pysrt
from datetime import datetime

video_name1 = 'MAH05391_1204_1024.MP4'
video_name2 = 'MAH05392_1204_1024.MP4'
video_name3 = 'MAH08405_074_0463.MP4'
video_name4 = 'MAH08406_074_0463.MP4'
# video_name4 = 'GH040059_3194_1805.MP4'
# video_name5 = 'GH050059_3194_1805.MP4'
# video_name6 = 'GH060059_3194_1805.MP4'
# video_name7 = 'GH070059_3194_1805.MP4'
# video_name8 = 'GH010061_3194_1805.MP4'

# video_path = '/home/hclee/workspace/git_adev2/lane_localization/video/20190110/goPro/'
video_path = '/home/hclee/workspace/git_adev2/lane_localization/video/20190108/sony2/'

output_path = '/home/data/highway_lane/newData/Source/'
output_orgin_path = '/home/data/highway_lane/newData/Origin/Origin_20190108-02/'
output_crop_path = '/home/data/highway_lane/newData/crop-300/crop-300_20190108-02/'

def crop(image, top_percent, bottom_percent):
    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
    return image[top:bottom, :]

def resize(image, new_dim):
    return scipy.misc.imresize(image, new_dim)

def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# def cv2_imshow(img):
#     ret = cv2.imencode('.png', img)[1].tobytes()
#     img_ip = IPython.display.Image(data=ret)
#     IPython.display.display(img_ip)

# def create_image(video_name, dirname):
#     cap = cv2.VideoCapture(video_name)
#     # vw = None
#     frame = -1  # counter for debugging (mostly), 0-indexed
#
#
#     while True:  # should 481 frames
#         frame += 1
#         ret, img = cap.read()
#         if not ret: break
#
#         dirpath1 = './video/image/' + dirname
#         text = dirpath1 + '/img{}.jpg'.format(frame)
#
#         dirpath2 = './video/image/crop/' + dirname
#         text_output = dirpath2 + '/img{}.jpg'.format(frame)
#         cv2.imwrite(text, img)
#
#         raw_img = plt.imread(text)
#         img, lane = imagedefine.generate_new_image(raw_img, 0)
#         mpimg.imsave(text_output, img)
#
#     # img1 = mpimg.imread('./video/img200.jpg')
#     # img2 = mpimg.imread('./video/img-1-200.jpg')
#
#     # img3 = cv2.imread('./video/img200.jpg', cv2.IMREAD_COLOR)
#     # img4 = cv2.imread('./video/img-1-200.jpg', cv2.IMREAD_COLOR)
#
#     # Relative Path
#     # img5 = Image.open("img200.jpg")
#     # img6 = Image.open("img-1-200.jpg")
#
#     # dirpath1 = './video/image/' + dirname
#     # text = dirpath1 + '/img{}.jpg'.format(frame)
#     # mpimg.imsave(text, img)
#
#     # dirpath2 = './video/image/crop/' + dirname
#     # text_output = dirpath2 + '/img{}.jpg'.format(frame)
#     # cv2.imwrite(text_output, img)
#
#     # raw_img = plt.imread(text)
#     # img, lane = imagedefine.generate_new_image(raw_img, 0)
#     # mpimg.imsave(text_output, img)
#
#     print(video_name + ' is doen!!!!')
#     cap.release()
#     # if vw is not None:
#
def create_image_caption_with_resize(video_name, type, output_path, frame_delay=10):

    starttime = datetime.now()
    print("start time : ")
    print(starttime)

    cap = cv2.VideoCapture(video_path + video_name)

    crop_size = video_name[:-4].split("_")
    crop_top = float("0." + crop_size[1])
    crop_bottom = float("0." + crop_size[2])

    # vw = None
    frame = -1  # counter for debugging (mostly), 0-indexed
    subscription = video_name.split(".")
    subs = pysrt.open(video_path + subscription[0] + ".srt")

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.

    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print
        "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print
        "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)

    subindex = 0
    first_sub = subs[subindex]
    startframe = (first_sub.start.hours * 3600 + first_sub.start.minutes * 60 + first_sub.start.seconds) * fps
    endframe = (first_sub.start.hours * 3600 + first_sub.end.minutes * 60 + first_sub.end.seconds) * fps
    textframe = first_sub.text

    while True:  # should 481 frames
        frame += 1
        ret, img = cap.read()
        if not ret:
            break

        if frame % frame_delay not 0:
            continue

        if frame > endframe:
            subindex = subindex + 1
            if subindex >= len(subs):
                break

            next_sub = subs[subindex]
            startframe = (next_sub.start.hours * 3600 + next_sub.start.minutes * 60 + next_sub.start.seconds) * fps
            endframe = (next_sub.start.hours * 3600 + next_sub.end.minutes * 60 + next_sub.end.seconds) * fps
            textframe = next_sub.text

        if subindex >= len(subs):
            break

        if frame < startframe:
            continue

        # dirpath1 = output_orgin_path + textframe + '/' + video_name[:-4] + '/' + textframe
        dirpath1 = 'Origin_' + output_path + textframe + '/' + video_name[:-4] + '/' + textframe
        text_orgin = dirpath1 + '/img{}.jpg'.format(frame)
        if os.path.exists(dirpath1) == False:
            os.makedirs(dirpath1)

        # dirpath2 = output_crop_path + textframe + '/' + video_name[:-4] + '/' + textframe
        dirpath2 = 'crop-300_' + output_path + textframe + '/' + video_name[:-4] + '/' + textframe
        text_resize = dirpath2 + '/img{}.jpg'.format(frame)
        if os.path.exists(dirpath2) == False:
            os.makedirs(dirpath2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if type == 'goPro':
            img_crop = crop(img, crop_bottom, crop_top)
            img_crop = rotateImage(img_crop, 180)
        else:
            img_crop = crop(img, crop_top, crop_bottom)

        ratio = 300 / img_crop.shape[1]
        crop_height = int(img_crop.shape[0] * ratio)
        img_resize = resize(img_crop, (crop_height, 300))

        mpimg.imsave(text_orgin, img_crop)
        mpimg.imsave(text_resize, img_resize)

    starttime = datetime.now()
    print("start time : ")
    print(starttime)
    print(video_name + ' is doen!!!!')
    cap.release()
#
#
# def create_image_caption_with_resize(video_name, dirname):
#     cap = cv2.VideoCapture(video_path + video_name)
#
#     # crop_size = video_name[:-4].split("_")
#     # crop_top = float("0." + crop_size[1])
#     # crop_bottom = float("0." + crop_size[2])
#
#     # vw = None
#     frame = -1  # counter for debugging (mostly), 0-indexed
#     subscription = video_name.split(".")
#     subs = pysrt.open(video_path + subscription[0] + ".srt")
#
#     # Find OpenCV version
#     (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
#
#     # With webcam get(CV_CAP_PROP_FPS) does not work.
#     # Let's see for ourselves.
#
#     if int(major_ver) < 3:
#         fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
#         print
#         "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
#     else:
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         print
#         "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
#
#     subindex = 0
#     first_sub = subs[subindex]
#     startframe = (first_sub.start.hours * 3600 + first_sub.start.minutes * 60 + first_sub.start.seconds) * fps
#     endframe = (first_sub.start.hours * 3600 + first_sub.end.minutes * 60 + first_sub.end.seconds) * fps
#     textframe = first_sub.text
#
#     while True:  # should 481 frames
#         frame += 1
#         ret, img = cap.read()
#         if not ret:
#             break
#
#         if frame > endframe:
#             subindex = subindex + 1
#             if subindex >= len(subs):
#                 break
#
#             next_sub = subs[subindex]
#             startframe = (next_sub.start.hours * 3600 + next_sub.start.minutes * 60 + next_sub.start.seconds) * fps
#             endframe = (next_sub.start.hours * 3600 + next_sub.end.minutes * 60 + next_sub.end.seconds) * fps
#             textframe = next_sub.text
#
#         if subindex >= len(subs):
#             break
#
#         if frame < startframe:
#             continue
#
#         # dirpath1 = output_orgin_path + textframe + '/' + video_name[:-4] + '/' + textframe
#         # text_orgin = dirpath1 + '/img{}.jpg'.format(frame)
#         # if os.path.exists(dirpath1) == False:
#         #     os.makedirs(dirpath1)
#         #
#         # dirpath2 = output_crop_path + textframe + '/' + video_name[:-4] + '/' + textframe
#         # text_resize = dirpath2 + '/img{}.jpg'.format(frame)
#         # if os.path.exists(dirpath2) == False:
#         #     os.makedirs(dirpath2)
#
#         # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         #
#         # img_crop = crop(img, crop_bottom, crop_top)
#         # img_crop = rotateImage(img_crop, 180)
#         #
#         # ratio = 300 / img_crop.shape[1]
#         # crop_height = int(img_crop.shape[0] * ratio)
#         # img_resize = resize(img_crop, (crop_height, 300))
#         #
#         # mpimg.imsave(text_orgin, img_crop)
#         # mpimg.imsave(text_resize, img_resize)
#
#     print(video_name + ' is doen!!!!')
#     cap.releases()

def create_image_caption(video_name, dirname):
    cap = cv2.VideoCapture(video_path + video_name)

    # vw = None
    frame = -1  # counter for debugging (mostly), 0-indexed
    subscription = video_name.split(".")
    subs = pysrt.open(video_path + subscription[0] + ".srt")

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.

    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print
        "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print
        "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)

    subindex = 0
    first_sub = subs[subindex]
    startframe = (first_sub.start.hours * 3600 + first_sub.start.minutes * 60 + first_sub.start.seconds) * fps
    endframe = (first_sub.start.hours * 3600 + first_sub.end.minutes * 60 + first_sub.end.seconds) * fps
    textframe = first_sub.text

    while True:  # should 481 frames
        frame += 1
        ret, img = cap.read()
        if not ret:
            break

        if frame > endframe:
            subindex = subindex + 1
            if subindex >= len(subs):
                break

            next_sub = subs[subindex]
            startframe = (next_sub.start.hours * 3600 + next_sub.start.minutes * 60 + next_sub.start.seconds) * fps
            endframe = (next_sub.start.hours * 3600 + next_sub.end.minutes * 60 + next_sub.end.seconds) * fps
            textframe = next_sub.text

        if subindex >= len(subs):
            break

        if frame < startframe:
            continue

        # dirpath1 = './video/image/' + textframe
        # dirpath1 = video_path + textframe
        dirpath1 = output_path + textframe + '/' + video_name[:-4] + '/' + textframe
        text = dirpath1 + '/img{}.jpg'.format(frame)

        if os.path.exists(dirpath1) == False:
            os.makedirs(dirpath1)

        # dirpath2 = './video/image/crop/' + textframe
        # dirpath2 = video_path + 'crop/' + textframe
        # text_output = dirpath2 + '/img{}.jpg'.format(frame)

        # if os.path.exists(dirpath2) == False:
        #     os.makedirs(dirpath2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mpimg.imsave(text, img)

        # cv2.imwrite(text, img)
        #
        # raw_img = plt.imread(text)
        # img, lane = imagedefine.generate_new_image(raw_img, 0, 0.35, 0.1, resize_dim = (100, 300))
        # mpimg.imsave(text_output, img)

    print(video_name + ' is doen!!!!')
    cap.releases()



if __name__ == '__main__':

# output_orgin_path = '/home/data/highway_lane/newData/Origin/Origin_20190108-02/'
# output_crop_path = '/home/data/highway_lane/newData/crop-300/crop-300_20190108-02/'

    t1 = threading.Thread(target=create_image_caption_with_resize, args=(video_name1, 'sony1', '20190108-01'))
    t2 = threading.Thread(target=create_image_caption_with_resize, args=(video_name2, 'sony1', '20190108-01'))
    t3 = threading.Thread(target=create_image_caption_with_resize, args=(video_name3, 'sony2', '20190108-02'))
    t4 = threading.Thread(target=create_image_caption_with_resize, args=(video_name4, 'sony2', '20190108-02'))
    # t3 = threading.Thread(target=create_image_caption_with_resize, args=(video_name3, 'goPro'))
    # t4 = threading.Thread(target=create_image_caption_with_resize, args=(video_name4, 'goPro'))
    # t5 = threading.Thread(target=create_image_caption_with_resize, args=(video_name5, 'goPro'))
    # t6 = threading.Thread(target=create_image_caption_with_resize, args=(video_name6, 'goPro'))
    # t7 = threading.Thread(target=create_image_caption_with_resize, args=(video_name1, 'sony1'))
    # t8 = threading.Thread(target=create_image_caption_with_resize, args=(video_name8, 'goPro'))

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    # t5.start()
    # t6.start()
    # t7.start()
    # t8.start()

