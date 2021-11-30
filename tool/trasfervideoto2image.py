import cv2 import IPython
import matplotlib.image as mpimg
import os
import threading
import numpy as np
import scipy.misc
import pysrt
from datetime import datetime
import enum


# video_name1 = 'MAH05391_1204_1024.MP4'

# video_path = '/home/hclee/workspace/git_adev2/lane_localization/video/20190110/goPro/'
# video_path = '/home/hclee/workspace/git_adev2/lane_localization/video/20190110/sony2/'
# video_path = '/home/hclee/workspace/git_adev2/lane_localization/video/validation/'

output_path = '/home/data/highway_lane/newData/Source/'

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
def create_image_caption_with_resize(video_path, video_name, type, originpath, croppath, frame_delay=5):

    print(video_name + ' strat!!!!')
    # starttime = datetime.now()
    # print("start time : ")
    # print(starttime)

    cap = cv2.VideoCapture(video_path + video_name)

    crop_size = video_name[:-4].split("_")
    if len(crop_size) > 1:
        crop_top = float("0." + crop_size[1])
        crop_bottom = float("0." + crop_size[2])
    else:
        crop_top = 0.0
        crop_bottom = 0.0

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

    if len(first_sub.text.split(",")) > 1:
        temp = first_sub.text.split(",")
        textframe = temp[0]
        # frame_delay = int(temp[1])
    else:
        textframe = first_sub.text

    while True:  # should 481 frames
        frame += 1
        ret, org_img = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

        if frame_delay != 0 and frame % frame_delay != 0:
            continue

        if frame > endframe:
            subindex = subindex + 1
            if subindex >= len(subs):
                break

            next_sub = subs[subindex]
            startframe = (next_sub.start.hours * 3600 + next_sub.start.minutes * 60 + next_sub.start.seconds) * fps
            endframe = (next_sub.start.hours * 3600 + next_sub.end.minutes * 60 + next_sub.end.seconds) * fps

            if len(next_sub.text.split(",")) > 1:
                temp = next_sub.text.split(",")
                textframe = temp[0]
                # frame_delay = int(temp[1])
            else:
                textframe = next_sub.text
                # frame_delay = 4

        if subindex >= len(subs):
            break

        if frame < startframe:
            continue

        if originpath != '':
            dirpath1 = originpath + textframe + '/' + video_name[:-4] + '/' + textframe
            text_orgin = dirpath1 + '/img{}.jpg'.format(frame)
            if os.path.exists(dirpath1) == False:
                os.makedirs(dirpath1)

        if croppath != '':
            # dirpath2 = croppath + textframe + '/' + video_name[:-4] + '/' + textframe
            dirpath2 = croppath + textframe + '/' + video_name[:-4]
            text_resize = dirpath2 + '/{}_img{}.jpg'.format(video_name[:-4], frame)
            if os.path.exists(dirpath2) == False:
                os.makedirs(dirpath2)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if type == 'goPro':
            img_crop = crop(img, crop_bottom, crop_top)
            img_crop = rotateImage(img_crop, 180)
            img_orign = rotateImage(img, 180)
        else:
            img_crop = crop(img, crop_top, crop_bottom)

        ratio = 300 / img_crop.shape[1]
        crop_height = int(img_crop.shape[0] * ratio)
        img_resize = resize(img_crop, (crop_height, 300))

        if originpath != '':
            mpimg.imsave(text_orgin, img_orign)

        if croppath != '':
            mpimg.imsave(text_resize, img_resize)

    # starttime = datetime.now()
    # print("end time : ")
    # print(starttime)
    print(video_name + ' is doen!!!!')
    cap.release()

def create_image_caption(video_path, video_name, type, output_path):

    print(video_name + ' is strat!!!!')

    cap = cv2.VideoCapture(video_path + video_name)

    # vw = None
    frame = -1  # counter for debugging (mostly), 0-indexed
    # subscription = video_name.split(".")
    # subs = pysrt.open(video_path + subscription[0] + ".srt")

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

    # subindex = 0
    # first_sub = subs[subindex]
    # startframe = (first_sub.start.hours * 3600 + first_sub.start.minutes * 60 + first_sub.start.seconds) * fps
    # endframe = (first_sub.start.hours * 3600 + first_sub.end.minutes * 60 + first_sub.end.seconds) * fps
    # textframe = first_sub.text

    while True:  # should 481 frames
        frame += 1
        ret, img = cap.read()
        if not ret:
            break

       #  if frame > endframe:
       #      subindex = subindex + 1
       #      if subindex >= len(subs):
       #          break

        #     next_sub = subs[subindex]
        #     startframe = (next_sub.start.hours * 3600 + next_sub.start.minutes * 60 + next_sub.start.seconds) * fps
        #     endframe = (next_sub.start.hours * 3600 + next_sub.end.minutes * 60 + next_sub.end.seconds) * fps
        #     textframe = next_sub.text

        # if subindex >= len(subs):
        #     break

        # if frame < startframe:
        #     continue

        if type == 'cam':
            # img_crop = crop(img, crop_bottom, crop_top)
            img = rotateImage(img, 180)

        # dirpath1 = './video/image/' + textframe
        # dirpath1 = './' + textframe + '/' + video_name[:-4] + '/' + textframe
        dirpath1 = output_path + '/' + video_name[:-4]
        text = dirpath1 + '/img{}.jpg'.format(frame)

        if os.path.exists(dirpath1) == False:
            os.makedirs(dirpath1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mpimg.imsave(text, img)

    print(video_name + ' is doen!!!!')
    cap.releases()


# class type_makeImg(enum.Enum):
#     MAKE_FILE = 0
#     MAKE_DIRECTORY = 1
#     MAKE_CROP_IMG = 2

if __name__ == '__main__':

    type_makeImg = 1

    video_path0 = '/home/data/Videos/Alpha-2018-2019/trainData/'
    # output_orgin_path0 = '/home/data/highway_lane/newData/Origin/demo-2019fh/'
    output_orgin_path0 = '/home/data/Alpha-2018-2019/'
    # output_crop_path0 = '/home/data/highway_lane/newData/crop-300/demo-2019fh/'
    output_crop_path0 = ''
    # video_name1 = '04. 신갈JC-인천-서울-20190522.mp4'
    # video_name1 = '04_25_2315.mp4'

    if type_makeImg == 0:
        # 20190102
        # video_path0 = '/home/hclee/workspace/git_adev2/lane_localization/video/20190509-cam/'
        # output_orgin_path0 = '/home/data/highway_lane/newData/Origin/Origin_20190509-cam/'
        # output_crop_path0 = '/home/data/highway_lane/newData/crop-300/crop-300_20190416-1/'
        # video_name1 = '4mm-01.mp4'
        # video_name2 = '12mm-01.mp4'
        # video_name3 = 'usbcam-01.mp4'

        test1 = threading.Thread(target=create_image_caption, args=(video_path0, video_name1, 'sony2',  output_orgin_path0))
        test1.start()
        # test2 = threading.Thread(target=create_image_caption, args=(video_path0, video_name2, 'sony2',  output_orgin_path0))
        # test2.start()
        # test3 = threading.Thread(target=create_image_caption, args=(video_path0, video_name3, 'sony2',  output_orgin_path0))
        # test3.start()

    elif type_makeImg == 1:
        # 20190328
        # dirname = '20190423-testSet/'
        # video_path0 = '/home/hclee/workspace/git_adev2/lane_localization/video/' + dirname
        # output_orgin_path0 = '/home/data/highway_lane/newData/Origin/Origin_20190415/'
        # output_orgin_path0 = ''
        # output_crop_path0 = '/home/data/highway_lane/newData/crop-300/crop-300_' + dirname

        filenames = os.listdir(video_path0)
        for filename in filenames:
            full_filename = os.path.join(video_path0, filename)
            ext = os.path.splitext(full_filename)[-1]

            if ext == '.mp4':
                t1 = threading.Thread(target=create_image_caption_with_resize, args=(video_path0, filename, 'sony1', output_orgin_path0, output_crop_path0))
                t1.start()

        # video_path0 = '/home/hclee/video/'
        # output_orgin_path0 = '/home/data/highway_lane/newData/Origin/demo-2019fh/'
        # output_crop_path0 = '/home/data/highway_lane/newData/crop-300/demo-2019fh/'
        # video_name1 = '04. 신갈JC-인천-서울-20190522.mp4'
        # video_name2 = '12mm-02.mp4'
        # video_name3 = 'usbcam-01.mp4'

        # test1 = threading.Thread(target=create_image_caption_with_resize, args=(video_path0, video_name1, 'sony1',  output_orgin_path0, output_crop_path0))
        # test1.start()
        # test2 = threading.Thread(target=create_image_caption_with_resize, args=(video_path0, video_name2, 'test',  output_orgin_path0, output_crop_path0))
        # test2.start()
        # test3 = threading.Thread(target=create_image_caption_with_resize, args=(video_path0, video_name3, 'goPro',  output_orgin_path0, output_crop_path0))
        # test3.start()
