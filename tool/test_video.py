import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import cv2
import IPython
from time import sleep
import os
import pysrt
import matplotlib.image as mpimg

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#video_file_name = '/home/feelsky/data/video/02-05/2019-03-15-lane2-total5.mp4' #, seong.mp4, lotte.mp4
# video_file_name = '/home/feelsky/data/video/old/hanam3.mp4'
# load_modelname='./checkpoint/weights-best.hdf5'

video_file_dir = '/home/data/Videos/Alpha-2018-2019/2019-fh-demo/demo-analysis/'
# video_file_dir = '/home/data/Videos/2019-fh-demo/01/'
# video_file_name = '/home/hclee/Videos/01-20190522_25_2315.mp4'
# srt_file_name = '/home/hclee/Videos/01-20190522_25_2315.srt'
#
load_modelname = '../checkpoint/model-201905240942-42-loss:0.0329-val_loss:0.0565-val_acc:0.9817.hdf5'

# 1_00_02_x4_01
# load_modelname = '../checkpoint/model-201906091502-47-loss:0.0281-val_loss:0.0766-val_acc:0.9754.hdf5'

# load_modelname='../checkpoint/model-201906211025-42-loss:0.0400-val_loss:0.0636-val_acc:0.9832.hdf5'
# load_modelname='../checkpoint/model-201907011406-88-loss:0.0148-val_loss:0.0569-val_acc:0.9876.hdf5'

flip = False
high_crop = 0.48
low_crop = 0.2
check_result = False

#########################################################################################
# to_classnum = {
# '0_00_01_01_00':(0, '1', '1' , 0),
# '0_00_01_02_00':(1, '1', '2' , 1),
# '0_00_01_x3_00':(2, '1', '3x', 2),
#
# '0_00_02_03_00':(3, '2', '3' , 3),
# '0_00_02_x4_00':(4, '2', '4x', 4),
#
# '0_00_03_x5_00':(5, '3', '5x', 5),
# '1_00_03_x5_00':(5, '3', '5x', 5),
#
# '1_00_01_02_00':(6, 'r1', '2', 6),
# '1_00_01_x3_00':(7, 'r1', '3x', 7),
#
# '1_00_02_x4_00':(8, 'r2', '4x', 8),
#
# '0_00_01_01_01':(9, '1', '1M',  1),
# '0_00_01_01_x2':(10, '1', '1M', 2),
# '0_00_01_02_x1':(11, '1', '2M', 2),
# '0_00_02_03_x1':(12, '2', '3M', 4),
#
# '1_00_01_02_01':(13, 'r1', '1M', 9),
# '1_00_01_02_x2':(14, 'r1', '2M', 9),
# '1_00_01_x3_01':(15, 'r1', '3xM', 9),
# '1_00_01_x3_x2':(16, 'r1', '3xM', 9),
# '1_00_02_x4_x1':(17, 'r2',  '4xM', 10),
#
# '1_01_01_01_00':(18, 'r1', '1B', 6),
# '1_x2_01_01_00':(19, 'r1', '1B', 7),
# '1_x1_01_02_00':(20, 'r1', '2B', 7),
# '1_x2_01_02_00':(21, 'r1', '2B', 7),
# '1_x2_02_02_00':(22, 'r2', '2B', 8),
# '1_x1_01_x3_00':(23, 'r1', '3xB', 7),
# '1_x1_02_03_00':(24, 'r2', '3B', 8),
# '1_x2_03_03_00':(25, 'r3', '3B', 5),
#
# '1_00_02_04_00':(26, '3', '4',   4),
# }

to_classnum = {
'0_00_01_01_00':(0, '1', '1' , 0,  10000),
'0_00_01_02_00':(1, '1', '2' , 1,  10000),
'0_00_01_x3_00':(2, '1', '3x', 2,  10000),

'0_00_02_03_00':(3, '2', '3' , 3,  10000),
'0_00_02_x4_00':(4, '2', '4x', 4,  10000),

'0_00_03_x5_00':(5, '3', '5x', 5,  10000),

'1_00_01_02_00':(6, 'r1', '2', 6,  10000),
'1_00_01_x3_00':(7, 'r1', '3x', 7, 10000),

'1_00_02_x5_00':(8, 'r2', '4x', 8, 10000),
'0_00_03_04_00':(9,  '3', '4',  11,10000),
'1_00_03_x5_00':(10, 'r3','5x', 12,  9000),

'0_00_01_01_01':(11, '1', '1M',  1,    0),
'0_00_01_01_x2':(12, '1', '1M',  2,    0),
'0_00_01_02_x1':(13, '1', '2M',  2, 550),
'0_00_02_03_x1':(14, '2', '3M',  4, 500),

'1_00_01_02_01':(15, 'r1', '1M', 9, 1500),
# '1_00_01_02_x2':(16, 'r2', '2M', 13,  180),
'1_00_01_02_x2':(16, 'r1', '2M', 9,  180),
'1_00_01_x3_01':(17, 'r1', '3xM',9, 900),
# '1_00_01_x3_x2':(18, 'r2', '3xM',13, 990),
'1_00_01_x3_x2':(18, 'r1', '3xM',9, 990),
'1_00_02_x4_x1':(19, 'r3', '4xM',10,2700),

'1_01_01_01_00':(20, 'r1', '1B', 6,    0),
'1_x2_01_01_00':(21, 'r1', '1B', 7, 1100),
'1_x1_01_02_00':(22, 'r1', '2B', 7,  90),
'1_x2_01_02_00':(23, 'r1', '2B', 7,   0),
'1_x2_02_02_00':(24, 'r2', '2B', 8, 450),
'1_x1_01_x3_00':(25, 'r1', '3xB', 7,  0),
'1_x1_02_03_00':(26, 'r2', '3B', 8,    0),
# '1_x2_03_03_00':(27, 'r3', '3B', 5,    0),
'1_x2_03_03_00':(27, 'r3', '3B', 12,    0),
# '1_00_02_04_00':(28, '2', '4',   4,    0),
'1_01_02_02_00':(28, 'r2', '2B', 8,    0),
'1_01_03_03_00':(28, 'r3', '2B', 12,    0),
}

the_class_type=3
class_number =13    # default

def classnum_to_type(class_num) :

    for key, value in  to_classnum.items() :
        if value[the_class_type] == class_num :
            return  key , value[1], value[3]
    return "", "", ""

def type_to_classnum(class_num) :

    for key, value in  to_classnum.items() :
        if key == class_num :
            return  value[0], value[1], value[2], value[3], value[4]
    return "", "", "", ""



def apply_brightness_contrast(input_img, brightness = 0, contrast = 50):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def cv2_imshow(img):
    ret = cv2.imencode('.png', img)[1].tobytes()
    img_ip = IPython.display.Image(data=ret)
    IPython.display.display(img_ip)

def crop(image, top_percent, bottom_percent):
    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
    return image[top:bottom, :]

def rotate_image(image, angle) :

    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    scale = 1.15
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, M, (w,h))
    return rotated_image



if __name__ == '__main__':
    model = load_model(load_modelname)

    filenames = os.listdir(video_file_dir)
    for filename in filenames:
        full_filename = os.path.join(video_file_dir, filename)
        ext = os.path.splitext(full_filename)[-1]

        # if ext == '.mp4':
        if filename == '10-20190522_25_2315.mp4':
            # cap = cv2.VideoCapture(video_file_name)
            cap = cv2.VideoCapture(full_filename)
            subs = pysrt.open(full_filename[:-4] + '.srt')
            # vw = None
            frame = -1

            ############################################
            # start frame, change this
            # Find OpenCV version
            (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

            # With webcam get(CV_CAP_PROP_FPS) does not work.
            # Let's see for ourselves.

            if int(major_ver) < 3:
                fps = round(cap.get(cv2.cv.CV_CAP_PROP_FPS))
                print
                "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
            else:
                fps = round(cap.get(cv2.CAP_PROP_FPS))
                print
                "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)

            startTime=0
            # startTime=90
            frame_no=startTime*fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no);


            display_lowquality=False
            save_file=True
            pause =False
            accuracy_is_high = True

            success_count = 0
            total_count = 0

            count_per_class=np.zeros(class_number)
            success_per_class=np.zeros(class_number)
            fail_per_class=np.zeros(class_number)


            subindex = 0
            first_sub = subs[subindex]
            startframe = (first_sub.start.hours * 3600 + first_sub.start.minutes * 60 + first_sub.start.seconds) * fps
            endframe = (first_sub.start.hours * 3600 + first_sub.end.minutes * 60 + first_sub.end.seconds) * fps

            if len(first_sub.text.split(",")) > 1:
                temp = first_sub.text.split(",")
                textframe = temp[0]
                frame_delay = int(temp[1])
            else:
                textframe = first_sub.text


            while True:

                #sleep(0.0001)  #display time
                if(pause == True) :
                    sleep(1)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('p'):
                        pause = not pause
                    continue

                frame += 1
                ret, org_img = cap.read()
                if not ret: break

                # if frame>200:
                #     break

                #rotate 180
                if flip == True :
                    org_img=cv2.flip(org_img, -1);

                img= cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
                crop_img = crop(img,high_crop,low_crop);
                # crop_img = crop(org_img,high_crop,low_crop);

                resize_img = cv2.resize(crop_img, (300,100))
                input_imag= resize_img

                classes = model.predict(input_imag[None, :, :, :], batch_size=1)

                max=np.max(classes[0])

                line_type = cv2.LINE_AA
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                x, y = 10, 60
                color = (255,255,255)

                accuracy_is_high=False
                if max > 0.9 :   #
                    color = (0, 0, 255)
                    accuracy_is_high=True

                # resize_img= cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
                disp_img = cv2.resize(resize_img, (0, 0), fx=2, fy=2)

                lanedata = np.around(classes[0], 1)
                sel_class=np.argmax(classes, axis=-1)[0]

                max=np.max(classes[0])


                if accuracy_is_high == True or display_lowquality == True :

                    d1, d2, d3 = classnum_to_type(sel_class)

                    text = "class={0}, type={1}, {2}".format(d1, d2, d3)
                    cv2.putText(disp_img, text=text, org=(x, y), fontScale=font_scale, fontFace=font_face, thickness=thickness, color=color, lineType=line_type)

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
                            frame_delay = int(temp[1])
                        else:
                            textframe = next_sub.text
                            frame_delay = 4

                    if subindex >= len(subs):
                        break

                    if frame < startframe:
                        continue

                    l1 = type_to_classnum(textframe)

                    try:
                        count_per_class[l1[3]] += 1
                    except IndexError:
                        print ('IndexError !!! :: {}'.format(textframe))

                    if d2 == '':
                        print ('IndexError !!! :: {}'.format(textframe))

                    if max > 0.9 :   #
                        total_count += 1
                        if d2 != l1[1]:
                            fail_per_class[l1[3]] += 1
                            if save_file==True:
                                if os.path.exists(full_filename[:-4]) == False:
                                    os.makedirs(full_filename[:-4])
                                dirpath1 = full_filename[:-4] + '/img{:05d}-{}-{}-1.jpg'.format(frame, textframe, d1)
                                dirpath2 = full_filename[:-4] + '/img{:05d}-{}-{}-2.jpg'.format(frame, textframe, d1)
                                org_resize_img = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)))
                                mpimg.imsave(dirpath1, org_resize_img)
                                mpimg.imsave(dirpath2, resize_img)
                            # else:
                                # print ('label={}, predict={}'.format(textframe, d1))
                            # if frame == 2978:
                            #     print ('2978 failed')
                        else:
                            success_count += 1
                            success_per_class[l1[3]] += 1

                cv2.imshow('frame',disp_img)

                key=cv2.waitKey(1) & 0xFF
                if key == ord('t'):
                    display_lowquality = not display_lowquality
                if key == ord('q'):
                    break
                if key == ord('p'):
                    pause = not pause
                    text = './testimageout/img_{0:d}.jpg'.format(frame)
                    cv2.imwrite(text, disp_img)

            print ('video={}, total={}, success={}, avg={}'.format(filename, total_count, success_count, round((success_count/total_count), 2)))
            for idx in range(class_number):
                if success_per_class[idx] > 0:
                   avg = round((success_per_class[idx]/count_per_class[idx]), 2)
                else:
                   avg = 0
                print ('class={}, success={}, fail={}, avg={}'.format(idx, success_per_class[idx], fail_per_class[idx], avg))

            cap.release()

    # if vw is not None:
    #     vw.release()


