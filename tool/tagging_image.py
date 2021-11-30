import cv2, shutil, IPython
import sys, termios, tty, os
import fcntl


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def search(dirname):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        print (full_filename)

def cv2_imshow(img):
    ret = cv2.imencode('.jpg', img)[1].tobytes()
    img_ip = IPython.display.Image(data=ret)
    IPython.display.display(img_ip)

button_delay = 0.2
inputpath = '/home/data/highway_lane/20190409/1/'

if __name__ == '__main__':

    path0 = inputpath + 'tmp'
    path1 = inputpath + '0_00_02_x4_00'
    path2 = inputpath + '1_00_02_x4_00'
    path3 = inputpath + '0_00_03_05_00'
    path4 = inputpath + '1_00_01_x3_00'

    if os.path.exists(path0) == False:
        os.makedirs(path0)

    if os.path.exists(path1) == False:
        os.makedirs(path1)

    if os.path.exists(path2) == False:
        os.makedirs(path2)

    if os.path.exists(path3) == False:
        os.makedirs(path3)

    if os.path.exists(path4) == False:
        os.makedirs(path4)

    # if os.path.exists(os.path.abspath(path1)) == False:
    #     os.makedirs(os.path.dirname(os.path.abspath(path1)))
    #
    # if os.path.exists(os.path.dirname(os.path.abspath(path2))) == False:
    #     os.makedirs(os.path.dirname(os.path.abspath(path2)))
    #
    # if os.path.exists(os.path.dirname(os.path.abspath(path3))) == False:
    #     os.makedirs(os.path.dirname(os.path.abspath(path3)))
    #
    # if os.path.exists(os.path.dirname(os.path.abspath(path4))) == False:
    #     os.makedirs(os.path.dirname(os.path.abspath(path4)))

    fd = sys.stdin.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    filenames = os.listdir(inputpath)
    for filename in filenames:
        full_filename = os.path.join(inputpath, filename)

        img = cv2.imread(full_filename)
        img_resize = cv2.resize(img, (1027, 760))
        cv2.imshow('frame', img_resize)
        key = cv2.waitKey(0)

        if key >= 48 and key < 57:
            if key == 49:               # key : 1
                shutil.move(full_filename, path1)
            elif key == 50:             # key : 2
                shutil.move(full_filename, path2)
            elif key == 51:
                shutil.move(full_filename, path3)
            elif key == 52:
                shutil.move(full_filename, path4)
            else:
                shutil.move(full_filename, path0)
                # os.remove(full_filename)
        elif key == 32:
            # print ("skip file")
            os.remove(full_filename)
        elif key == 117:                # key = 'u'
            print ("undo")
            # os.remove(full_filename)
            shutil.move(prev_filename, path0)
        elif key == 27:
            break

        cv2.destroyAllWindows()
        prev_filename = full_filename

        # p = input()
        # while True:
        #     try:
        #         input("press class number : ")
        #     except EOFError:
        #         continue
        #
            # key = input("press class number : ")
            # if key >= 48 and key < 57:
            #     if key == 49:
            #         shutil.move(filename, path1)
            #     elif key == 50:
            #         shutil.move(filename, path2)
            #     elif key == 51:
            #         shutil.move(filename, path3)
            #     elif key == 52:
            #         shutil.move(filename, path4)
            # elif key == 27:
            #     break
