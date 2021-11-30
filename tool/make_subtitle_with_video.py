import cv2

# def read_kbd_input(inputQueue):
#     print('Ready for keyboard input:')
#     while (True):
#         input_str = input()
#         inputQueue.put(input_str)
#
# def main():
#     EXIT_COMMAND = "exit"
#     inputQueue = queue.Queue()
#
#     inputThread = threading.Thread(target=read_kbd_input, args=(inputQueue,), daemon=True)
#     inputThread.start()
#
#     while (True):
#         if (inputQueue.qsize() > 0):
#             input_str = inputQueue.get()
#             print("input_str = {}".format(input_str))
#
#             if (input_str == EXIT_COMMAND):
#                 print("Exiting serial terminal.")
#                 break
#
#             Insert your code here to do whatever you want with the input_str.
        #
        # The rest of your program goes here.
        #
        # time.sleep(0.01)
    # print("End.")



# video_name = '/home/hclee/workspace/git_adev2/lane_localization/video/서하남-구리1.mp4'
video_name = '/home/hclee/workspace/git_adev2/lane_localization/video/20190110/sony1/movie/MAH08411.MP4'
# video_name = "/home/data/highway_lane/video/20190226/sony1.MAH01283.MP4"



caption_name = video_name[:-4] + '.srt'
capture = cv2.VideoCapture(video_name)

if capture.isOpened() == False:
    print('capture is not open')

fps = capture.get(cv2.CAP_PROP_FPS)

mode = 0
srt_index = 0
startframe = 0
endframe = 0
fps_count = 0
enterkey = 0
writeSubtitle = False

text = "waiting input code ('s') :: "

while True:
    if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        # capture.open("/home/hclee/workspace/git_adev2/lane_localization/video/서하남-구리1.mp4")
        # capture.open("/home/data/highway_lane/video/20190226/sony1.MAH01283.MP4")
        capture.open(video_name)

    ret, frame = capture.read()
    fps_count = fps_count + 1

    # if cv2.waitKey(33) > 0: break
    # try:
    #     number = nput("what's your name???")
    # except ValueError:
    #     print("Invalid number")

    key = cv2.waitKey(9)

    if key >= 48 and key < 57:
        if mode == 0:
            if key == 49:
                text = '000010400'
                number = '000010400'
            elif key == 50:
                text = '000020400'
                number = '000020400'
            elif key == 51:
                text = '000030400'
                number = '000030400'
            elif key == 52:
                text = '000040400'
                number = '000040400'
            elif key == 53:
                text = '100020400'
                number = '100020400'

            enterkey = key
            mode = 1
            startframe = fps_count
            srt_index = srt_index + 1
        # elif mode == 1 and enterkey == key:
        # elif mode == 1:
        #     mode = 0
        #     endframe = fps_count
        #     writeSubtitle = True
    # spacebar
    elif key == 32:
        if mode == 1:
            mode = 0
            endframe = fps_count
            writeSubtitle = True
            number = '00000000'
            text = "ready subtitle~"
        else:
            mode = 1
            startframe = fps_count
            srt_index = srt_index + 1
            text = "save subtitle~"
    elif key == ord('s'):
    # if key == ord('s'):
        try:
            number = input("enter label code :: ")
            text = number
        except ValueError:
            print("Invalid number")

        enterkey = key
        startframe = fps_count
        srt_index = srt_index + 1
    elif key == ord('e'):
        endframe = fps_count
        writeSubtitle = True
        # text = "input code ('s') :: "
    elif key == 27:
        break
    else:
        output = frame

    if writeSubtitle == True:
        startTime = '{:02d}:{:02d}:{:02d},00'.format(int((startframe / fps) // 3600), int((startframe / fps) % 3600 // 60), int((startframe / fps) % 60))
        endTime = '{:02d}:{:02d}:{:02d},00'.format(int((endframe / fps) // 3600), int((endframe / fps) % 3600 // 60), int((endframe / fps) % 60))
        srtfile = open(caption_name, mode='a', encoding='utf-8')
        str = (("%d\n%s --> %s\n%s\n\n") % (srt_index, startTime, endTime, number))
        srtfile.write(str)
        srtfile.close()
        # text = 'please input code :: '
        number = ''
        writeSubtitle = False

    # Write some Text

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(frame, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    imS = cv2.resize(frame, (480, 270))
    cv2.imshow("VideoFrame", imS)

capture.release()
cv2.destroyAllWindows()
