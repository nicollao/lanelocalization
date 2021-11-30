import sys
import pysrt

srt_path = '/home/hclee/workspace/git_adev2/lane_localization/video/20190108/sony2/MAH08406_1204_1024.srt'

if __name__ == '__main__':

    # if len(sys.argv) == 1:  # 옵션 없으면 도움말 출력하고 종료
    #     print ('변경할 srt 파일명을 입력해 주세요')
    #     exit(1)

    # print(sys.argv[1], sys.argv[2])
    subs = pysrt.open(srt_path)

    # subs.shift(minutes=-17, seconds=-38)  # Move all subs 2 seconds earlier
    subs.shift(seconds=+1)  # Move all subs 2 seconds earlier
    # subs.shift(seconds=-23)  # Move all subs 1 minutes later
    # >> > subs.shift(ratio=25 / 23.9)  # convert a 23.9 fps subtitle in 25 fps

    subs.save(srt_path + '.1', encoding='utf-8')
