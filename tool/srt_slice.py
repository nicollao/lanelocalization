import pysrt

srt_path = '/home/hclee/workspace/git_adev2/lane_localization/video/20190110/goPro/MAH08409.srt'

if __name__ == '__main__':
    subs = pysrt.open(srt_path)

    part1 = subs.slice(starts_after={'minutes': 0, 'seconds': -1}, ends_before={'minutes': 17, 'seconds': 38})
    part2 = subs.slice(starts_after={'minutes': 17, 'seconds': 38}, ends_before={'minutes': 35, 'seconds': 9})

    part2.clean_indexes()
    part2.shift(minutes=-17, seconds=-38)
    part1.save(srt_path + '.1', encoding='utf-8')
    part2.save(srt_path + '.2', encoding='utf-8')
