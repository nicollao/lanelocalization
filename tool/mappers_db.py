#-*- coding: utf-8 -*-

'''
Created on 2018. 8. 17.

@author: Mappers
'''

import platform
import cx_Oracle
import os
import pandas as pd
from datetime import datetime
from smb.SMBConnection import SMBConnection
import shutil
import threading
import math

class MappersDBModule:
    def __init__(self, db_id, db_pw, db_path, db_encoding = 'UTF-8'):
        # self.db = cx_Oracle.connect(db_id, db_pw, db_path, encoding = db_encoding)
        # self.cursor = self.db.cursor()
        self.media_server_ip_list = {
            'MEDIASERVER04':'172.16.102.81',
            'MEDIASERVER05':'172.16.102.82',
            'MEDIASERVER06':'172.16.102.83',
            'MEDIASERVER07':'172.16.102.84',
            'MEDIASERVER08':'172.16.102.85',
            'MEDIASERVER09':'172.16.102.86',
            'MEDIASERVER10':'172.16.102.87',
            'MEDIASERVER11':'172.16.102.88',
            'MEDIASERVER12':'172.16.102.89',
            'MEDIASERVER13':'172.16.102.90',
            'MEDIASERVER14':'172.16.102.93',
            'MEDIASERVER15':'172.16.102.94',
            'MEDIASERVER16':'172.16.102.95',
            'MEDIASERVER17':'172.16.102.96',
            'MEDIASERVER18':'172.16.102.97',
            'MEDIASERVER19':'172.16.102.98',
        }
        # self.smbConns = {}
        # self.make_smb_conns()
        # self.smbConns = self.make_smb_conns()

        pass
    
    def make_smb_conns(self):

        smbConns = {}
        for server_name in self.media_server_ip_list.keys():
            conn = SMBConnection('Guest', '', 'gtx1080', server_name, domain='WORKGROUP', use_ntlm_v2=True)
            conn.connect(self.media_server_ip_list[server_name], 139)
            smbConns[server_name]=conn
        
        return smbConns
    
    def close_smb_conns(self, smbConns):
        for server_name in self.media_server_ip_list.keys():
            conn = smbConns[server_name]
            conn.close()

    # def close_smb_conns(self, ):
    #     for server_name in self.media_server_ip_list.keys():
    #         conn = self.smbConns[server_name]
    #         conn.close()

    def run_query(self, query):
        res = self.cursor.execute(query)

        # keys = ['IMG_PATH', 'UTMK_X', 'UTMK_Y', 'MM_UTMK_X', 'MM_UTMK_Y', 'PHOTO_SEQ']
        keys = [d[0] for d in self.cursor.description]

        return res, keys
    
    def save_result(self, output_img_path, mapping_file, res):
        #output folder check
        if not os.path.exists(output_img_path):
            os.makedirs(output_img_path)

        if platform.system() != 'Windows':
            smbConns = self.make_smb_conns()
        
        with open(mapping_file, mode = 'w', encoding = 'utf-8') as f:
            filecnt = 1
            # opener = urllib.request.build_opener(SMBHandler)
            
            for line in res:
                real_path = line[0]
                # print(real_path)
                # utmk_x, utmk_y = line[1], line[2]
                # mm_utmk_x, mm_utmk_y = line[3], line[4]
                # photo_seq = line[5]
            
                cur_fname = '[{}]_img.jpg'.format(filecnt)
                # f.write('{},{},{},{},{},{},{}\n'.format(cur_fname, real_path, utmk_x, utmk_y, mm_utmk_x, mm_utmk_y, photo_seq))

                # 2018.08.21 platform에 맞게 분기
                if platform.system() != 'Windows':
                    imgpath = 'smb:' + real_path.replace('\\\\', '//')
                    imgpath = imgpath.replace('\\', '/')
                    sIdxOfservername = imgpath.find('://') + 3
                    lIdxOfservername = imgpath.find('/', sIdxOfservername)
                    
                    servername = imgpath[sIdxOfservername:lIdxOfservername].upper()

                    # 2018.08.22 smb 속도 개선
                    lIdxOfsharefolder = imgpath.find('/', lIdxOfservername+1)
                    shareFoldername = imgpath[lIdxOfservername+1:lIdxOfsharefolder]
                    filepath = imgpath[lIdxOfsharefolder:]
                    
                    conn = smbConns[servername]

                    ofile = open(os.path.join(output_img_path, cur_fname), 'wb')
                    try:
                        _, _ = conn.retrieveFile(shareFoldername, filepath, ofile, 60)
                    except BaseException:
                        continue
                    ofile.close()

                else:
                    #im = Image.open(real_path)
                    #im.save(os.path.join(output_img_path, cur_fname))
                    # 2018-12-12
                    with open(real_path, mode = 'rb') as fp:
                        byte_data = fp.read()
                    
                    with open(os.path.join(output_img_path, cur_fname), mode = 'wb') as fp:
                        fp.write(byte_data)
                    
                filecnt += 1
                
            print('{} total {} img data is saved'.format(output_img_path, filecnt - 1))
        
        if platform.system() != 'Windows':
            self.close_smb_conns(smbConns)

    def make_query_for_finding_mediaserver_path(self, target_date, target_user_name, from_time = '08', to_time = '22'):
        tmp_select = 'distinct(photo.media01) as MEDIA01'
        tmp_from = 'networkuser.tbw_nphotolog photo'
        tmp_where = 'photo.photo_dt >= to_date(\'{}\', \'yyyymmddhh24\') '.format(target_date + from_time)\
                  + 'and photo.photo_dt < to_date(\'{}\', \'yyyymmddhh24\') '.format(target_date + to_time)\
                  + 'and photo.fo_img_flag = 1 '\
                  + 'and photo.user_nm = \'{}\''.format(target_user_name)
        
        with_tmp_query = 'with tmp\n'\
                       + 'as\n(\n\t' \
                       + 'select {} from {} where {}\n)'.format(tmp_select, tmp_from, tmp_where)
    
        real_select = 'server.mediapath'
        real_from = 'networkuser.TBDMEDIASERVER server, tmp'
        real_where = 'server.MEDIAPATH not like \'\\\\MEDIASERVER\\%\' '\
                   + 'and server.MEDIAPATH not like \'\\\\MEDIASERVER2\\%\' '\
                   + 'and server.MEDIAPATH not like \'\\\\MEDIASERVER3\\%\' '\
                   + 'and tmp.MEDIA01 = server.MEDIA01'
        
        real_query = '{}\nselect {} from {} where {} '.format(with_tmp_query, real_select, real_from, real_where)
        return real_query

    
    # make_query('20180702', '20180702', '09', '12')
    # def make_query(self, from_date, to_date, from_time, to_time, row_limit = 0):
    def make_query(self, username, date, from_time, to_time, row_limit = 0):
        tmp_select_list = 'photo.log_dt, photo.user_nm, photo.photo_seq, photo_dt,' \
                        + 'photo.media01, photo.utmk_x, photo.utmk_y, photo.mm_utmk_x,' \
                        + 'photo.mm_utmk_y'
        tmp_from = 'networkuser.tbw_nphotolog photo'
        tmp_where = 'photo.USER_NM = \'{}\' AND LOG_DT = {} '.format(username, date) \
                    + 'AND photo.photo_dt >= to_date(\'{}\', \'yyyymmddhh24\') '.format(date + from_time) \
                    + 'AND photo.photo_dt < to_date(\'{}\', \'yyyymmddhh24\') ORDER BY PHOTO_SEQ'.format(date + to_time)\

        with_tmp_query = 'with tmp\n' \
                       + 'as\n(\n\t' \
                       + 'select {} from {} where {}\n)'.format(tmp_select_list, tmp_from, tmp_where)
    
        real_select_list = 'server.mediapath    ||'\
                         + 'tmp.LOG_DT        ||\'\\\'||'\
                         + 'tmp.USER_NM       ||\'\\\'||'\
                         + '\'FO\'            ||\'\\\'||'\
                         + 'to_char(tmp.PHOTO_DT, \'hh24\') ||\'\\\'||'\
                         + '\'[\'|| tmp.PHOTO_SEQ ||\']\'|| to_char(tmp.PHOTO_DT, \'hh24miss\') || \'-0.jpg\' '\
                         + 'as IMG_PATH, '\
                         + 'tmp.utmk_x, tmp.utmk_y, tmp.mm_utmk_x, tmp.mm_utmk_y, tmp.photo_seq'
        real_from = 'networkuser.TBDMEDIASERVER server, tmp'
        real_where = 'server.MEDIAPATH not like \'\\\\MEDIASERVER\\%\' '\
                   + 'and server.MEDIAPATH not like \'\\\\MEDIASERVER2\\%\' '\
                   + 'and server.MEDIAPATH not like \'\\\\MEDIASERVER3\\%\' '\
                   + 'and tmp.MEDIA01 = server.MEDIA01'
        
        if row_limit > 0:
            real_where += ' and rownum <= {}'.format(row_limit)
        
        real_query = '{}\nselect {} from {} where {} '.format(with_tmp_query, real_select_list, real_from, real_where)
        # real_query = '{}\nselect * from {} where {} '.format(with_tmp_query, real_from, real_where)
        return real_query


    def save_result_from_smb(self,data, output_img_path, start, end):
        # output folder check
        filecnt = 1
        sub_dir_name = 1

        #sub_img_path = '{}{}'.format(output_img_path, sub_dir_name)
        #if not os.path.exists(sub_img_path):
        #    os.makedirs(sub_img_path)

        if platform.system() != 'Windows':
            smbConns = self.make_smb_conns()


        for i in range(start, end):
            real_path = data.iloc[i].IMG_PATH
            # f.write('{},{},{},{},{},{},{}\n'.format(cur_fname, real_path, utmk_x, utmk_y, mm_utmk_x, mm_utmk_y, photo_seq))

            #if filecnt % 1000 == 0:
            #    # print ("filecnt : ",  filecnt)
            #    sub_dir_name += 1
            #    sub_img_path = '{}{}'.format(output_img_path, sub_dir_name)
            #    if not os.path.exists(sub_img_path):
            #        os.makedirs(sub_img_path)


            # 2018.08.21 platform에 맞게 분기
            if platform.system() != 'Windows':
                # imgpath = 'smb:' + real_path.replace('\\\\', '//')
                # imgpath = imgpath.replace('\\', '/')
                imgpath = 'smb:' + real_path
                sIdxOfservername = imgpath.find('://') + 3
                lIdxOfservername = imgpath.find('/', sIdxOfservername)
                # cur_fname = '[{}]_{}'.format(filecnt, real_path.split('/')[-1])
                token = real_path.split('/')
                #cur_fname = '{}_{}_{}_{}'.format(token[4], token[5], token[7], token[8])
                cur_fname = token[8]
                sub_img_path = os.path.join(output_img_path, token[4], token[5], token[7])

                if not os.path.exists(sub_img_path):
                    os.makedirs(sub_img_path)

                servername = imgpath[sIdxOfservername:lIdxOfservername].upper()

                # 2018.08.22 smb 속도 개선
                lIdxOfsharefolder = imgpath.find('/', lIdxOfservername + 1)
                shareFoldername = imgpath[lIdxOfservername + 1:lIdxOfsharefolder]
                filepath = imgpath[lIdxOfsharefolder:]

                conn = smbConns[servername]

                ofile = open(os.path.join(sub_img_path, cur_fname), 'wb')
                #ofile = open(os.path.join(cur_fname), 'wb')
                try:
                    _, _ = conn.retrieveFile(shareFoldername, filepath, ofile, 60)
                except BaseException:
                    continue
                ofile.close()

            else:
                # im = Image.open(real_path)
                # im.save(os.path.join(output_img_path, cur_fname))
                # 2018-12-12
                with open(real_path, mode='rb') as fp:
                    byte_data = fp.read()

                with open(os.path.join(output_img_path, cur_fname), mode='wb') as fp:
                    fp.write(byte_data)
            filecnt += 1

        print('{} total {} img data is saved'.format(output_img_path, filecnt - 1))

        if platform.system() != 'Windows':
            self.close_smb_conns(smbConns)


def imageDownload(username, date, from_time, to_time):
    # database용 hyper-parameters
    db_id, db_pw, db_path = 'contentsteam', 'zjsxpscm', '172.16.103.22:1521/IMDB'
    # make db connection
    # print('Load database Module...')

    mdbM = MappersDBModule(db_id, db_pw, db_path)
    cur_query = mdbM.make_query(username, date, from_time, to_time, row_limit = 0)
    reslist, keys = mdbM.run_query(cur_query)

    outpath = '/home/data/highway_lane/20190410/{}_{}'.format(date, username)
    mdbM.save_result(outpath, 'indo.txt', reslist)

    def save_result(self, output_img_path, mapping_file, res):
        # output folder check
        if not os.path.exists(output_img_path):
            os.makedirs(output_img_path)

        if platform.system() != 'Windows':
            smbConns = self.make_smb_conns()

        with open(mapping_file, mode='w', encoding='utf-8') as f:
            filecnt = 1
            # opener = urllib.request.build_opener(SMBHandler)

            for line in res:
                real_path = line[0]
                # print(real_path)
                # utmk_x, utmk_y = line[1], line[2]
                # mm_utmk_x, mm_utmk_y = line[3], line[4]
                # photo_seq = line[5]

                cur_fname = '[{}]_img.jpg'.format(filecnt)
                # f.write('{},{},{},{},{},{},{}\n'.format(cur_fname, real_path, utmk_x, utmk_y, mm_utmk_x, mm_utmk_y, photo_seq))

                # 2018.08.21 platform에 맞게 분기
                if platform.system() != 'Windows':
                    imgpath = 'smb:' + real_path.replace('\\\\', '//')
                    imgpath = imgpath.replace('\\', '/')
                    sIdxOfservername = imgpath.find('://') + 3
                    lIdxOfservername = imgpath.find('/', sIdxOfservername)

                    servername = imgpath[sIdxOfservername:lIdxOfservername].upper()

                    # 2018.08.22 smb 속도 개선
                    lIdxOfsharefolder = imgpath.find('/', lIdxOfservername + 1)
                    shareFoldername = imgpath[lIdxOfservername + 1:lIdxOfsharefolder]
                    filepath = imgpath[lIdxOfsharefolder:]

                    conn = smbConns[servername]

                    ofile = open(os.path.join(output_img_path, cur_fname), 'wb')
                    try:
                        _, _ = conn.retrieveFile(shareFoldername, filepath, ofile, 60)
                    except BaseException:
                        continue
                    ofile.close()

                else:
                    # im = Image.open(real_path)
                    # im.save(os.path.join(output_img_path, cur_fname))
                    # 2018-12-12
                    with open(real_path, mode='rb') as fp:
                        byte_data = fp.read()

                    with open(os.path.join(output_img_path, cur_fname), mode='wb') as fp:
                        fp.write(byte_data)

                filecnt += 1

            print('{} total {} img data is saved'.format(output_img_path, filecnt - 1))

        if platform.system() != 'Windows':
            self.close_smb_conns(smbConns)



if __name__ == '__main__':

    # database용 hyper-parameters
    db_id, db_pw, db_path = 'contentsteam', 'zjsxpscm', '172.16.103.22:1521/IMDB'
    # make db connection
    # print('Load database Module...')
    mdbM = MappersDBModule(db_id, db_pw, db_path)
    # print('Completed..')

    # f = open('경부고속도로.csv', 'r')
    # lines = f.readlines()
    # index = 0
    # for mm_linkid in lines:
        # mm_linkid = data.iloc[index]['LINKD_ID'].strip()
        # make query


    # 이재훈, 20170424 10 -> 12
    # 오주인, 20190401 10 -> 12
    # 염창훈, 20161121 11 -> 13
    # 이재훈, 20170321 10 -> 12

    # 고광용, 20190405 08 -> 10
    # 고광용, 20190315 07 -> 12
    # 서성훈, 20190401 09 -> 13

    # test1 = threading.Thread(target=imageDownload, args=('고광용', '20190405', '08', '10'))
    # test1.start()
    # test2 = threading.Thread(target=imageDownload, args=('고광용', '20190315', '07', '12'))
    # test2.start()
    # test3 = threading.Thread(target=imageDownload, args=('서성훈', '20190401', '09', '13'))
    # test3.start()
    # test4 = threading.Thread(target=imageDownload, args=('이재훈', '20170321', '10', '12'))
    # test4.start()

    # for i in lstDownload:
    #     cur_query = mdbM.make_query(i[0], i[1],  row_limit = 0)
    #     reslist, keys = mdbM.run_query(cur_query)
    #
    #     outpath = '/home/data/highway_lane/20190410/{}_{}'.format(i[0], i[1])
    #     mdbM.save_result(outpath, 'indo.txt', reslist)

    # index = index + 1

        # print('table keys: ', keys)
        # for line in reslist:
        #     real_path = line[0]
        #     shutil.copy(real_path, '/home/data/highway_lane/20190409/')
            # utmk_x, utmk_y = line[1], line[2]
            # mm_utmk_x, mm_utmk_y = line[3], line[4]
            # photo_seq = line[5]
            #
            # print('*', line)

    # if platform.system() != 'Windows':
    #     mdbM.close_smb_conns()

    # data = pd.read_csv('경부고속도로_UTF8_2019.csv')
    data = pd.read_csv('진출입_이미지.txt')
    num_of_img = len(data)
    # num_of_img = 10000

    offset = math.ceil(num_of_img / 8)
    path = {}
    thread_id = {}

    # path = '/home/data/highway_lane/20190411-1/'
    # th = threading.Thread(target=mdbM.save_result_from_smb, args=(data, path, 1, 3000))
    # th.start()

    for i in range(0, 8):
        path = '/home/data/highway_lane/20190610-innout/'
        start_idx = (offset * i)
        end_idx = (offset * i) + offset
        print (start_idx, end_idx)

        thread_id[i] = threading.Thread(target=mdbM.save_result_from_smb, args=(data, path, start_idx, end_idx))
        thread_id[i].start()
