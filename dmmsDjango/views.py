import os, sys
from RLType1.configs.mysql_config import mysql_config

from django.http import HttpResponse, HttpRequest
import pymysql
from numpy import double
import torch.nn.functional as F

import os, sys
lib_path = os.path.abspath(os.path.join('../../../..'))
result_path = os.path.join(lib_path, 'results')

import torch
import numpy as np
import time

actiont_re = []
dayt_episode = 0
statet_re = []


def dayaction(request):
    print(request.get_full_path())
    getaction_flag = False
    with open(os.path.join(result_path, 'timere.txt', 'a+')) as timefile:
        timefile.write('receive service:\t' + time.strftime("%Y-%m-%d %H:%M:%S",
                                                            time.localtime()) + '\trequest:' + request.get_full_path() + '\n')
    timea = request.GET.get('p1')
    min_bg = double(request.GET.get('p2'))
    max_bg = double(request.GET.get('p3'))
    bG = [min_bg, max_bg]  # 大小？？？
    statet_re.append(bG)
    # if len(statet_re) == 11 or len(statet_re) == 21 or len(statet_re) == 31:
    #     time.sleep(60)
    print(statet_re)
    print(len(statet_re))
    global dayt_episode
    db = pymysql.connect(host=mysql_config.mysql_host, user=mysql_config.mysql_user,password=mysql_config.mysql_password, database=mysql_config.mysql_database)
    cursor = db.cursor()  # 数据库操作
    sql = "insert into state(id,minstate,maxstate)values('%d','%lf',%lf)" % (
        len(statet_re) - 1, min_bg, max_bg)  # 存入数据库
    cursor.execute(sql)  # 执行数据库语句
    db.commit()  # 提交

    if len(actiont_re) != 0:  # 在每一天开头得到上一个动作引起的状态
        re = len(actiont_re)
        next_state = bG
        pre_state = statet_re[re - 1]
        r = reward(bG[0], bG[1])
        db = pymysql.connect(host=mysql_config.mysql_host, user=mysql_config.mysql_user,password=mysql_config.mysql_password, database=mysql_config.mysql_database)
        cursor = db.cursor()  # 数据库操作
        sql = "insert into sasrecord(id,pre_min,pre_max,action,next_min,next_max)values('%d','%lf',%lf,'%lf',%lf,'%lf')" % (
            re - 1, pre_state[0], pre_state[1], actiont_re[re - 1], next_state[0], next_state[1])  # 存入数据库
        cursor.execute(sql)  # 执行数据库语句
        db.commit()  # 提交
        with open(os.path.join(result_path, 'timere.txt', 'a+')) as timefile:
            dfile.write(
                '\tnext_state:' + str(next_state[0]) + '\t' + str(next_state[1]) + '\treward:' + str(r) + '\n')
        if len(actiont_re) == 59:  # 因为模拟器会多跑一个，可修改
            actiont_re.clear()
            statet_re.clear()
            print(len(statet_re))
            return HttpResponse('Episode end')

    while not getaction_flag:  # 2021-03-29加
        db = pymysql.connect(host=mysql_config.mysql_host, user=mysql_config.mysql_user,password=mysql_config.mysql_password, database=mysql_config.mysql_database)
        cursor = db.cursor()  # 数据库操作
        sql = "select * from action"
        cursor.execute(sql)  # 查询
        db.commit()
        actionarr = cursor.fetchall()
        actionlen = actionarr.__len__()
        if actionlen == len(actiont_re) + 1:  # 得到了新的动作
            getaction_flag = True
            action = actionarr[len(actiont_re)][1]
    print(action)
    print(type(action))
    value = (action * 6000.0) / 24.0
    actiont_re.append(value)
    print(actiont_re, len(actiont_re))
    with open(os.path.join(result_path, 'dataDeal2.txt', 'a+')) as dfile:
        dfile.write('state:' + str(bG[0]) + '\t' + str(bG[1]) + '\tnet_action:' + str(
            action) + '\taction:' + str(value) + '\tMeals:\t6:00,59g,15min\t12:00,79g,15min\t18:00,59g,15min')

    with open(os.path.join(result_path, 'timere.txt', 'a+')) as timefile:
        timefile.write('finish service:\t' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\tdatare:\t' + str(
            timea) + '\ttime:' + request.get_full_path() + '\n')
    return HttpResponse(str(value))
    
def reward( min_state, max_state):
    min_state, max_state = torch.from_numpy(np.array(min_state)), torch.from_numpy(np.array(max_state)) # 转为tensor
    score_offset = abs(F.relu(70 - max_state) + F.relu(max_state - 180))
    score_center = abs(125 - max_state)
    score_center_plus = abs(min(180, max(70, max_state)) - 125)
    score = 24 - score_offset
    score_center = 24 - score_center
    score_center_plus = 24 - score_offset - 0.2 * score_center_plus
    score_offset_y = abs(F.relu(70 - min_state) + F.relu(min_state - 180))
    score_center_y = abs(125 - min_state)
    score_center_plus_y = abs(abs(min(180, max(70, min_state))) - 125)
    score_y = 24 - score_offset_y
    score_center_y = 24 - score_center_y
    score_center_plus_y = 24 - score_offset_y - 0.2 * score_center_plus_y
    score = score_center_plus + score_center_plus_y
    r = score.numpy()
    return r


def LGSaction(request):
    print(request.get_full_path())
    getaction_flag = False
    with open(os.path.join(result_path, 'timere.txt', 'a+')) as timefile:
        timefile.write('receive service:\t' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\trequest:' + request.get_full_path() + '\n')
    timea = request.GET.get('p1')
    state = double(request.GET.get('p2'))
    patient = int(request.GET.get('p4'))
    
    
    value = [123.48, 136.08, 151.2, 95.76, 92.4, 191.94, 126, 106.68, 95.34, 126]
    with open(os.path.join(result_path, 'timere.txt', 'a+')) as timefile:
        timefile.write('finish service:\t' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\tdatare:\t' + str(timea) + '\ttime:' + request.get_full_path() + '\n')
    print('patient:', patient, 'action:', value[patient-1])
    return HttpResponse(str(value[patient-1]))

def Test(request):
    return HttpResponse('test')
