import time


import os, sys
lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)
from configs.mysql_config import mysql_config

DMMS_ip = ""


import pymysql
import winrm
episode_new = -1
max_episode = 0

class Patient:
    def __init__(self):
        self.person = 1

def cmdTest():
    global episode_new
    while True:
        episode_flag = False
        while not episode_flag:
            db = pymysql.connect(host=mysql_config.mysql_host, user=mysql_config.mysql_user,password=mysql_config.mysql_password, database=mysql_config.mysql_database)
            cursor = db.cursor()
            sql = "select * from episode"
            cursor.execute(sql)
            db.commit()
            earr = cursor.fetchall()
            elen = earr.__len__()
            if elen != 0:
                if episode_new+1 == earr[elen-1][0]:
                    print(earr[elen - 1])
                    episode_flag = True
                    episode_new = earr[elen-1][0]
                    sql = "select * from patient"
                    cursor.execute(sql)  # 执行数据库语句
                    db.commit()  # 提交
                    parr = cursor.fetchall()
                    plen = parr.__len__()
                    p = parr[plen-1][0]
                    s = winrm.Session(DMMS_ip, auth=('XXX', 'XXX'))
                    r = s.run_cmd('XXXX')
                    time.sleep(1)


cmdTest()