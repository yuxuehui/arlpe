# -*- coding: UTF-8 -*-
import time

import pymysql
import winrm

episode_new = -1
max_episode = 0

def cmdTest():
    global episode_new
    while True:
        episode_flag = False
        while not episode_flag:
            # 查询数据库，是否开始新的一回合
            db = pymysql.connect(host="localhost", user="root", password="123456", database="dmms")  # 打开数据库，配置数据库
            cursor = db.cursor()  # 数据库操作
            sql = "select * from episode"
            cursor.execute(sql)  # 执行数据库语句
            db.commit()  # 提交
            earr = cursor.fetchall()
            elen = earr.__len__()
            print(elen)
            print(episode_new)
            if elen != 0:
                # print('111111111111')
                # print(earr[elen-1][0])
                # print(type(earr[elen-1][0]))
                if episode_new+1 == earr[elen-1][0]:
                    # print('2222222222222')
                    print(earr[elen - 1])
                    episode_flag = True
                    episode_new = earr[elen-1][0]
                    print('episode_new', episode_new)
                    sql = "select * from patient"
                    cursor.execute(sql)  # 执行数据库语句
                    db.commit()  # 提交
                    parr = cursor.fetchall()
                    plen = parr.__len__()
                    print('patient num',plen)
                    p = parr[plen-1][0]
                    s = winrm.Session('http://192.168.0.113:5985/wsman', auth=('jjc', '20100324'))
                    # r = s.run_cmd(
                    #     'D: & cd "D:\\Program Files\\The Epsilon Group\\DMMS.R\\simulator" & DMMS.R C:\\Users\\Administrator\\Documents\\DMMS.R\\config\\FarNosqlNew' + str(p) + '.xml C:\\Users\\Administrator\\Documents\\DMMS.R\\config\\testc.txt')
                    # print(r.std_out)
                    # r = s.run_cmd(
                    #     'D: & cd D:\\virtualenv\\transfertest\\Scripts & activate & D: & cd D:\\controlDMMS & python pythonVariability' + str(p) + '.py')
                    # r = s.run_cmd(
                    #     'D: & cd D:\\virtualenv\\DjangoTest\\Scripts & activate & D: & cd D:\\controlDMMS & python pythonVariability' + str(p) + '.py')  # 加入激活环境以后无法运行，原因不明
                    # r = s.run_cmd('D: & cd D:\\6_virtualenv\\DjangoTest\\Scripts & activate & E: & cd E:\\controlDMMS & python pythonTest.py')
                    # r = s.run_cmd('python D:\\controlDMMS\\test2.py')  # 只产生44条请求，原因不明，DMMS被强制中断 
                    r = s.run_cmd('python D:\\controlDMMS\\pythonVariability' + str(p) + '.py')
                    print('添加变异性：', r.std_out)
                    r = s.run_cmd('D: & cd D:\\DMMS\\simulator & DMMS.R C:\\Users\\jjc\\Documents\\DMMS.R\\config\\VariabilityPatient' + str(p) + '.xml C:\\Users\\jjc\\Documents\\DMMS.R\\config\\testc.txt')
                    print('执行模拟器：', r.std_out)
                    time.sleep(1)

# r = s.run_cmd('cd /d d: & dir')
# 链接服务器 需要在服务器端开启WINRM服务,具体如何开启百度
# s = winrm.Session('http://192.168.0.158:5985/wsman', auth=('Administrator', '123456'))
# r = s.run_cmd('python --version')
# r = s.run_cmd('D: & mkdir software')
# E:\controlDMMS
# r = s.run_cmd('E: & cd E:\\controlDMMS & python pythonVariability1.py')
# r = s.run_cmd('E: & cd E:\\controlDMMS & python pythonTest.py')
# r = s.run_cmd('D: & cd D:\\6_virtualenv\\DjangoTest\\Scripts & activate & E: & cd E:\\controlDMMS & python pythonVariability1.py')
# # r = s.run_cmd('D: & cd D:\\6_virtualenv\\DjangoTest\\Scripts & activate & E: & cd E:\\controlDMMS & python pythonTest.py')
# print(r.std_out)
cmdTest()