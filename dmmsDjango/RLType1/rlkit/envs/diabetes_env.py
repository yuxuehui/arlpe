# coding:utf-8
import os, sys
lib_path = os.path.abspath(os.path.join('../../..'))
result_path =  os.path.join(lib_path, 'results')

mysql_path = os.path.abspath(os.path.join('../..'))
sys.path.append(mysql_path)
from configs.mysql_config import mysql_config

import csv
import os
from sklearn import preprocessing
import numpy as np
import torch
import gym
import pymysql
import torch.nn.functional as F
from gym.spaces.box import Box
import rlkit.torch.pytorch_util as ptu


class DiabetesEnvs:
    def __init__(self, task_num, task_type="adult"):
        self.num = 0
        self.anum = 0
        self.action_space = Box(-np.inf, np.inf, (1,), dtype=np.float64)
        self.observation_space = Box(-np.inf, np.inf, (2,), dtype=np.float64)
        self.person_space = Box(-np.inf, np.inf, (5,), dtype=np.float64)
        self.tasks = range(task_num)
        self._task = self.tasks[0]
        self.task_type = task_type
        self._task_name = self.get_task_name(self._task)
        self.personal_feature_dict = self.get_all_personal_feature()
        self.personal_feature = ptu.FloatTensor(self.personal_feature_dict[self._task_name])   #
        self.demonstrations = self.get_demonstrations()


    def get_demonstrations(self):
        patience_state = []
        path = os.path.join(result_path, 'patientData.txt')
        with open(path, 'r+') as pfile:
            num = 0
            content = pfile.readlines()
            for line in content:
                num += 1
                if num > 1:
                    if int(line.strip().split('\t')[0].split('#')[1]) == self._task+1:
                        info = line.strip().split('\t')[2]
                        state = info.split(',')
                        for i in state:
                            patience_state.append(float(i))
        return np.array(patience_state)   

    def get_all_personal_feature(self):

        current_path = os.path.abspath(__file__)
        father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
        father_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
        father_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
        father_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
        data_dir = config_file_path = os.path.join(father_path,'personal_data', 'data')

        path_list = os.listdir(data_dir)

        print(path_list)
        personal_data = {}
        for data_csv in path_list:
            with open(os.path.join(data_dir, data_csv), 'r', encoding='utf-8') as f:
                resder = csv.reader(f)
                filenames = next(resder)
                csv_reader = csv.DictReader(f, fieldnames=filenames)
                for row in csv_reader:
                    data = {}
                    for k, v in row.items():
                        data[k] = v
                    for k in data.keys():
                        if k == "Name":
                            personal_data[data[k]] = {}
                        elif k == "Daily Basal Insulin (Units)" or k == "OGTT2 (mg/dL)":
                            continue
                        else:
                            personal_data[data["Name"]][k] = data[k]

        data_format = []
        for k in personal_data.keys():
            data_format_temp = []
            for k_low in personal_data[k].keys():
                data_format_temp.append(float(personal_data[k][k_low]))
            data_format.append(data_format_temp)

        min_max_scaler = preprocessing.MinMaxScaler()
        data_format = min_max_scaler.fit_transform(data_format)

        return_data = {}
        i, j = 0, 0
        for k in personal_data.keys():
            return_data[k] = torch.from_numpy(data_format[i]).type(torch.FloatTensor)
            return_data[k] = torch.unsqueeze(return_data[k], 0)
            j = 0
            for k_low in personal_data[k].keys():
                personal_data[k][k_low] = data_format[i][j]
                j = j + 1
            i = i + 1

        # Demo: {'adolescent#001': array([ 0.02291095,  0.17390497,  0.25925926,  0.73333333,  0.3506695 ])}
        return return_data


    def get_all_task_idx(self):

        return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


    def reset(self):
        print('********************reset start *********************')
        flag = False

        while not flag:
            db = pymysql.connect(host=mysql_config.mysql_host, user=mysql_config.mysql_user, password=mysql_config.mysql_password, database=mysql_config.mysql_database)
            cursor = db.cursor()  # 数据库操作
            sql = "select * from state"
            cursor.execute(sql)  # 查询
            db.commit()
            sarr = cursor.fetchall()
            slen = sarr.__len__()
            if slen != 0:
                flag = True
        print(sarr[0][1], sarr[0][2], slen)
        state = [sarr[0][1], sarr[0][2]]
        print('********************reset end *********************')

        # 将state进行归一化
        state[0] = torch.sigmoid(torch.from_numpy(np.array((state[0] - 160) / 18))).numpy()
        state[1] = torch.sigmoid(torch.from_numpy(np.array((state[1] - 160) / 18))).numpy()

        return np.array(state)

    def reward(self, min_state, max_state):
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

    def step(self, action):
        print('**********************step start***************')
        nextstate_flag = False
        while not nextstate_flag:
            db = pymysql.connect(host=mysql_config.mysql_host, user=mysql_config.mysql_user,
                                 password=mysql_config.mysql_password, database=mysql_config.mysql_database)
            cursor = db.cursor()  # 数据库操作
            sql = "select * from sasrecord"
            cursor.execute(sql)  # 查询
            db.commit()
            aarr = cursor.fetchall()
            alen = aarr.__len__()
            if alen == self.num + 1:
                nextstate_flag = True
        pre_state = [aarr[self.num][1], aarr[self.num][2]]
        next_state = [aarr[self.num][4], aarr[self.num][5]]
        self.num += 1
        reward = self.reward(next_state[0], next_state[1])
        info =  {"task_name": self._task_name}
        done = False
        done_bool = float(done)

        next_state[0] = torch.sigmoid(torch.from_numpy(np.array((next_state[0] - 160) / 18))).numpy()
        next_state[1] = torch.sigmoid(torch.from_numpy(np.array((next_state[1] - 160) / 18))).numpy()

        return np.array(next_state), reward, done_bool, info


    def reset_task(self, idx):
        """
        :param idx:
        :return:
        """
        self._task = self.tasks[idx]
        self._task_name = self._task_name.split('#')[0]  # 设置当前任务的名称，例如：“Type1 adult#001”
        if self._task < 9:
            self._task_name = self._task_name + "#00" + str(self._task + 1)
        else:
            self._task_name = self._task_name + "#0" + str(self._task + 1)

        self.personal_feature = ptu.FloatTensor(self.personal_feature_dict[self._task_name])  # 默认为“Type1 adult#001”的
        self.demonstrations = self.get_demonstrations()

    def get_task_name(self, index_temp):
        """
        :param index_temp:
        :return:
        """
        if index_temp < 9:
            return(self.task_type + "#00" + str(index_temp+1))
        else:
            return (self.task_type + "#0" + str(index_temp+1))

    def get_batch_p(self, indices):
        """
        :param indices:
        :return:
        """
        p_list = []
        for idx in indices:
            name_temp = self.get_task_name(idx)
            p_list.append(self.personal_feature_dict[name_temp].cuda())
        return p_list