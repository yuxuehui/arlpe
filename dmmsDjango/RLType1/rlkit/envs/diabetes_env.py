# coding:utf-8
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
        self.num = 0  # 记录步数
        self.anum = 0  # 记录产生的动作数量
        self.action_space = Box(-np.inf, np.inf, (1,), dtype=np.float64)
        self.observation_space = Box(-np.inf, np.inf, (2,), dtype=np.float64)
        self.person_space = Box(-np.inf, np.inf, (5,), dtype=np.float64)

        self.tasks = range(task_num)       # 所有task的list，即[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self._task = self.tasks[0]  # 当前task, 默认为 task 0，即“adult#001”
        self.task_type = task_type
        self._task_name = self.get_task_name(self._task)     # 设置当前任务的名称，例如：“adult#001”


        ### 需要定义当前 task 的 meal，大剂量的值，当前患者的体征

        ### 患者个人信息，年龄等
        self.personal_feature_dict = self.get_all_personal_feature()    # 默认为“Type1 adult#001”的
        self.personal_feature = ptu.FloatTensor(self.personal_feature_dict[self._task_name])   # 默认为“Type1 adult#001”的

        ### 患者第一天血糖值
        self.demonstrations = self.get_demonstrations()


    def get_demonstrations(self):
        """
        读取数据库，返回 self._task_name 对应的第一天的血糖数据
        :return: np.array 的格式
        """
        patience_state = []
        path = '/home/Data/yanlian/yxhfile/dmmsDjSA_end/dmmsDjango/results/patientData.txt'
        with open(path, 'r+') as pfile:
            num = 0
            content = pfile.readlines()
            for line in content:
                num += 1
                if num > 1:
                    # if int(line.strip().split('\t')[0].split('#')[1]) == (self._task+1):
                    if int(line.strip().split('\t')[0].split('#')[1]) == self._task+1:
                        info = line.strip().split('\t')[2]
                        #print(info)
                        state = info.split(',')
                        #print(len(state))
                        for i in state:
                            patience_state.append(float(i))
                        #print(patience_state)
        return np.array(patience_state)   

    def get_all_personal_feature(self):
        ### 1、
        # 获取当前文件路径
        current_path = os.path.abspath(__file__)
        # 获取当前文件的父目录
        father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
        father_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
        father_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
        father_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
        # 获取 personal data 数据
        data_dir = config_file_path = os.path.join(father_path,'personal_data', 'data')

        path_list = os.listdir(data_dir)

        print(path_list)
        personal_data = {}
        for data_csv in path_list:
            with open(os.path.join(data_dir, data_csv), 'r', encoding='utf-8') as f:
                resder = csv.reader(f)
                filenames = next(resder)  # 获取数据的第一列，作为后续要转为字典的键名 生成器，next方法获取
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

        ### 进行归一化
        data_format = []
        for k in personal_data.keys():
            data_format_temp = []
            for k_low in personal_data[k].keys():
                data_format_temp.append(float(personal_data[k][k_low]))
            data_format.append(data_format_temp)

        # print(data_format)
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

        # 示例：{'adolescent#001': array([ 0.02291095,  0.17390497,  0.25925926,  0.73333333,  0.3506695 ])}
        return return_data


    def get_all_task_idx(self):
        """
        :return: 所有任务的index；list；list中的内容为int
        """
        return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


    def reset(self):
        """
        重置当前task的环境，通过调用命令行，重新run第一天
        :return: 环境的初始化状态。
        """
        print('********************reset start *********************')
        flag = False
        # s = winrm.Session('http://192.168.0.158:5985/wsman', auth=('Administrator', '123456'))
        # r = s.run_cmd(
        #     'D: & cd "D:\\Program Files\\The Epsilon Group\\DMMS.R\\simulator" & DMMS.R C:\\Users\\Administrator\\Documents\\DMMS.R\\config\\FarNosql.xml C:\\Users\\Administrator\\Documents\\DMMS.R\\config\\testc.txt')
        # print(r.std_out)
        # while len(state_re) == 0:
        #     print(state_re)
        #     pass
        # if len(state_re) != 0:
        #     print(state_re)
        while not flag:
            db = pymysql.connect(host="localhost", user="root", password="123456", database="dmms")  # 打开数据库，配置数据库
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
        # ## reward 设计1、2
        # f1 = max(180 - max_state, 0)  # 230改为180
        # f2 = max(min_state - 70, 0)
        # f3 = max(max_state - 130, 0)
        # f4 = max(150 - min_state,0)
        # # r = 0.1 * f1 + 0.01 * f2
        # # r = 0.5 * f1 + 0.5 * f2
        # ## reward function 2
        # a_max = f2/(f2+f1+1)
        # a_min = f1/(f2+f1+1)
        # r = a_max*f1 + a_min*f2
        # ## reward function 1
        # # r = -0.5*f1-0.5*f2-0.5*f3-0.5*f4+80

        ### reward 设计3
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

        ## reward 设计3_2
        # min_state, max_state = torch.from_numpy(np.array(min_state)), torch.from_numpy(np.array(max_state))  # 转为tensor
        # score_offset = abs(F.relu(70 - max_state) + F.relu(max_state - 150))
        # score_center = abs(105 - max_state)
        # score_center_plus = abs(min(180, max(70, max_state)) - 105)
        # score = 24 - score_offset
        # score_center = 24 - score_center
        # score_center_plus = 24 - score_offset - 0.2 * score_center_plus
        #
        # score_offset_y = abs(F.relu(130 - min_state) + F.relu(min_state - 180))
        # score_center_y = abs(155 - min_state)
        # score_center_plus_y = abs(abs(min(180, max(130, min_state))) - 155)
        # score_y = 24 - score_offset_y
        # score_center_y = 24 - score_center_y
        # score_center_plus_y = 24 - score_offset_y - 0.2 * score_center_plus_y
        #
        # score = score_center_plus + score_center_plus_y
        # r = score.numpy()

        return r

    def step(self, action):
        """
        将action传输给模拟器端；
        由于将action传给模拟器之后，运行一段时间才能得到下一时刻状态，所以
        :param action:
        :return: next_o, r, d, env_info ： 下一时刻状态，reward，是否结束（直接返回0即可），环境信息（self._task_name即可）
        """
        print('**********************step start***************')
        nextstate_flag = False
        while not nextstate_flag:
            db = pymysql.connect(host="localhost", user="root", password="123456", database="dmms")  # 打开数据库，配置数据库
            cursor = db.cursor()  # 数据库操作
            sql = "select * from sasrecord"
            cursor.execute(sql)  # 查询
            db.commit()
            aarr = cursor.fetchall()
            alen = aarr.__len__()
            if alen == self.num + 1:
                nextstate_flag = True
        print('self.num', self.num)
        print(alen)
        print(aarr[self.num], alen)
        pre_state = [aarr[self.num][1], aarr[self.num][2]]
        # next_state = [aarr[len(actiont_re)-1][4], aarr[len(actiont_re)-1][5]]
        next_state = [aarr[self.num][4], aarr[self.num][5]]
        self.num += 1
        reward = self.reward(next_state[0], next_state[1])
        #info = self._task_name  
        info =  {"task_name":self._task_name}
        done = False
        done_bool = float(done)
        print('**********************step end***************')

        # 将state进行归一化
        next_state[0] = torch.sigmoid(torch.from_numpy(np.array((next_state[0] - 160) / 18))).numpy()
        next_state[1] = torch.sigmoid(torch.from_numpy(np.array((next_state[1] - 160) / 18))).numpy()

        return np.array(next_state), reward, done_bool, info




    def reset_task(self, idx):
        """
        设置当前env为第index个task的env，不需要得到当前task的状态
        :param idx:
        :return:
        """
        ### 将env的各种信息设置为当前患者
        self._task = self.tasks[idx]
        self._task_name = self._task_name.split('#')[0]  # 设置当前任务的名称，例如：“Type1 adult#001”
        if self._task < 9:
            self._task_name = self._task_name + "#00" + str(self._task + 1)
        else:
            self._task_name = self._task_name + "#0" + str(self._task + 1)

        # 更新当前任务患者个人体征
        self.personal_feature = ptu.FloatTensor(self.personal_feature_dict[self._task_name])  # 默认为“Type1 adult#001”的
        # 更新第一天的血糖曲线
        self.demonstrations = self.get_demonstrations()

    def get_task_name(self, index_temp):
        """
        根据任务idx返回任务的名字
        :param index_temp:
        :return:
        """
        if index_temp < 9:
            return(self.task_type + "#00" + str(index_temp+1))
        else:
            return (self.task_type + "#0" + str(index_temp+1))

    def get_batch_p(self, indices):
        """
        根据任务list——indices，返回对应的p的list
        :param indices:
        :return:
        """
        p_list = []
        for idx in indices:
            name_temp = self.get_task_name(idx)
            p_list.append(self.personal_feature_dict[name_temp].cuda())
        return p_list