import numpy as np
import gym

class GymEnvs:
    def __init__(self, task_num, task_name="Hopper-v3"):
        self.env = gym.make(task_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.tasks = range(1, task_num+1)       # 所有task的list，即[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self._task = self.tasks[0]  # 当前task, 默认为 task 1
        # self._goal = self.env._goal

        self._task_name = task_name     # 设置当前任务的名称，例如：“Type1 adult#001”
        # if self._task < 10:
        #     self._task_name = self._task_name + "#00" + str(self._task)
        # else:
        #     self._task_name = self._task_name + "#0" + str(self._task)

        ### 需要定义当前 task 的 meal，大剂量的值，当前患者的体征


    def reset(self):
        return self.env.reset()

    def get_all_task_idx(self):
        """
        :return: 所有任务的index；list；list中的内容为int
        """
        return np.array([0])


    def step(self, action):
        """
        将action传输给模拟器端；
        :param action:
        :return: next_o, r, d, env_info ： 下一时刻状态，reward，是否结束（直接返回0即可），环境信息（None即可）
        """
        return self.env.step(action)

    def reset_task(self, idx):
        """
        设置当前env为第index个task的env
        :param idx:
        :return:
        """
        self._task = self.tasks[idx]
        self.env.reset()