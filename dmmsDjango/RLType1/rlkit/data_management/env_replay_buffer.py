import numpy as np

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from gym.spaces import Box, Discrete, Tuple


class MultiTaskReplayBuffer(object):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            tasks,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        :param tasks: for multi-task setting
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.task_buffers = dict([(idx, SimpleReplayBuffer(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
        )) for idx in tasks])


    def add_sample(self, task, observation, action, reward, weight, terminal,
            next_observation, **kwargs):

        if isinstance(self._action_space, Discrete):
            action = np.eye(self._action_space.n)[action]
        self.task_buffers[task].add_sample(
                observation, action, reward, weight, terminal,
                next_observation, **kwargs)

    def terminate_episode(self, task):
        self.task_buffers[task].terminate_episode()

    def random_batch(self, task, batch_size, sequence=False):
        print('=================================================')
        print('任务'+str(task))
        print(batch_size)
        print(self.task_buffers[task]._size)
        print('=================================================')
        if sequence:
            batch = self.task_buffers[task].random_sequence(batch_size)
        else:
            batch = self.task_buffers[task].random_batch(batch_size)
        return batch

    def num_steps_can_sample(self, task):
        return self.task_buffers[task].num_steps_can_sample()

    def add_path(self, task, path):
        task_name = ["adult#001", "adult#002", "adult#003", "adult#004", "adult#005", "adult#006", "adult#007",
                     "adult#008",
                     "adult#009", "adult#010"]
        # print("path keys:", path.keys())
        if "agent_infos" not in path.keys():
            path_size = path["observations"].shape[0]
            path["agent_infos"] = np.array([{}] * path_size)
            path["env_infos"] = np.array([{'task_name': task_name[task]}] * path_size)
        self.task_buffers[task].add_path(path)

    def add_paths(self, task, paths):
        for path in paths:
            self.task_buffers[task].add_path(path)

    def clear_buffer(self, task):
        self.task_buffers[task].clear()


    def get_all_data(self, tasks):
        """
        根据 tasks 任务index列表，返回对应的 data
        data 为 np.array 格式，二维，第一维度为数据个数，第二维度为数据维度
        :param tasks:
        :return:
        """
        return_data = []
        for ind in tasks:
            data_dict = self.task_buffers[ind].sample_data(list(range(self.task_buffers[ind]._size)))
            temp_data = np.concatenate([data_dict["observations"], data_dict["actions"]], axis=1)   # 进行拼接
            for data in temp_data:
                return_data.append(data)
        return return_data

    def get_obs_data(self, tasks):
        """
        根据 tasks 任务index列表，返回对应的 data
        data 为 np.array 格式，二维，第一维度为数据个数，第二维度为数据维度
        :param tasks:
        :return:
        """
        return_data = []
        for ind in tasks:
            data_dict = self.task_buffers[ind].sample_data(list(range(self.task_buffers[ind]._size)))
            temp_data = data_dict["observations"]
            for data in temp_data:
                return_data.append(data)
        return return_data

def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        # import OldBox here so it is not necessary to have rand_param_envs 
        # installed if not running the rand_param envs
        from gym.spaces.box import Box as OldBox
        if isinstance(space, OldBox):
            return space.low.size
        else:
            raise TypeError("Unknown space: {}".format(space))
