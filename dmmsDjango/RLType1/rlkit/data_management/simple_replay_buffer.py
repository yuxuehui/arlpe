import numpy as np
import random
from rlkit.data_management.replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._weights = np.ones((max_replay_buffer_size, 1))
        self._sparse_rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self.clear()

    def add_sample(self, observation, action, reward, weight, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._weights[self._top] = weight
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._sparse_rewards[self._top] = kwargs['env_info'].get('sparse_reward', 0)
        self._advance()

    def terminate_episode(self):
        # store the episode beginning once the episode is over
        # n.b. allows last episode to loop but whatever
        self._episode_starts.append(self._cur_episode_start)
        self._cur_episode_start = self._top

    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def sample_data(self, indices):
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            weights=self._weights[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            sparse_rewards=self._sparse_rewards[indices],
        )
    def delete_buffer(self, indices):
        """
        根据 list 的 ind 索引 删除replay buffer 对应位置的元素
        :param ind:
        :return:
        """
        self._observations = np.delete(self._observations, indices, axis=0)
        self._actions = np.delete(self._actions, indices, axis=0)
        self._rewards = np.delete(self._rewards, indices, axis=0)
        self._weights = np.delete(self._weights, indices, axis=0)
        self._terminals = np.delete(self._terminals, indices, axis=0)
        self._next_obs = np.delete(self._next_obs, indices, axis=0)
        self._sparse_rewards = np.delete(self._sparse_rewards, indices, axis=0)
        self._top = self._top - len(indices)
        self._size = self._size - len(indices)

    def change_weights(self, beta):
        """
        将 self._weights 替换成 beta
        :param beta:
        :return:
        """
        self._weights[0:beta.shape[0]] = beta


    def random_batch(self, batch_size):
        ''' batch of unordered transitions '''
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)


    def random_batch_del(self, batch_size):
        """
        给 sample_sac_and_del 调用的，随机采样SU中一批数据，并且在SU中删除
        :param batch_size:
        :return:
        """
        ''' batch of unordered transitions '''
        indices = list(range(self._size))
        indices = random.sample(indices, batch_size) # 生成 batch_size 大小的随机列表，且元素不重复
        return_data = self.sample_data(indices)
        # indices = list(set(indices))    # indices 中会包含很多重复元素，此步进行去重
        indices.sort(reverse=True)  # 降序排列
        self.delete_buffer(indices)
        return return_data

    def random_sequence(self, batch_size):
        ''' batch of trajectories '''
        # take random trajectories until we have enough
        i = 0
        indices = []
        while len(indices) < batch_size:
            # TODO hack to not deal with wrapping episodes, just don't take the last one
            start = np.random.choice(self.episode_starts[:-1])
            pos_idx = self._episode_starts.index(start)
            indices += list(range(start, self._episode_starts[pos_idx + 1]))
            i += 1
        # cut off the last traj if needed to respect batch size
        indices = indices[:batch_size]
        return self.sample_data(indices)

    def num_steps_can_sample(self):
        return self._size
