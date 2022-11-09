import os, sys
lib_path = os.path.abspath(os.path.join('../../..'))
result_path =  os.path.join(lib_path, 'results')
import abc
from collections import OrderedDict
import time

import gtimer as gt
import numpy as np
import pymysql

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.torch import pytorch_util as ptu


# from dmmsDjango.RLType1.rlkit.samplers.cmdcontrol import Patient

class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            train_tasks,
            eval_tasks,
            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=True,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.env = env
        self.agent = agent
        self.exploration_agent = agent  # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment

        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter

        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        # 为 RL、encoder 的训练分别设置缓冲区
        print("self.train_tasks:", self.train_tasks)

        # 存储 label data
        self.replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            np.append(self.train_tasks, self.eval_tasks),
        )

        self.enc_replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            np.append(self.train_tasks, self.eval_tasks),
        )

        self.replay_buffer_TU = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            np.append(self.train_tasks, self.eval_tasks),
        )

        self.replay_buffer_SU = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            np.append(self.train_tasks, self.eval_tasks),
        )

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

    def make_exploration_policy(self, policy):
        return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def train(self):
        '''
        元训练过程
        meta-training loop
        '''
        # patient = Patient()  # 实例化，记录哪个病人 2021-04-01 无用
        self.pretrain()  # 预训练过程，现在是pass
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)  # 记录模型参数
        gt.reset()  # 设置计时器
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        # 每次迭代，首先从任务中收集数据，进行 meta-updates， 然后进行评估
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)
            if it_ == 0:
                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:  # 依次收集每个任务的 initial pool
                    print('使用病人：', idx + 1)
                    # 通过数据库告知cmdcontrol哪个病人
                    db = pymysql.connect(host="localhost", user="root", password="123456",
                                         database="dmms")  # 打开数据库，配置数据库
                    cursor = db.cursor()  # 数据库操作
                    sql = "insert into patient(id)values('%d')" % (
                            idx + 1)  # 存入数据库
                    cursor.execute(sql)  # 执行数据库语句
                    db.commit()  # 提交
                    with open(os.path.join(result_path, 'dataDeal2.txt'), 'a+') as dfile:
                        dfile.write('\npatient:' + str(idx + 1) + '\n')
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.collect_data(self.num_initial_steps, 1, np.inf)
            # Sample data from train tasks.
            for i in range(self.num_tasks_sample):  # 重复5次
                idx = np.random.randint(len(self.train_tasks))  # 随机采样一个 task 的 idx
                self.task_idx = self.train_tasks[idx]
                self.env.reset_task(self.task_idx)
                # 告知cmdcontrol
                # patient.person = idx  # 无用
                # print("idx:", idx, "\nenc_replay_buffer:",self.enc_replay_buffer,"\nself.enc_replay_buffer.task_buffers:",self.enc_replay_buffer.task_buffers)

                self.enc_replay_buffer.task_buffers[self.task_idx].clear()

                print('使用病人：', self.task_idx + 1)
                # 通过数据库告知cmdcontrol哪个病人
                db = pymysql.connect(host="localhost", user="root", password="123456", database="dmms")  # 打开数据库，配置数据库
                cursor = db.cursor()  # 数据库操作
                sql = "insert into patient(id)values('%d')" % (
                        self.task_idx + 1)  # 存入数据库
                cursor.execute(sql)  # 执行数据库语句
                db.commit()  # 提交
                with open(os.path.join(result_path, 'dataDeal2.txt'), 'a+') as dfile:
                    dfile.write('\npatient:' + str(self.task_idx + 1) + '\n')

                # collect some trajectories with z ~ prior
                # 由于一开始 latent 为空，所以一开始是从q(z)中采集，然后将q(z)更新变为q(z|c)
                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf)  # 采集100个数据，不更新 q(z|c)
                # collect some trajectories with z ~ posterior
                # 从q(z|c)中采集，更新encoder q(z|c)
                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train)  # 采集100个数据，更新1次q(z|c)
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                # 从更新后的q(z|c)中采集新的数据
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train,
                                      add_to_enc_buffer=False)

            # Sample train tasks and compute gradient updates on parameters.
            # 更新参数，随机选择task进行训练
            print("**************************** 开始训练 *******************************")
            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.train_tasks, self.meta_batch)  # self.meta_batch = 16
                self._do_training(indices)
                self._n_train_steps_total += 1
            gt.stamp('train')

            self.training_mode(False)

            # eval
            self._try_to_eval(it_)
            gt.stamp('eval')

            self._end_epoch()

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:  # 与环境进行交互产生一条轨迹path，
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=False,
                                                           resample=resample_z_rate)
            # path 为 dict 数据格式；n_samples 是在该轮rollout中一共有多少step，也就是有多少transitions
            self._n_rollouts_total += len(paths)  # 统计共有多少次 rollout
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)  # 添加到 RL replay buffer 中
            if add_to_enc_buffer:  # 添加到 encoder replay buffer 中
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:  # 更新q(z|c)
                context = self.sample_context(self.task_idx)
                self.agent.infer_posterior(context)  # 根据context计算出一个新的z
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def _try_to_eval(self, epoch):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))  # 记录在snapshot中没有保存的内容，可自选内容
        if self._can_evaluate():  # True
            self.evaluate(epoch)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            with open(os.path.join(result_path, 'Epochrecord.txt'), 'a+') as dfile:
                dfile.write(
                    'Number of train steps total:\t' + str(self._n_train_steps_total) + '\n'
                    + 'Number of env steps total:\t' + str(self._n_env_steps_total) + '\n'
                    + 'Number of rollouts total:\t' + str(self._n_rollouts_total) + '\n'
                    + 'Train Time (s):\t' + str(train_time) + '\n'
                    + '(Previous) Eval Time (s):\t' + str(eval_time) + '\n'
                    + 'Sample Time (s):\t' + str(sample_time) + '\n'
                    + 'Epoch Time (s):\t' + str(epoch_time) + '\n'
                    + 'Total Train Time (s):\t' + str(total_time) + '\n'
                    + 'Epoch:\t' + str(epoch) + '\n'
                )
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")
            with open(os.path.join(result_path, 'Epochrecord.txt'), 'a+') as dfile:
                dfile.write('Skipping eval for now.\n')

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation, )

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        with open(os.path.join(result_path, 'Epochrecord.txt'), 'a+') as dfile:
            dfile.write(
                'Epoch Duration:\t' + str(time.time() - self._epoch_start_time) + '\n'
                + 'Started Training:\t' + str(self._can_train()) + '\n'
            )
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%% 存储 replay buffer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                    max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1,
                                                    accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self.num_exp_traj_eval:
                self.agent.infer_posterior(self.agent.context)

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        # goal = self.env._goal
        # for path in paths:
        #     path['goal'] = goal # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def collect_paths_meta(self, idx, epoch, run):
        """
        用于meta testing任务中
        :param idx:
        :param epoch:
        :param run:
        :return:
        """
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                    max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1,
                                                    accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self.num_exp_traj_eval:
                self.agent.infer_posterior(self.agent.context)
            self.replay_buffer.add_paths(self.task_idx, paths)  # 添加到 RL replay buffer 中
            self.enc_replay_buffer.add_paths(self.task_idx, paths)  # 添加到 encoder replay buffer 中

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def collect_paths_mmd(self, idx, epoch, run):
        """
        用于meta testing任务中
        :param idx:
        :param epoch:
        :param run:
        :return:
        """
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            path, num = self.sampler.obtain_samples_mmd(deterministic=self.eval_deterministic,
                                                    max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1,
                                                    accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self.num_exp_traj_eval:
                self.agent.infer_posterior(self.agent.context)
            self.replay_buffer.add_paths(self.task_idx, paths)  # 添加到 RL replay buffer 中
            self.enc_replay_buffer.add_paths(self.task_idx, paths)  # 添加到 encoder replay buffer 中

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def _do_eval_meta(self, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:
            print('使用病人：', idx + 1)
            # 通过数据库告知cmdcontrol哪个病人
            db = pymysql.connect(host="localhost", user="root", password="123456", database="dmms")  # 打开数据库，配置数据库
            cursor = db.cursor()  # 数据库操作
            sql = "insert into patient(id)values('%d')" % (
                    idx + 1)  # 存入数据库
            cursor.execute(sql)  # 执行数据库语句
            db.commit()  # 提交
            with open(os.path.join(result_path, 'dataDeal2.txt'), 'a+') as dfile:
                dfile.write('\npatient:' + str(idx + 1) + '\n')
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths_mmd(idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0)  # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def _do_eval(self, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:
            print('使用病人：', idx + 1)
            # 通过数据库告知cmdcontrol哪个病人
            db = pymysql.connect(host="localhost", user="root", password="123456", database="dmms")  # 打开数据库，配置数据库
            cursor = db.cursor()  # 数据库操作
            sql = "insert into patient(id)values('%d')" % (
                    idx + 1)  # 存入数据库
            cursor.execute(sql)  # 执行数据库语句
            db.commit()  # 提交
            with open(os.path.join(result_path, 'dataDeal2.txt'), 'a+') as dfile:
                dfile.write('\npatient:' + str(idx + 1) + '\n')
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])  # 存放所有path的 sum reward
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])  # 4
            all_rets = [a[:n] for a in all_rets]    # 裁剪一下，防止每个a的维度不同。有的任务有done，所以可能提前终止
            all_rets = np.mean(np.stack(all_rets), axis=0)  # avg return per nth rollout    计算每一列的均值
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def evaluate(self, epoch):

        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        ### 保存eval轨迹可用于可视化、debugging
        if self.dump_eval_paths:  # 是否保存eval过程中的轨迹，False
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            self.agent.clear_z()
            prior_paths, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                         max_samples=self.max_path_length * 20,
                                                         accum_context=False,
                                                         resample=1)
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer  q(z|c)
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            print('使用病人：', idx + 1)
            # 通过数据库告知cmdcontrol哪个病人
            db = pymysql.connect(host="localhost", user="root", password="123456", database="dmms")  # 打开数据库，配置数据库
            cursor = db.cursor()  # 数据库操作
            sql = "insert into patient(id)values('%d')" % (
                    idx + 1)  # 存入数据库
            cursor.execute(sql)  # 执行数据库语句
            db.commit()  # 提交
            with open(os.path.join(result_path, 'dataDeal2.txt'), 'a+') as dfile:
                dfile.write('\n**************测试***************' + '\n' + 'patient:' + str(idx + 1) + '\n')
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context = self.sample_context(idx)
                self.agent.infer_posterior(context)
                p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                   max_samples=self.max_path_length,
                                                   accum_context=False,
                                                   max_trajs=1,
                                                   resample=np.inf)
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            # eval_util.get_average_returns(paths) 1、计算每个path的sum reward；2、将多个path的sum reward求平均
            # train_returns 中记录的是 indices 中 各个任务的sum reward 的平均值。
            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)  # 各个任务的 sum reward 求均值，train_returns为一个实数
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns = self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))
        with open(os.path.join(result_path, 'Epochrecord.txt'), 'a+') as dfile:
            dfile.write('------------------------------------------------------------------\n')
        for key, value in self.eval_statistics.items():
            with open(os.path.join(result_path, 'Epochrecord.txt'), 'a+') as dfile:
                dfile.write(key + ':\t' + str(value) + '\n')
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    def evaluate_meta_test(self, indices, num_epoch):
        """
        meta test 阶段的测试。与indices中的患者分别交互 iter 次，每次60step，生成患者的血糖曲线
        :param indices:
        :return:
        """
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        for epoch in range(num_epoch):
            ### test tasks
            eval_util.dprint('evaluating on {} test tasks'.format(len(indices)))
            test_final_returns, test_online_returns = self._do_eval_meta(indices, epoch)
            eval_util.dprint('test online returns')
            eval_util.dprint(test_online_returns)

            # save the final posterior
            self.agent.log_diagnostics(self.eval_statistics)

            avg_test_return = np.mean(test_final_returns)
            avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
            self.eval_statistics['Eval_patient_num'] = indices[0]
            self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
            logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

            with open(os.path.join(result_path, 'Epochrecord.txt'), 'a+') as dfile:
                dfile.write('------------------------------------------------------------------\n')
            for key, value in self.eval_statistics.items():
                with open(os.path.join(result_path, 'Epochrecord.txt'), 'a+') as dfile:
                    dfile.write(key + ':\t' + str(value) + '\n')
                logger.record_tabular(key, value)
        self.eval_statistics = None

        logger.save_extra_data(self.get_extra_data_to_save(num_epoch), path='reply_buffer')

        if self.plotter:
            self.plotter.draw()

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass
