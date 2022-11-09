import time

import numpy as np
import pymysql

from rlkit.samplers.util import rollout, rollout_mmd
from rlkit.torch.sac.policies import MakeDeterministic


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length):
        self.env = env
        self.policy = policy

        self.max_path_length = max_path_length

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        accum_context：更新 agent 当前的上下文，如果 accum_context=True 则使用上下文信息
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        # 转换为确定型策略
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0

        while n_steps_total < max_samples and n_trajs < max_trajs:
            # 运行前清空db = pymysql.connect(host="localhost", user="root", password="xxx", database="xxxx")
            db = pymysql.connect(host="localhost", user="root", password="123456", database="dmms")  # 打开数据库，配置数据库
            cursor = db.cursor()  # 数据库操作
            print("successful")

            # 查询上次回合到哪里了
            sql = "select * from episode"
            cursor.execute(sql)  # 查询
            db.commit()
            episode_arr = cursor.fetchall()
            episode_alen = episode_arr.__len__()
            print(episode_arr[episode_alen - 1][0])

            # 存入新回合数以开启新回合
            sql = "insert into episode(id)values('%d')" % (
                episode_arr[episode_alen - 1][0]+1)  # 存入数据库
            cursor.execute(sql)  # 执行数据库语句
            db.commit()  # 提交

            # 每次操作前删除原有数据
            sql = "delete from state"
            cursor.execute(sql)  # 执行数据库语句
            db.commit()  # 提交
            sql = "delete from action"
            cursor.execute(sql)  # 执行数据库语句
            db.commit()  # 提交
            sql = "delete from sasrecord"
            cursor.execute(sql)  # 执行数据库语句
            db.commit()  # 提交

            self.env.num = 0
            self.env.anum = 0

            path = rollout(
                self.env,
                policy,
                max_path_length=self.max_path_length,
                accum_context=accum_context
            )
            # save the latent context that generated this trajectory
            # 保存生成当前轨迹的 latent context —— policy.z
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            # don't we also want the option to resample z ever transition?
            # 重新采样z
            if n_trajs % resample == 0:
                policy.sample_z()
            time.sleep(2)  # 等待重新开始，不然数据库清楚后台还未响应最后一次
        return paths, n_steps_total


    def obtain_samples_mmd(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        accum_context：更新 agent 当前的上下文，如果 accum_context=True 则使用上下文信息
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        # 转换为确定型策略
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0

        while n_steps_total < max_samples and n_trajs < max_trajs:
            # 运行前清空db = pymysql.connect(host="localhost", user="root", password="xxx", database="xxxx")
            db = pymysql.connect(host="localhost", user="root", password="123456", database="dmms")  # 打开数据库，配置数据库
            cursor = db.cursor()  # 数据库操作
            print("successful")

            # 查询上次回合到哪里了
            sql = "select * from episode"
            cursor.execute(sql)  # 查询
            db.commit()
            episode_arr = cursor.fetchall()
            episode_alen = episode_arr.__len__()
            print(episode_arr[episode_alen - 1][0])

            # 存入新回合数以开启新回合
            sql = "insert into episode(id)values('%d')" % (
                episode_arr[episode_alen - 1][0]+1)  # 存入数据库
            cursor.execute(sql)  # 执行数据库语句
            db.commit()  # 提交

            # 每次操作前删除原有数据
            sql = "delete from state"
            cursor.execute(sql)  # 执行数据库语句
            db.commit()  # 提交
            sql = "delete from action"
            cursor.execute(sql)  # 执行数据库语句
            db.commit()  # 提交
            sql = "delete from sasrecord"
            cursor.execute(sql)  # 执行数据库语句
            db.commit()  # 提交

            self.env.num = 0
            self.env.anum = 0

            path = rollout_mmd(
                self.env,
                policy,
                max_path_length=self.max_path_length,
                accum_context=accum_context
            )
            # save the latent context that generated this trajectory
            # 保存生成当前轨迹的 latent context —— policy.z
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            # don't we also want the option to resample z ever transition?
            # 重新采样z
            if n_trajs % resample == 0:
                policy.sample_z()
            time.sleep(2)  # 等待重新开始，不然数据库清楚后台还未响应最后一次
        return paths, n_steps_total

