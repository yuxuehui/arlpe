import os, sys
mysql_path = os.path.abspath(os.path.join('../..'))
sys.path.append(mysql_path)
from configs.mysql_config import mysql_config

import numpy as np
import pymysql
import rlkit.torch.pytorch_util as ptu

def rollout(env, agent, max_path_length=np.inf, accum_context=True, animated=False, save_frames=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :param save_frames: if True, save video of rollout1
    :return: dict
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0


    while path_length < max_path_length:
        p = env.personal_feature.cpu()
        p = p.numpy()[0]
        o = np.array(o)
        a, agent_info = agent.get_action(o)
        db = pymysql.connect(host=mysql_config.mysql_host, user=mysql_config.mysql_user,
                             password=mysql_config.mysql_password, database=mysql_config.mysql_database)
        cursor = db.cursor()
        sql = "insert into action(id,dose)values('%d','%lf')" % (
            env.anum, a)
        cursor.execute(sql)
        db.commit()
        env.anum += 1
        next_o, r, d, env_info = env.step(a)
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info], p)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o
        env_infos.append(env_info)
        if d:  # done
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )  # 将两个数组按垂直方向叠加
    weights = np.ones(np.array(rewards).shape)
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),  # 将数据变成1列
        weights=np.array(weights).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def rollout_mmd(env, agent, max_path_length=np.inf, accum_context=True, animated=False, save_frames=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated: 是否可视化环境，已删除该参数
    :param save_frames: if True, save video of rollout1
    :return: 字典格式 dict
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()     # 初始化环境，返回的为 环境的初始化状态。
    next_o = None
    path_length = 0


    while path_length < max_path_length:
        p = env.personal_feature.cpu()
        p = p.numpy()[0]
        o = np.array(o)
        a, agent_info = agent.get_action(o)
        db = pymysql.connect(host=mysql_config.mysql_host, user=mysql_config.mysql_user,
                             password=mysql_config.mysql_password, database=mysql_config.mysql_database)
        cursor = db.cursor()
        sql = "insert into action(id,dose)values('%d','%lf')" % (
            env.anum, a)
        cursor.execute(sql)
        db.commit()
        env.anum += 1
        next_o, r, d, env_info = env.step(a)

        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info], p)
        agent.infer_posterior(agent.context)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o
        env_infos.append(env_info)
        if d:  # done
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    weights = np.ones(np.array(rewards).shape)
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),  # 将数据变成1列
        weights=np.array(weights).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )

def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
