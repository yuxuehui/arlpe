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
    print("o:", o)
    next_o = None
    path_length = 0


    while path_length < max_path_length:
        #num_flag = False
        p = env.personal_feature.cpu()
        p = p.numpy()[0]
        o = np.array(o)
        print("o", o)
        # o = ptu.from_numpy(o[None])
        a, agent_info = agent.get_action(o)
        print("a:", a, "\nagent_info:", agent_info)
        # 方案1
        #if a < 0:
            #a = -a
            #num_flag = True
        # 方案2
        #a = a.clip(0,60)
        #a = (a+1)/2
        print('a_clip', a)
        # 存起来，告诉URL端
        print('env.anum', env.anum)
        db = pymysql.connect(host="localhost", user="root", password="123456", database="dmms")  # 打开数据库，配置数据库
        cursor = db.cursor()  # 数据库操作
        sql = "insert into action(id,dose)values('%d','%lf')" % (
            env.anum, a)  # 存入数据库
        cursor.execute(sql)  # 执行数据库语句
        db.commit()  # 提交
        env.anum += 1
        next_o, r, d, env_info = env.step(a)
        # 方案1
        #if num_flag:
            #r = -r
        print('step', next_o, r, d, env_info)
        # update the agent's current context
        # 更新 agent 当前的上下文，如果 accum_context=True 则使用上下文信息
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info], p)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o

        # if save_frames:     # 记录rollout的video
        #     from PIL import Image
        #     image = Image.fromarray(np.flipud(env.get_image()))
        #     env_info['frame'] = image

        env_infos.append(env_info)
        if d:  # done
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:  # return的需要是一个2D的array
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
    print("o:", o)
    next_o = None
    path_length = 0


    while path_length < max_path_length:
        #num_flag = False
        p = env.personal_feature.cpu()
        p = p.numpy()[0]
        o = np.array(o)
        print("o, env.personal_feature:", o)
        # o = ptu.from_numpy(o[None])
        a, agent_info = agent.get_action(o)
        print("a:", a, "\nagent_info:", agent_info)
        # 方案1
        #if a < 0:
            #a = -a
            #num_flag = True
        # 方案2
        #a = a.clip(0,60)
        #a = (a+1)/2
        print('a_clip', a)
        # 存起来，告诉URL端
        print('env.anum', env.anum)
        db = pymysql.connect(host="localhost", user="root", password="123456", database="dmms")  # 打开数据库，配置数据库
        cursor = db.cursor()  # 数据库操作
        sql = "insert into action(id,dose)values('%d','%lf')" % (
            env.anum, a)  # 存入数据库
        cursor.execute(sql)  # 执行数据库语句
        db.commit()  # 提交
        env.anum += 1
        next_o, r, d, env_info = env.step(a)
        # 方案1
        #if num_flag:
            #r = -r
        print('step', next_o, r, d, env_info)
        # update the agent's current context
        # 更新 agent 当前的上下文，如果 accum_context=True 则使用上下文信息
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

        # if save_frames:     # 记录rollout的video
        #     from PIL import Image
        #     image = Image.fromarray(np.flipud(env.get_image()))
        #     env_info['frame'] = image

        env_infos.append(env_info)
        if d:  # done
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:  # return的需要是一个2D的array
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
