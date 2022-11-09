"""
    meta training
"""

import os
import pathlib
import numpy as np
import click
import json

import pymysql
import torch

from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
from rlkit.envs.diabetes_env import DiabetesEnvs
from configs.mysql_config import mysql_config


def experiment(variant):

    env = DiabetesEnvs(10)
    tasks = env.get_all_task_idx()  # np.array ; [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    obs_dim = int(np.prod(env.observation_space.shape))  # state的维度，2
    action_dim = int(np.prod(env.action_space.shape))  # 1
    reward_dim = 1
    person_dim = int(np.prod(env.person_space.shape))

    ### instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params'][
        'use_next_obs_in_context'] else obs_dim + action_dim + reward_dim  # (o,a,o',r)
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim

    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']  # False, 不使用 RNN，使用 MLP
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[300, 300, 300],
        input_size=context_encoder_input_dim + person_dim,
        output_size=context_encoder_output_dim,
    )

    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,  # s,a,z
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,  # s,a,z
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,  # s,z
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=[0,1,2,3,4,5,6,7,8,9],  # 使用*号患者作为测试任务
        eval_tasks=[1],
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id,
                                      base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()


if __name__ == "__main__":
    db = pymysql.connect(host=mysql_config.mysql_host, user=mysql_config.mysql_user, password=mysql_config.mysql_password, database=mysql_config.mysql_database)
    cursor = db.cursor()
    sql = "delete from patient"
    cursor.execute(sql)
    db.commit()
    sql = "delete from episode"
    cursor.execute(sql)
    db.commit()
    sql = "insert into episode(id)values('%d')" % (-1)  # 存入数据库
    cursor.execute(sql)
    db.commit()

    variant = default_config
    experiment(variant)
