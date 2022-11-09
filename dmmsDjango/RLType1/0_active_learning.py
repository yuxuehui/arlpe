import pickle
import numpy as np
from numpy import matrix
import sklearn.metrics
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
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

generalized_model_dir = ""
SL_data_dir = ""    # containing reply_buffer.pkl
SU_data_dir = ""
TU_data_dir = ""    # containing reply_buffer.pkl



# %% Kernel
def kernel(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K


# %% Kernel Mean Matching (KMM)
class KMM:
    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        '''
        Initialization function
        :param kernel_type: 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel
        :param B: bound for beta
        :param eps: bound for sigma_beta
        '''
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        '''
        Fit source and target using KMM (compute the coefficients)
        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt / Ps) value vector (Beta in the paper)
        '''
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = kernel(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(kernel(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1)

        K = matrix(K)
        kappa = matrix(kappa)
        G = matrix(
            np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])  # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])
        return beta


def TASL(SL, SU, TL, TU, gSU, nQ, lamb, num_iter=2, gamma=1.0):
    """
    :param SL: 源域 有标签数据
    :param SU: 源域 无标签数据
    :param TL: 目标域 有标签数据
    :param TU: 目标域 无标签数据
    :param gSU: 模型在 源域未标签数据 上的预测结果
    :param nQ: batch size of active query
    :param lamb: tradeoff parameter
    :return:
    """
    nSL = SL.shape[0]
    nSU = SU.shape[0]
    nTL = TL.shape[0]
    nTU = TU.shape[0]
    nL = nSL + nQ + nTL
    nU = nSU - nQ + nTU
    nS = nSU + nSL
    nT = nTU + nTL

    # 1 Calculate the kernel matrix K and initialize beta;
    #   beta 初始化为全1向量
    beta = np.ones((nSL + nSU, 1))
    #   Calculate the kernel matrix K
    K_SU = kernel('rbf', SU, None, gamma)
    K_SU_SL = kernel('rbf', SU, SL, gamma)
    K_SU_TL = kernel('rbf', SU, TL, gamma)
    K_SU_TU = kernel('rbf', SU, TU, gamma)
    K_SL = kernel('rbf', SL, None, gamma)
    K_SL_SU = kernel('rbf', SL, SU, gamma)
    K_SL_TL = kernel('rbf', SL, TL, gamma)
    K_SL_TU = kernel('rbf', SL, TU, gamma)
    K_S = kernel('rbf', np.r_[SU, SL], None, gamma)

    # 2 For each active querying iteration:
    for step in range(num_iter):
        ## 2.1 Update alpha by solving Eq. (5);
        A = matrix(np.dot(beta[1:nSU + 1, :], np.transpose(beta[1:nSU + 1, :])) * K_SU * (
                    (float(1. / nL) + float(1. / nU)) ** 2))
        a = - (float(1. / (nU ** 2)) + float(2. / (nL * nU))) * np.dot(
            np.dot(beta[1:nSU + 1, :], np.transpose(beta[1:nSU + 1, :])) * K_SU, np.ones((nSU, 1))) \
            + (float(1. / (nL ** 2)) + float(2. / (nL * nU))) * np.dot(
            np.dot(beta[1:nSU + 1, :], np.transpose(beta[nSU:nSU + nSL + 1, :])) * K_SU_SL, np.ones((nSL, 1))) \
            + (float(1. / (nL ** 2)) + float(2. / (nL * nU))) * np.dot(
            np.dot(beta[1:nSU + 1, :], np.transpose(np.ones((nTL, 1)))) * K_SU_TL, np.ones((nTL, 1))) \
            - (float(1. / (nU ** 2)) + float(2. / (nL * nU))) * np.dot(
            np.dot(beta[1:nSU + 1, :], np.transpose(np.ones((nTU, 1)))) * K_SU_TU, np.ones((nTU, 1))) \
            + float(lamb / 2.) * beta[1:nSU + 1, :] * abs(gSU)
        a = matrix(a)
        G_a = matrix(np.r_[np.eye(nSU), -np.eye(nSU)])
        h_a = matrix(np.r_[np.ones((nSU,)), np.zeros((nSU,))])
        A_a = matrix(np.ones((1, nSU)))
        b_a = matrix(nQ)
        sol_alpha = solvers.qp(A, a, G_a, h_a, A_a, b_a)
        alpha = np.array(sol_alpha['x'])

        ## 2.2 Update beta by solving Eq. (6);
        B11 = matrix(float(1. / (nL ** 2)) * K_SL)
        B12 = matrix(((float(1. / (nL ** 2)) + float(1. / (nL * nU))) * np.dot(np.ones((nSU, 1)),
                                                                               np.transpose(alpha)) - float(
            1. / (nL * nU)) * np.ones((nSL, nSU))) * K_SL_SU)
        B21 = matrix(((float(1. / (nL ** 2)) + float(1. / (nL * nU))) * np.dot(alpha,
                                                                               np.transpose(np.ones((nSL, 1)))) - float(
            1. / (nL * nU)) * np.ones((nSU, nSL))) * K_SU_SL)
        B22 = matrix(
            (
                    ((float(1. / nL) + float(1. / nU)) ** 2) * np.dot(alpha, np.transpose(alpha))
                    - (float(1. / (nU ** 2)) + float(1. / (nL * nU)))
                    * (
                            np.dot(np.ones((nSU, 1)), np.transpose(alpha))
                            + np.dot(alpha, np.transpose(np.ones((nSU, 1))))
                            + float(1. / (nU ** 2)) * np.ones((nSU, nSU))
                    )
            )
            * K_SU
        )
        B = matrix(lamb * float(1. / (nS ** 2)) * K_S + np.r_[
            np.c_[B11, B12], np.c_[B21, B22]])
        b1 = matrix(float(1. / (nL ** 2)) * np.dot(K_SL_TL, np.ones(nTL, 1)) - float(1. / (nL * nU)) * np.dot(K_SL_TU,np.ones(nTU, 1)))
        b2 = matrix(
            np.dot(K_SU_TL, np.ones(nTL, 1)) * (
                        (float(1. / (nL ** 2)) + float(1. / (nL * nU))) * alpha - float(1. / (nL * nU)) * np.ones(nSU,1))
            - np.dot(K_SU_TU, np.ones(nTU, 1)) * (
                        (float(1. / (nL ** 2)) + float(1. / (nL * nU))) * alpha - float(1. / (nU ** 2)) * np.ones(nSU,1))
            - lamb * alpha * abs(gSU)
        )
        b = matrix(np.r_[b1, b2])
        G_b = matrix(np.r_[np.eye(nS), -np.eye(nS)])
        h_b = matrix(np.r_[np.ones((nS,)), np.zeros((nS,))])
        sol_beta = solvers.qp(B, b, G_b, h_b)
        beta = np.array(sol_beta['x'])

    # 3 Q <-- top nQ instances of SU with largest alpha values;
    Q_index = np.array([])
    for step in range(nQ):
        Q_index.append(alpha.index(max(alpha)))

    # 4 SU = SU \ Q; SL = SL and Q;
    # 5 Train the model based on TL and adapted SL with beta.
    return alpha, beta, Q_index


if __name__ == "__main__":
    ######################################################################################################################
    ##                                          1 clear database
    ######################################################################################################################
    db = pymysql.connect(host=mysql_config.mysql_host, user=mysql_config.mysql_user, password=mysql_config.mysql_password, database=mysql_config.mysql_database)
    cursor = db.cursor()
    sql = "delete from patient"
    cursor.execute(sql)
    db.commit()
    sql = "delete from episode"
    cursor.execute(sql)
    db.commit()
    sql = "insert into episode(id)values('%d')" % (-1)
    cursor.execute(sql)
    db.commit()

    ######################################################################################################################
    ##                                          2 loading meta training model
    ######################################################################################################################
    training_model_dir = generalized_model_dir

    variant = default_config
    env = DiabetesEnvs(10)
    tasks = env.get_all_task_idx()  # np.array
    obs_dim = int(np.prod(env.observation_space.shape))  # 2
    action_dim = int(np.prod(env.action_space.shape))  # 1
    reward_dim = 1
    person_dim = int(np.prod(env.person_space.shape))

    ### instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params'][
        'use_next_obs_in_context'] else obs_dim + action_dim + reward_dim  # (o,a,o',r)
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    context_encoder = encoder_model(
        hidden_sizes=[300, 300, 300],
        input_size=context_encoder_input_dim + person_dim,
        output_size=context_encoder_output_dim,
    )

    print("obs_dim", obs_dim)
    print("action_dim:", action_dim)
    print("latent_dim:", latent_dim)
    print("obs_dim + action_dim + latent_dim:", obs_dim + action_dim + latent_dim)
    print("obs_dim + latent_dim:", obs_dim + latent_dim)
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
        train_tasks=[0, 2, 3, 4, 5, 6, 7, 8, 9],
        eval_tasks=[1],
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    # optionally load pre-trained weights
    if training_model_dir is not None:
        path = training_model_dir
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

    ######################################################################################################################
    ###                                         3、active learning
    ######################################################################################################################
    print("************************** generate SL data *****************************")
    data = pickle.load(open(os.path.join(SL_data_dir, 'reply_buffer.pkl'), 'rb'))
    algorithm.replay_buffer = data["replay_buffer"]
    algorithm.enc_replay_buffer = data["replay_buffer"]

    print("************************** generate SU data *****************************")
    su_initial_size = 500
    for key in algorithm.replay_buffer.task_buffers.keys():
        print("key:", key)
        print("replay buffer:", algorithm.replay_buffer.task_buffers[key]._size)
        batch_data = algorithm.replay_buffer.task_buffers[key].random_batch_del(su_initial_size)
        print("replay buffer 2:", algorithm.replay_buffer.task_buffers[key]._size)
        algorithm.replay_buffer_SU.add_path(key, batch_data)
        print("replay buffer SU:", algorithm.replay_buffer_SU.task_buffers[key]._size)

    print("************************** generate TU data *****************************")
    TU_data = pickle.load(open(os.path.join(TU_data_dir, 'reply_buffer.pkl'), 'rb'))
    algorithm.replay_buffer_TU = TU_data["replay_buffer"]
    print("replay buffer TU:", algorithm.replay_buffer_TU.task_buffers[3]._size)

    algorithm.mmd_meta_testing(algorithm.train_tasks[0], algorithm.train_tasks, 2, 5, nQ=20)

