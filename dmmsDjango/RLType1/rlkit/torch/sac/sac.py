import heapq
import math
import random

import os, sys
lib_path = os.path.abspath(os.path.join('../../../..'))
result_path =  os.path.join(lib_path, 'results')

mysql_path = os.path.abspath(os.path.join('../../..'))
sys.path.append(mysql_path)
from configs.mysql_config import mysql_config



import sklearn.metrics
from cvxopt import matrix, solvers

from collections import OrderedDict
import numpy as np
import pymysql
import sklearn
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger, eval_util
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm


class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,  #
            eval_tasks,  #
            latent_dim,  #
            nets,  # [agent, qf1, qf2, vf],

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.qf1, self.qf2, self.vf = nets[1:]
        self.target_vf = self.vf.copy()

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(),
            lr=context_lr,
        )

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        if 'weights' in batch.keys():
            w = batch['weights'][None, ...]
            return [o, a, r, w, no, t]
        else:
            return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in
                   indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]  # 变成[[o, a, r, no, t],...]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]  # 变成[ [[o], [a], [r], [no], [t]] , ... ]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]  # 变成[[o,...], [a,...], [r,...], [no,...], [t,...]]
        return unpacked

    def sample_sac_meta(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        # 1、采集TL数据
        TL_batch_size = math.floor(self.batch_size * 0.5)
        batches_TL = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=TL_batch_size)) for idx in indices]
        unpacked_TL = [self.unpack_batch(batch) for batch in batches_TL]  # [[o, a, r, no, t],...]
        # 2、采集SL数据
        neighbor_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for ind in indices:
            if ind in neighbor_indices:
                neighbor_indices.remove(ind)
        SL_batch_size = self.batch_size - TL_batch_size
        SL_batch_size_per_task = math.ceil(SL_batch_size / len(neighbor_indices))
        SL_batch_size_total = 0
        batches_SL_temp = []
        for ind in neighbor_indices:
            if SL_batch_size_total < SL_batch_size:
                if SL_batch_size - SL_batch_size_total < SL_batch_size_per_task:
                    SL_batch_size_per_task = SL_batch_size - SL_batch_size_total
                batches_SL_temp.append(ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(ind, batch_size=SL_batch_size_per_task)))
                SL_batch_size_total = SL_batch_size_total + SL_batch_size_per_task
        batches_SL = [batches_SL_temp[0]]
        for ind in range(1, len(batches_SL_temp)):
            for key in batches_SL_temp[ind].keys():
                batches_SL[0][key] = torch.cat([batches_SL_temp[ind][key], batches_SL[0][key]], dim=0)
        unpacked_SL = [self.unpack_batch(batch2) for batch2 in batches_SL]  # 变成[[o, a, r, no, t],...]
        unpacked = [[]]
        for ind in range(len(unpacked_TL[0])):
            unpacked[0].append(torch.cat([unpacked_TL[0][ind], unpacked_SL[0][ind]], dim=1))
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]  # 变成[ [[o], [a], [r], [no], [t]] , ... ]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]  # 变成[[o,...], [a,...], [r,...], [no,...], [t,...]]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):  # 判断indices对象里面是否有'__iter__'属性（方法）
            indices = [indices]
        batches = []
        for idx in indices:
            batch_dict = self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size,
                                                             sequence=self.recurrent)
            del batch_dict['weights']
            batches.append(ptu.np_to_pytorch_batch(batch_dict))
        # batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]

        # 得到每个任务的 p
        p_batch = self.env.get_batch_p(indices)
        for idx_temp in range(len(p_batch)):
            p_batch[idx_temp] = torch.unsqueeze(p_batch[idx_temp].repeat(self.embedding_batch_size, 1), 0)

        if len(p_batch) != 1:
            p_batch_temp = p_batch[0]
            for temp in range(1, len(p_batch)):
                p_batch_temp = torch.cat([p_batch_temp, p_batch[temp]], dim=0)
            p_batch = [p_batch_temp]

        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1]+p_batch, dim=2)
        else:
            context = torch.cat(context[:-2] + p_batch, dim=2)
        # return 的时候删除weight权重
        return context

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size  # 32
        num_updates = self.embedding_batch_size // mb_size  # 32 // 32 = 1

        # sample context batch
        context_batch = self.sample_context(indices)
        # 得到每个任务的 p
        # p_batch = self.env.get_batch_p(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size,
                      :]  # context.shape:  torch.Size([16, 32, 15])
            self._take_step(indices, context)
            # stop backprop
            self.agent.detach_z()

    def _do_training_meta(self, indices):
        mb_size = self.embedding_mini_batch_size  # 32
        num_updates = self.embedding_batch_size // mb_size  # 32 // 32 = 1

        # sample context batch
        context_batch = self.sample_context(indices)    # 在这个函数中返回的context中包含了personal data信息
        # 得到每个任务的 p
        # p_batch = self.env.get_batch_p(indices)
        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size,
                      :]  # context.shape:  torch.Size([16, 32, 15])
            self._take_step_meta_testing(indices, context)
            # stop backprop
            self.agent.detach_z()

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step_meta_testing(self, indices, context):
        """
        :param indices:
        :param context:
        :param p_batch:
        :return:
        """
        num_tasks = len(indices)  # 16

        # data is (task, batch, feat)
        obs, actions, rewards, weights, next_obs, terms = self.sample_sac_meta(
            indices)  # "batch_size": 256,"meta_batch": 16,   terms:是否done
        # torch.Size([16, 256, 11]) torch.Size([16, 256, 3]) torch.Size([16, 256, 1]) torch.Size([16, 256, 11]) torch.Size([16, 256, 1])

        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)  # forward(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()  # 总共16*256个transitions
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())  # 将 task_z 从网络中隔离开,不参与参数更新
        # get targets for use in V and Q updates
        # target网络不参与梯度的回传
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:  # use_information_bottleneck = True 表示使用KL散度信息
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div  # kl_lambda=.1
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        weights_flat = weights.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale  # reward_scale=5.
        terms_flat = terms.view(self.batch_size * num_tasks, -1)  # terms_flat: tensor([0.])
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2 * weights_flat) + torch.mean(
            (q2_pred - q_target) ** 2 * weights_flat)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion((weights_flat ** (1 / 2)) * v_pred, (weights_flat ** (1 / 2)) * v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                (log_pi - log_policy_target) * weights_flat
        ).mean()

        mean_reg_loss = (self.policy_mean_reg_weight * weights_flat * (policy_mean ** 2)).mean()
        std_reg_loss = (self.policy_std_reg_weight * weights_flat * (policy_log_std ** 2)).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def _take_step(self, indices, context):

        num_tasks = len(indices)  # 16

        # data is (task, batch, feat)
        obs, actions, rewards, weights, next_obs, terms = self.sample_sac(
            indices)  # "batch_size": 256,"meta_batch": 16,   terms:是否done
        # torch.Size([16, 256, 11]) torch.Size([16, 256, 3]) torch.Size([16, 256, 1]) torch.Size([16, 256, 11]) torch.Size([16, 256, 1])

        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)  # forward(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()  # 总共16*256个transitions
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())  # 将 task_z 从网络中隔离开,不参与参数更新
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:  # use_information_bottleneck = True 表示使用KL散度信息
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div  # kl_lambda=.1
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        weights_flat = weights.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale  # reward_scale=5.
        terms_flat = terms.view(self.batch_size * num_tasks, -1)  # terms_flat: tensor([0.])
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2 * weights_flat) + torch.mean(
            (q2_pred - q_target) ** 2 * weights_flat)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion((weights_flat ** (1 / 2)) * v_pred, (weights_flat ** (1 / 2)) * v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                (log_pi - log_policy_target) * weights_flat
        ).mean()

        mean_reg_loss = (self.policy_mean_reg_weight * weights_flat * (policy_mean ** 2)).mean()
        std_reg_loss = (self.policy_std_reg_weight * weights_flat * (policy_log_std ** 2)).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
        )
        return snapshot

    def kernel(self, ker, X1, X2, gamma):
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

    def TASL(self, SL, SU, TL, TU, gSU, nQ, lamb, num_iter=2, gamma=1.0):
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
        gSU = gSU.cpu().detach().numpy()

        nSL = SL.shape[0]
        nSU = SU.shape[0]
        nTL = TL.shape[0]
        nTU = TU.shape[0]
        nL = nSL + nQ + nTL
        nU = nSU - nQ + nTU
        nS = nSU + nSL
        nT = nTU + nTL
        print(" nSL:", nSL, "\n nSU:", nSU, "\n nTL:", nTL, "\n nTU:", nTU,
              "\n nL:", nL, "\n nU:", nU, "\n nS:", nS, "\n nT:", nT)
        print("1 TASL: Calculate the kernel matrix K and initialize beta")
        # # 1 Calculate the kernel matrix K and initialize beta;
        beta = np.ones((nSL + nSU, 1))
        #   Calculate the kernel matrix K
        print("\t\t\tK_SU")
        K_SU = self.kernel('rbf', SU, None, gamma)
        print("\t\t\tK_SU_SL")
        K_SU_SL = self.kernel('rbf', SU, SL, gamma)
        print("\t\t\tK_SU_TL")
        K_SU_TL = self.kernel('rbf', SU, TL, gamma)
        print("\t\t\tK_SU_TU")
        K_SU_TU = self.kernel('rbf', SU, TU, gamma)
        print("\t\t\tK_SL")
        K_SL = self.kernel('rbf', SL, None, gamma)
        print("\t\t\tK_SL_SU")
        K_SL_SU = self.kernel('rbf', SL, SU, gamma)
        print("\t\t\tK_SL_TL")
        K_SL_TL = self.kernel('rbf', SL, TL, gamma)
        print("\t\t\tK_SL_TU")
        K_SL_TU = self.kernel('rbf', SL, TU, gamma)
        print("\t\t\tK_S")
        K_S = self.kernel('rbf', np.r_[SU, SL], None, gamma)

        print("2 TASL: For each active querying iteration")
        # 2 For each active querying iteration:
        for step in range(num_iter):
            print("2.1 TASL: Update alpha by solving Eq. (5)")
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
                + float(lamb / 2.) * beta[1:nSU + 1, :] * abs(gSU)  ############# 注意这里gSU在原文中是 求各元素绝对值 还是 求矩阵gSU的行列式值？
            a = matrix(a)
            G_a = matrix(np.r_[np.eye(nSU), -np.eye(nSU)])
            h_a = matrix(np.r_[np.ones((nSU,)), np.zeros((nSU,))])
            A_a = matrix(np.ones((1, nSU)))
            b_a = matrix(float(nQ))
            print('A.shape', A.size)
            print('a.shape', a.size)
            print('G_a.shape', G_a.size)
            print('h_a.shape', h_a.size)
            print('A_a.shape', A_a.size)
            print('b_a.shape', b_a.size)
            sol_alpha = solvers.qp(A, a, G_a, h_a, A_a, b_a)
            alpha = np.array(sol_alpha['x'])
            print("2.2 TASL: Update beta by solving Eq. (6)")
            ## 2.2 Update beta by solving Eq. (6);
            B11 = matrix(float(1. / (nL ** 2)) * K_SL)
            B12 = matrix(((float(1. / (nL ** 2)) + float(1. / (nL * nU))) * np.dot(np.ones((nSL, 1)),
                                                                                   np.transpose(alpha)) - float(
                1. / (nL * nU)) * np.ones((nSL, nSU))) * K_SL_SU)
            B21 = matrix(((float(1. / (nL ** 2)) + float(1. / (nL * nU))) * np.dot(alpha, np.transpose(
                np.ones((nSL, 1)))) - float(1. / (nL * nU)) * np.ones((nSU, nSL))) * K_SU_SL)
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
            b1 = matrix(
                float(1. / (nL ** 2)) * np.dot(K_SL_TL, np.ones((nTL, 1))) - float(1. / (nL * nU)) * np.dot(K_SL_TU,
                                                                                                            np.ones((
                                                                                                                nTU,
                                                                                                                1))))
            b2 = matrix(
                np.dot(K_SU_TL, np.ones((nTL, 1))) * (
                        (float(1. / (nL ** 2)) + float(1. / (nL * nU))) * alpha - float(1. / (nL * nU)) * np.ones(
                    (nSU, 1)))
                - np.dot(K_SU_TU, np.ones((nTU, 1))) * (
                        (float(1. / (nL ** 2)) + float(1. / (nL * nU))) * alpha - float(1. / (nU ** 2)) * np.ones(
                    (nSU, 1)))
                - lamb * alpha * abs(gSU)
            )
            b = matrix(np.r_[b1, b2])
            G_b = matrix(np.r_[np.eye(nS), -np.eye(nS)])
            h_b = matrix(np.r_[np.ones((nS,)), np.zeros((nS,))])
            sol_beta = solvers.qp(B, b, G_b, h_b)
            beta = np.array(sol_beta['x'])
        print("3 TASL: Q <-- top nQ instances of SU with largest alpha values")
        # 3 Q <-- top nQ instances of SU with largest alpha values;
        Q_index = heapq.nlargest(nQ, range(len(alpha)), alpha.take)
        # 4 SU = SU \ Q; SL = SL and Q;
        # 5 Train the model based on TL and adapted SL with beta.
        return alpha, beta, Q_index

    def mmd_meta_testing(self, task_idx, source_idx, iter, num_iter_step, nQ):
        """
        与 task_idx 上进行 基于主动学习的迁移
        :param task_idx: 目标域任务
        :param iter:更新轮数
        :param num_iter_step:每一轮走的步数
        :return:
        """
        print('使用病人：', task_idx + 1)
        db = pymysql.connect(host=mysql_config.mysql_host, user=mysql_config.mysql_user,password=mysql_config.mysql_password, database=mysql_config.mysql_database)
        cursor = db.cursor()  # 数据库操作
        sql = "insert into patient(id)values('%d')" % (
                task_idx + 1)  # 存入数据库
        cursor.execute(sql)  # 执行数据库语句
        db.commit()  # 提交
        with open(os.path.join(result_path, 'dataDeal2.txt'), 'a+') as dfile:
            dfile.write('\npatient:' + str(task_idx + 1) + '\n')

        db = pymysql.connect(host=mysql_config.mysql_host, user=mysql_config.mysql_user,password=mysql_config.mysql_password, database=mysql_config.mysql_database)
        cursor = db.cursor()  # 数据库操作
        print("successful")

        sql = "select * from episode"
        cursor.execute(sql)  # 查询
        db.commit()
        episode_arr = cursor.fetchall()
        episode_alen = episode_arr.__len__()

        sql = "insert into episode(id)values('%d')" % (
                episode_arr[episode_alen - 1][0] + 1)
        cursor.execute(sql)
        db.commit()

        sql = "delete from state"
        cursor.execute(sql)
        db.commit()
        sql = "delete from action"
        cursor.execute(sql)
        db.commit()
        sql = "delete from sasrecord"
        cursor.execute(sql)
        db.commit()

        self.replay_buffer.task_buffers[task_idx].clear()
        self.enc_replay_buffer.task_buffers[task_idx].clear()
        self.task_idx = task_idx
        self.env.reset_task(task_idx)
        self.agent.clear_z()
        num_transitions = 0
        num_trajs = 0

        self.env.num = 0
        self.env.anum = 0

        o = self.env.reset()
        next_o = None
        for epoch in range(iter):
            path_length = 0
            observations = []
            actions = []
            rewards = []
            terminals = []
            agent_infos = []
            env_infos = []
            while path_length < num_iter_step:
                # num_flag = False
                p = self.env.personal_feature.cpu()
                p = p.numpy()[0]
                o = np.array(o)
                a, agent_info = self.agent.get_action(o)
                db = pymysql.connect(host=mysql_config.mysql_host, user=mysql_config.mysql_user,password=mysql_config.mysql_password, database=mysql_config.mysql_database)
                cursor = db.cursor()  # 数据库操作
                sql = "insert into action(id,dose)values('%d','%lf')" % (
                    self.env.anum, a)  # 存入数据库
                cursor.execute(sql)  # 执行数据库语句
                db.commit()  # 提交
                self.env.anum += 1
                next_o, r, d, env_info = self.env.step(a)
                print('step', next_o, r, d, env_info)
                self.agent.update_context([o, a, r, next_o, d, env_info], p)
                self.agent.infer_posterior(self.agent.context)
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

            path = dict(
                observations=observations,
                actions=actions,
                rewards=np.array(rewards).reshape(-1, 1),  # 将数据变成1列
                weights=np.array(weights).reshape(-1, 1),
                next_observations=next_observations,
                terminals=np.array(terminals).reshape(-1, 1),
                agent_infos=agent_infos,
                env_infos=env_infos,
            )

            path['context'] = self.sampler.policy.z.detach().cpu().numpy()
            self.replay_buffer.add_paths(self.task_idx, [path])  # 添加到 RL replay buffer 中

            SU = np.array(self.replay_buffer_SU.get_all_data(source_idx))
            TU = np.array(self.replay_buffer_TU.get_all_data([task_idx]))
            SL = np.array(self.replay_buffer.get_all_data(source_idx))
            TL = np.array(self.replay_buffer.get_all_data([task_idx]))

            gSU_tensor_list = []
            for idx in source_idx:
                obs = [np.array(self.replay_buffer_SU.get_obs_data([idx]))]
                obs = ptu.from_numpy(np.array(obs))
                p_batch = self.env.get_batch_p([idx])
                context = self.sample_context([idx])
                policy_outputs, task_z = self.agent(obs, context)  # forward(obs, context)
                new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
                gSU_tensor_list.append(policy_log_std.exp())
            gSU = gSU_tensor_list[0]
            for ind in range(1, len(gSU_tensor_list)):
                gSU = torch.cat([gSU, gSU_tensor_list[ind]], dim=0)

            alpha, beta, Q_index = self.TASL(SL, SU, TL, TU, gSU, nQ, 0.5)
            Q_index = np.array(Q_index)

            temp_ind = 0
            for ind in source_idx:
                self.replay_buffer.task_buffers[ind].change_weights(
                    beta[temp_ind : temp_ind + self.replay_buffer.task_buffers[ind]._size])
                temp_ind = temp_ind + self.replay_buffer.task_buffers[ind]._size
            for ind in source_idx:
                self.replay_buffer_SU.task_buffers[ind].change_weights(
                    beta[temp_ind : temp_ind + self.replay_buffer_SU.task_buffers[ind]._size])
                temp_ind = temp_ind + self.replay_buffer_SU.task_buffers[ind]._size

            temp_ind = 0
            for ind in source_idx:
                print("Q_index:", Q_index)
                total_cur = temp_ind + self.replay_buffer_SU.task_buffers[ind]._size
                print("total_cur:", total_cur)
                indices = Q_index[np.where(Q_index < (total_cur))]    # 返回索引
                print("本次需要查询的 indices:", indices)
                if len(indices) != 0:
                    indices = indices - temp_ind
                    print("查询前 replay buffer SU:", self.replay_buffer_SU.task_buffers[ind]._size)
                    print(indices)
                    add_data = self.replay_buffer_SU.task_buffers[ind].sample_data(indices)    # 取出SU元素
                    self.replay_buffer_SU.task_buffers[ind].delete_buffer(indices)     # 删除SU元素
                    print("查询后 replay buffer SU:", self.replay_buffer_SU.task_buffers[ind]._size)
                    self.replay_buffer.add_path(ind, add_data)
                    print("查询后 replay buffer SL:", self.replay_buffer.task_buffers[ind]._size)
                indices_ind = np.where(Q_index < total_cur)
                Q_index = np.delete(Q_index, indices_ind, axis=0)
                temp_ind = total_cur

            # training
            self._do_training_meta([task_idx])

        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        next_o = None
        for step in range(self.sampler.max_path_length - iter * num_iter_step):
            p = self.env.personal_feature.cpu()
            p = p.numpy()[0]
            o = np.array(o)
            a, agent_info = self.agent.get_action(o)
            db = pymysql.connect(host=mysql_config.mysql_host, user=mysql_config.mysql_user,password=mysql_config.mysql_password, database=mysql_config.mysql_database)
            cursor = db.cursor()  # 数据库操作
            sql = "insert into action(id,dose)values('%d','%lf')" % (
                self.env.anum, a)  # 存入数据库
            cursor.execute(sql)  # 执行数据库语句
            db.commit()  # 提交
            self.env.anum += 1
            next_o, r, d, env_info = self.env.step(a)
            self.agent.update_context([o, a, r, next_o, d, env_info], p)
            self.agent.infer_posterior(self.agent.context)

            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            agent_infos.append(agent_info)
            o = next_o
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
        )

        weights = np.ones(np.array(rewards).shape)

        path = dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),  # 将数据变成1列
            weights=np.array(weights).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
        )

        path['context'] = self.sampler.policy.z.detach().cpu().numpy()

        self.replay_buffer.add_paths(self.task_idx, [path])

        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)  # 记录模型参数
