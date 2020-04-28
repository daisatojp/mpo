import os
from time import sleep
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from IPython import embed
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
import gym
from mpo.actor import ActorContinuous, ActorDiscrete
from mpo.critic import CriticContinuous, CriticDiscrete
from mpo.replaybuffer import ReplayBuffer


def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def gaussian_kl(μi, μ, Ai, A):
    """
    decoupled KL between two multivariate gaussian distribution
    C_μ = KL(f(x|μi,Σi)||f(x|μ,Σi))
    C_Σ = KL(f(x|μi,Σi)||f(x|μi,Σ))
    :param μi: (B, n)
    :param μ: (B, n)
    :param Ai: (B, n, n)
    :param A: (B, n, n)
    :return: C_μ, C_Σ: mean and covariance terms of the KL
    """
    n = A.size(-1)
    μi = μi.unsqueeze(-1)  # (B, n, 1)
    μ = μ.unsqueeze(-1)  # (B, n, 1)
    Σi = Ai @ bt(Ai)  # (B, n, n)
    Σ = A @ bt(A)  # (B, n, n)
    Σi_inv = Σi.inverse()  # (B, n, n)
    Σ_inv = Σ.inverse()  # (B, n, n)
    inner_μ = ((μ - μi).transpose(-2, -1) @ Σi_inv @ (μ - μi)).squeeze()  # (B,)
    inner_Σ = torch.log(Σ.det() / Σi.det()) - n + btr(Σ_inv @ Σi)  # (B,)
    C_μ = 0.5 * torch.mean(inner_μ)
    C_Σ = 0.5 * torch.mean(inner_Σ)
    return C_μ, C_Σ


def categorical_kl(p1, p2):
    """
    calculates KL between two Categorical distributions
    :param p1: (B, D)
    :param p2: (B, D)
    """
    p1 = torch.clamp_min(p1, 0.0001)
    p2 = torch.clamp_min(p2, 0.0001)
    return torch.mean((p1 * torch.log(p1 / p2)).sum(dim=-1))


class MPO(object):
    """
    Maximum A Posteriori Policy Optimization (MPO)
    :param env: (Gym Environment) gym environment to learn on
    :param dual_constraint: (float) hard constraint of the dual formulation in the E-step
    :param mean_constraint: (float) hard constraint of the mean in the M-step
    :param var_constraint: (float) hard constraint of the covariance in the M-step
    :param learning_rate: (float) learning rate in the Q-function
    :param alpha: (float) scaling factor of the lagrangian multiplier in the M-step
    :param episodes: (int) number of training (evaluation) episodes
    :param episode_length: (int) step size of one episode
    :param lagrange_it: (int) number of optimization steps of the Lagrangian
    :param mb_size: (int) size of the sampled mini-batch
    :param sample_episodes: (int) number of sampling episodes
    :param add_act: (int) number of additional actions
    :param actor_layers: (tuple) size of the hidden layers in the actor net
    :param critic_layers: (tuple) size of the hidden layers in the critic net
    :param log: (boolean) saves log if True
    :param log_dir: (str) directory in which log is saved
    :param render: (boolean) renders the simulation if True
    :param save: (boolean) saves the model if True
    :param save_path: (str) path for saving and loading a model
    """
    def __init__(self,
                 device,
                 env,
                 policy_evaluation='td',
                 dual_constraint=0.1,
                 kl_mean_constraint=0.01,
                 kl_var_constraint=0.01,
                 kl_constraint=0.01,
                 discount_factor=0.99,
                 alpha=10,
                 sample_process_num=5,
                 sample_episode_num=30,
                 sample_episode_maxlen=200,
                 sample_action_num=64,
                 backward_length=0,
                 episode_rerun_num=5,
                 batch_size=64,
                 lagrange_iteration_num=5):
        self.device = device
        self.env = env
        if self.env.action_space.dtype == np.float32:
            self.continuous_action_space = True
        else:
            self.continuous_action_space = False
        self.ds = env.observation_space.shape[0]
        if self.continuous_action_space:
            self.da = env.action_space.shape[0]
        else:
            self.da = env.action_space.n

        self.policy_evaluation = policy_evaluation
        self.ε_dual = dual_constraint
        self.ε_kl_μ = kl_mean_constraint  # hard constraint for the KL
        self.ε_kl_Σ = kl_var_constraint  # hard constraint for the KL
        self.ε_kl = kl_constraint
        self.γ = discount_factor
        self.α = alpha  # scaling factor for the update step of η_μ
        self.sample_process_num = sample_process_num
        self.sample_episode_num = sample_episode_num
        self.sample_episode_maxlen = sample_episode_maxlen
        self.sample_action_num = sample_action_num
        self.episode_rerun_num = episode_rerun_num
        self.batch_size = batch_size
        self.lagrange_iteration_num = lagrange_iteration_num

        if not self.continuous_action_space:
            self.A_eye = torch.eye(self.da).to(self.device)

        if self.continuous_action_space:
            self.actor = ActorContinuous(env).to(self.device)
            self.critic = CriticContinuous(env).to(self.device)
            self.target_actor = ActorContinuous(env).to(self.device)
            self.target_critic = CriticContinuous(env).to(self.device)
        else:
            self.actor = ActorDiscrete(env).to(self.device)
            self.critic = CriticDiscrete(env).to(self.device)
            self.target_actor = ActorDiscrete(env).to(self.device)
            self.target_critic = CriticDiscrete(env).to(self.device)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.norm_loss_q = nn.SmoothL1Loss()

        self.η = np.random.rand()
        self.η_kl_μ = 0.0
        self.η_kl_Σ = 0.0
        self.η_kl = 0.0

        self.replaybuffer = ReplayBuffer(backward_length=backward_length)

        self.iteration = 0
        self.render = False

    def train(self, iteration_num=100, log_dir='log', model_save_period=10, render=False):
        """
        :param iteration_num:
        :param log_dir:
        :param model_save_period:
        :param render:
        """

        self.render = render

        model_save_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        writer = SummaryWriter(os.path.join(log_dir, 'tb'))

        for it in range(self.iteration, iteration_num):
            self.__sample_trajectory(self.sample_episode_num)
            buff_sz = len(self.replaybuffer)

            mean_reward = self.replaybuffer.mean_reward()
            mean_return = self.replaybuffer.mean_return()
            mean_loss_q = []
            mean_loss_p = []
            mean_loss_l = []
            mean_est_q = []
            max_kl_μ = []
            max_kl_Σ = []
            max_kl = []

            # Find better policy by gradient descent
            for _ in range(self.episode_rerun_num):
                for indices in tqdm(BatchSampler(SubsetRandomSampler(range(buff_sz)), self.batch_size, False)):
                    B = len(indices)
                    M = self.sample_action_num
                    L = self.replaybuffer.backward_length
                    ds = self.ds
                    da = self.da

                    states_batch, actions_batch, next_states_batch, rewards_batch = zip(
                        *[self.replaybuffer[index] for index in indices])

                    states_batch = torch.from_numpy(np.stack(states_batch)).type(torch.float32).to(self.device)  # (B, L, ds)
                    actions_batch = torch.from_numpy(np.stack(actions_batch)).type(torch.float32).to(self.device)  # (B, L, da) or (B, L)
                    next_states_batch = torch.from_numpy(np.stack(next_states_batch)).type(torch.float32).to(self.device)  # (B, L, ds)
                    rewards_batch = torch.from_numpy(np.stack(rewards_batch)).type(torch.float32).to(self.device)  # (B, L)

                    state_batch = states_batch[:, 0, :]  # (B, ds)
                    if self.continuous_action_space:
                        action_batch = actions_batch[:, 0, :]  # (B, da)
                    else:
                        action_batch = actions_batch[:, 0]  # (B,)
                    next_state_batch = next_states_batch[:, 0, :]  # (B, ds)
                    reward_batch = rewards_batch[:, 0]  # (B,)

                    # Policy Evaluation
                    loss_q = None
                    q = None
                    if self.policy_evaluation == 'td':
                        loss_q, q = self.__update_critic_td(
                            state_batch=state_batch,
                            action_batch=action_batch,
                            next_state_batch=next_state_batch,
                            reward_batch=reward_batch,
                            sample_num=self.sample_action_num
                        )
                    if self.policy_evaluation == 'retrace':
                        loss_q = self.__update_critic_retrace(
                            states_batch=states_batch,
                            actions_batch=actions_batch,
                            next_states_batch=next_states_batch,
                            rewards_batch=rewards_batch,
                            A=self.sample_action_num
                        )
                    if loss_q is None:
                        raise RuntimeError('invalid policy evaluation')
                    mean_loss_q.append(loss_q.item())
                    mean_est_q.append(q.abs().mean().item())

                    # sample M additional action for each state
                    with torch.no_grad():
                        if self.continuous_action_space:
                            b_μ, b_A = self.target_actor.forward(state_batch)  # (B,)
                            b = MultivariateNormal(b_μ, scale_tril=b_A)  # (B,)
                            sampled_actions = b.sample((M,))  # (M, B, da)
                            expanded_states = state_batch[None, ...].expand(M, -1, -1)  # (M, B, ds)
                            target_q = self.target_critic.forward(
                                expanded_states.reshape(-1, ds),  # (M * B, ds)
                                sampled_actions.reshape(-1, da)  # (M * B, da)
                            ).reshape(M, B)  # (M, B)
                            target_q_np = target_q.cpu().numpy()  # (M, B)
                        else:
                            actions = torch.arange(da)[..., None].expand(da, B).to(self.device)  # (da, B)
                            b_p = self.target_actor.forward(state_batch)  # (B, da)
                            b = Categorical(probs=b_p)  # (B,)
                            b_prob = b.expand((da, B)).log_prob(actions).exp()  # (da, B)
                            expanded_actions = self.A_eye[None, ...].expand(B, -1, -1)  # (B, da, da)
                            expanded_states = state_batch.reshape(B, 1, ds).expand((B, da, ds))  # (B, da, ds)
                            target_q = (
                                self.target_critic.forward(
                                    expanded_states.reshape(-1, ds),  # (B * da, ds)
                                    expanded_actions.reshape(-1, da)  # (B * da, da)
                                ).reshape(B, da)  # (B, da)
                            ).transpose(0, 1)  # (da, B)
                            b_prob_np = b_prob.cpu().numpy()  # (da, B)
                            target_q_np = target_q.cpu().numpy()  # (da, B)

                    # E-step
                    # Update Dual-function
                    def dual(η):
                        """
                        Dual function of the non-parametric variational
                        g(η) = η*ε + η \sum \log (\sum \exp(Q(a, s)/η))
                        """
                        max_q = np.max(target_q_np, 0)
                        return η * self.ε_dual + np.mean(max_q) \
                            + η * np.mean(np.log(np.mean(np.exp((target_q_np - max_q) / η), 0)))

                    bounds = [(1e-6, None)]
                    res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
                    self.η = res.x[0]

                    qij = torch.softmax(target_q / self.η, dim=0)  # (M, B)

                    # M-step
                    # update policy based on lagrangian
                    for _ in range(self.lagrange_iteration_num):
                        if self.continuous_action_space:
                            μ, A = self.actor.forward(state_batch)
                            π1 = MultivariateNormal(loc=μ, scale_tril=b_A)  # (B,)
                            π2 = MultivariateNormal(loc=b_μ, scale_tril=A)  # (B,)
                            loss_p = torch.mean(
                                qij * (
                                    π1.expand((M, B)).log_prob(sampled_actions)  # (M, B)
                                    + π2.expand((M, B)).log_prob(sampled_actions)  # (M, B)
                                )
                            )
                            mean_loss_p.append((-loss_p).item())

                            kl_μ, kl_Σ = gaussian_kl(
                                μi=b_μ, μ=μ,
                                Ai=b_A, A=A)
                            max_kl_μ.append(kl_μ.item())
                            max_kl_Σ.append(kl_Σ.item())

                            if np.isnan(kl_μ.item()):
                                print('kl_μ is nan')
                                embed()
                            if np.isnan(kl_Σ.item()):
                                print('kl_Σ is nan')
                                embed()

                            # Update lagrange multipliers by gradient descent
                            self.η_kl_μ -= self.α * (self.ε_kl_μ - kl_μ).detach().item()
                            self.η_kl_Σ -= self.α * (self.ε_kl_Σ - kl_Σ).detach().item()

                            if self.η_kl_μ < 0.0:
                                self.η_kl_μ = 0.0
                            if self.η_kl_Σ < 0.0:
                                self.η_kl_Σ = 0.0

                            self.actor_optimizer.zero_grad()
                            loss_l = -(
                                    loss_p
                                    + self.η_kl_μ * (self.ε_kl_μ - kl_μ)
                                    + self.η_kl_Σ * (self.ε_kl_Σ - kl_Σ)
                            )
                            mean_loss_l.append(loss_l.item())
                            loss_l.backward()
                            clip_grad_norm_(self.actor.parameters(), 0.1)
                            self.actor_optimizer.step()
                        else:
                            π_p = self.actor.forward(state_batch)  # (B, da)
                            π = Categorical(probs=π_p)  # (B,)
                            loss_p = torch.mean(
                                qij * π.expand((da, B)).log_prob(actions)
                            )
                            mean_loss_p.append((-loss_p).item())

                            kl = categorical_kl(p1=π_p, p2=b_p)
                            max_kl.append(kl.item())

                            # Update lagrange multipliers by gradient descent
                            self.η_kl -= self.α * (self.ε_kl - kl).detach().item()

                            if self.η_kl < 0.0:
                                self.η_kl = 0.0

                            self.actor_optimizer.zero_grad()
                            loss_l = -(loss_p + self.η_kl * (self.ε_kl - kl))
                            mean_loss_l.append(loss_l.item())
                            loss_l.backward()
                            clip_grad_norm_(self.actor.parameters(), 0.1)
                            self.actor_optimizer.step()

            self.__update_param()

            self.η_kl_μ = 0.0
            self.η_kl_Σ = 0.0
            self.η_kl = 0.0

            it = it + 1
            mean_loss_q = np.mean(mean_loss_q)
            mean_loss_p = np.mean(mean_loss_p)
            mean_loss_l = np.mean(mean_loss_l)
            mean_est_q = np.mean(mean_est_q)
            if self.continuous_action_space:
                max_kl_μ = np.max(max_kl_μ)
                max_kl_Σ = np.max(max_kl_Σ)
            else:
                max_kl = np.max(max_kl)

            print('iteration :', it)
            print('  mean return :', mean_return)
            print('  mean reward :', mean_reward)
            print('  mean loss_q :', mean_loss_q)
            print('  mean loss_p :', mean_loss_p)
            print('  mean loss_l :', mean_loss_l)
            print('  mean est_q :', mean_est_q)
            print('  η :', self.η)
            if self.continuous_action_space:
                print('  max_kl_μ :', max_kl_μ)
                print('  max_kl_Σ :', max_kl_Σ)
            else:
                print('  max_kl :', max_kl)

            # saving and logging
            self.save_model(os.path.join(model_save_dir, 'model_latest.pt'))
            if it % model_save_period == 0:
                self.save_model(os.path.join(model_save_dir, 'model_{}.pt'.format(it)))
            writer.add_scalar('return', mean_return, it)
            writer.add_scalar('reward', mean_reward, it)
            writer.add_scalar('loss_q', mean_loss_q, it)
            writer.add_scalar('loss_p', mean_loss_p, it)
            writer.add_scalar('loss_l', mean_loss_l, it)
            writer.add_scalar('mean_q', mean_est_q, it)
            writer.add_scalar('η', self.η, it)
            if self.continuous_action_space:
                writer.add_scalar('max_kl_μ', max_kl_μ, it)
                writer.add_scalar('max_kl_Σ', max_kl_Σ, it)
            else:
                writer.add_scalar('η_kl', max_kl, it)
            writer.flush()

        # end training
        if writer is not None:
            writer.close()

    def load_model(self, path=None):
        """
        loads a model from a given path
        :param path: (str) file path (.pt file)
        """
        load_path = path if path is not None else self.save_path
        checkpoint = torch.load(load_path)
        self.iteration = checkpoint['iteration']
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic.train()
        self.target_critic.train()
        self.actor.train()
        self.target_actor.train()

    def save_model(self, path=None):
        """
        saves the model
        :param path: (str) file path (.pt file)
        """
        data = {
            'iteration': self.iteration,
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict()
        }
        torch.save(data, path)

    def sample_trajectory_worker(self, i):
        buff = []
        state = self.env.reset()
        for steps in range(self.sample_episode_maxlen):
            action = self.target_actor.action(
                torch.from_numpy(state).type(torch.float32).to(self.device)
            ).cpu().numpy()
            next_state, reward, done, _ = self.env.step(action)
            buff.append((state, action, next_state, reward))
            if self.render and ((mp.current_process()._identity[0] % self.sample_process_num) == 1):
                self.env.render()
                sleep(0.01)
            if done:
                break
            else:
                state = next_state
        return buff

    def __sample_trajectory(self, sample_episode_num):
        self.replaybuffer.clear()
        with Pool(self.sample_process_num) as p:
            episodes = p.map(self.sample_trajectory_worker, range(sample_episode_num))
        self.replaybuffer.store_episodes(episodes)

    def __update_critic_td(self,
                           state_batch,
                           action_batch,
                           next_state_batch,
                           reward_batch,
                           sample_num=64):
        """
        :param state_batch: (B, ds)
        :param action_batch: (B, da) or (B,)
        :param next_state_batch: (B, ds)
        :param reward_batch: (B,)
        :param sample_num:
        :return:
        """
        B = state_batch.size(0)
        ds = self.ds
        da = self.da
        with torch.no_grad():
            r = reward_batch  # (B,)
            if self.continuous_action_space:
                π_μ, π_A = self.target_actor.forward(next_state_batch)  # (B,)
                π = MultivariateNormal(π_μ, scale_tril=π_A)  # (B,)
                sampled_next_actions = π.sample((sample_num,)).transpose(0, 1)  # (B, sample_num, da)
                expanded_next_states = next_state_batch[:, None, :].expand(-1, sample_num, -1)  # (B, sample_num, ds)
                expected_next_q = self.target_critic.forward(
                    expanded_next_states.reshape(-1, ds),  # (B * sample_num, ds)
                    sampled_next_actions.reshape(-1, da)  # (B * sample_num, da)
                ).reshape(B, sample_num).mean(dim=1)  # (B,)
            else:
                π_p = self.target_actor.forward(next_state_batch)  # (B, da)
                π = Categorical(probs=π_p)  # (B,)
                π_prob = π.expand((da, B)).log_prob(
                    torch.arange(da)[..., None].expand(-1, B).to(self.device)  # (da, B)
                ).exp().transpose(0, 1)  # (B, da)
                sampled_next_actions = self.A_eye[None, ...].expand(B, -1, -1)  # (B, da, da)
                expanded_next_states = next_state_batch[:, None, :].expand(-1, da, -1)  # (B, da, ds)
                expected_next_q = (
                    self.target_critic.forward(
                        expanded_next_states.reshape(-1, ds),  # (B * da, ds)
                        sampled_next_actions.reshape(-1, da)  # (B * da, da)
                    ).reshape(B, da) * π_prob  # (B, da)
                ).sum(dim=-1)  # (B,)
            y = r + self.γ * expected_next_q
        self.critic_optimizer.zero_grad()
        if self.continuous_action_space:
            t = self.critic(
                state_batch,
                action_batch
            ).squeeze()
        else:
            t = self.critic(
                state_batch,
                self.A_eye[action_batch.long()]
            ).squeeze(-1)  # (B,)
        loss = self.norm_loss_q(y, t)
        loss.backward()
        self.critic_optimizer.step()
        return loss, y

    def __update_critic_retrace(
            self, states_batch, actions_batch, next_states_batch, rewards_batch, A=64):
        """
        :param states_batch: (B, L, ds)
        :param actions_batch: (B, L, da)
        :param next_states_batch: (B, L, ds)
        :param rewards_batch: (B, L)
        :return:
        """
        B = states_batch.size(0)
        L = states_batch.size(1)
        ds = states_batch.size(-1)
        da = actions_batch.size(-1)
        state_batch = states_batch[:, 0, :]  # (B, ds)
        action_batch = actions_batch[:, 0, :]  # (B, da)

        with torch.no_grad():
            r = rewards_batch  # (B, L)

            Qφ = self.target_critic.forward(
                states_batch.reshape(-1, ds),
                actions_batch.reshape(-1, da)
            ).reshape(B, L)  # (B, L)

            π_μ_next, π_Σ_next = self.actor.forward(next_states_batch.reshape(B * L, ds))  # (B * L,)
            π_next = MultivariateNormal(π_μ_next, scale_tril=π_Σ_next)  # (B * L,)
            next_sampled_actions_batch = π_next.sample((A,)).transpose(0, 1).reshape(B, L, A, da)  # (B, L, A, da)
            next_expanded_states_batch = next_states_batch.reshape(B, L, 1, ds).expand(-1, -1, A, -1)  # (B, L, A, ds)
            EQφ = self.target_critic.forward(
                next_expanded_states_batch.reshape(-1, ds),
                next_sampled_actions_batch.reshape(-1, da)
            ).reshape(B, L, A).mean(dim=-1)  # (B, L)

            π_μ, π_Σ = self.actor.forward(states_batch.reshape(B * L, ds))  # (B * L,)
            π = MultivariateNormal(π_μ.reshape(B, L, 1), scale_tril=π_Σ.reshape(B, L, 1, 1))  # (B, L)
            b_μ, b_Σ = self.target_actor.forward(states_batch.reshape(B * L, ds))  # (B * L,)
            b = MultivariateNormal(b_μ.reshape(B, L, 1), scale_tril=b_Σ.reshape(B, L, 1, 1))  # (B, L)
            c = π.log_prob(actions_batch).exp() / b.log_prob(actions_batch).exp()  # (B, L)
            c = torch.clamp_max(c, 1.0)  # (B, L)
            c[:, 0] = 1.0  # (B, L)
            c = torch.cumprod(c, dim=1)  # (B, L)

            γ = torch.ones(size=(B, L), dtype=torch.float32).to(self.device) * self.γ  # (B, L)
            γ[:, 0] = 1.0  # (B, L)
            γ = torch.cumprod(γ, dim=1)  # (B, L)

            Qret = (Qφ[:, 0] + (γ * c * (r + EQφ - Qφ)).sum(dim=1))  # (B,)

        self.critic_optimizer.zero_grad()
        Qθ = self.critic.forward(state_batch, action_batch).squeeze()  # (B,)
        loss = torch.sqrt(self.mse_loss(Qθ, Qret))
        loss.backward()
        clip_grad_norm_(self.critic.parameters(), 0.1)
        self.critic_optimizer.step()
        return loss.item()

    def __update_param(self):
        """
        Sets target parameters to trained parameter
        """
        # Update policy parameters
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        # Update critic parameters
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
