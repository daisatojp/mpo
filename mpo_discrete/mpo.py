import os
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
import gym
from mpo_discrete.actor import Actor
from mpo_discrete.critic import Critic
from mpo_discrete.replaybuffer import ReplayBuffer


def categorical_kl(p1, p2):
    """
    calculates KL between two Categorical distributions
    :param p1: (B, D)
    :param p2: (B, D)
    """
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
                 env,
                 policy_evaluation='td',
                 dual_constraint=0.1,
                 kl_constraint=0.1,
                 discount_factor=0.99,
                 alpha=10,
                 sample_episode_num=30,
                 sample_episode_maxlen=200,
                 sample_action_num=64,
                 backward_length=0,
                 rerun_num=5,
                 mb_size=64,
                 lagrange_iteration_num=5):
        self.gpu = 0
        self.env = env

        self.policy_evaluation = policy_evaluation
        self.ε = dual_constraint  # hard constraint for the KL
        self.ε_kl = kl_constraint  # hard constraint for the KL
        self.γ = discount_factor
        self.α = alpha  # scaling factor for the update step of η_μ

        self.sample_episode_num = sample_episode_num
        self.sample_episode_maxlen = sample_episode_maxlen
        self.sample_action_num = sample_action_num
        self.rerun_num = rerun_num
        self.mb_size = mb_size
        self.lagrange_iteration_num = lagrange_iteration_num

        self.actor = Actor(env).to(self.gpu)
        self.critic = Critic(env).to(self.gpu)
        self.target_actor = Actor(env).to(self.gpu)
        self.target_critic = Critic(env).to(self.gpu)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.mse_loss = nn.MSELoss()

        # initialize Lagrange Multiplier
        self.η = np.random.rand()
        self.η_kl = np.random.rand()

        # control/log variables
        self.iteration = 0

        self.replaybuffer = ReplayBuffer(backward_length=backward_length)

    def __sample_trajectory(self, sample_episode_num, sample_episode_maxlen, render):
        self.replaybuffer.clear()
        for i in range(sample_episode_num):
            state = self.env.reset()
            for steps in range(sample_episode_maxlen):
                action = self.target_actor.action(
                    torch.from_numpy(state).type(torch.float32).to(self.gpu)
                ).cpu().numpy()
                next_state, reward, done, _ = self.env.step(action)
                if render and i == 0:
                    self.env.render()
                self.replaybuffer.store(state, action, next_state, reward)
                if done:
                    break
                else:
                    state = next_state
            self.replaybuffer.done_episode()

    def __update_critic_td(self, state_batch, action_batch, next_state_batch, reward_batch):
        """
        :param state_batch: (B, ds)
        :param action_batch: (B,)
        :param next_state_batch: (B, ds)
        :param reward_batch: (B,)
        :return:
        """
        B = state_batch.size(0)
        ds = state_batch.size(-1)
        da = self.env.action_space.n
        with torch.no_grad():
            r = reward_batch  # (B,)
            π_p = self.target_actor.forward(next_state_batch)  # (B, da)
            π = Categorical(probs=π_p)  # (B,)
            π_prob = π.expand((da, B)).log_prob(
                torch.arange(da)[..., None].expand(da, B).to(self.gpu)  # (da, B)
            ).exp().transpose(0, 1)  # (B, da)
            sampled_next_actions = self.A_eye[None, ...].expand(B, -1, -1)  # (B, da, da)
            expanded_next_states = next_state_batch.reshape(B, 1, ds).expand((B, da, ds))  # (B, da, ds)
            expected_next_q = (
                self.target_critic.forward(
                    expanded_next_states.reshape(-1, ds),  # (B * da, ds)
                    sampled_next_actions.reshape(-1, da)   # (B * da, da)
                ).reshape(B, da) * π_prob  # (B, da)
            ).sum(dim=-1)  # (B,)
            y = r + self.γ * expected_next_q  # (B,)
        self.critic_optimizer.zero_grad()
        t = self.critic(state_batch, self.A_eye[action_batch.long()]).squeeze(-1)  # (B,)
        loss = self.mse_loss(y, t)
        loss.backward()
        self.critic_optimizer.step()
        return loss.item()

    def __update_critic_retrace(
            self, states_batch, actions_batch, next_states_batch, rewards_batch):
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

            γ = torch.ones(size=(B, L), dtype=torch.float32).to(self.gpu) * self.γ  # (B, L)
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

    def _update_param(self):
        """
        Sets target parameters to trained parameter
        """
        # Update policy parameters
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        # Update critic parameters
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def train(self, iteration_num=100, log_dir='log', render=False):
        """
        :param iteration_num:
        :param log_dir:
        :param render:
        """

        writer = SummaryWriter(os.path.join(log_dir, 'tf'))

        for it in range(self.iteration, iteration_num):
            self.__sample_trajectory(
                self.sample_episode_num, self.sample_episode_maxlen, render)
            buff_sz = len(self.replaybuffer)

            mean_reward = self.replaybuffer.mean_reward()
            mean_return = self.replaybuffer.mean_return()
            mean_q_loss = []
            mean_lagrange = []

            # Find better policy by gradient descent
            for _ in range(self.rerun_num):
                for indices in tqdm(BatchSampler(SubsetRandomSampler(range(buff_sz)), self.mb_size, False)):
                    B = len(indices)
                    L = self.replaybuffer.backward_length

                    states_batch, actions_batch, next_states_batch, rewards_batch = zip(
                        *[self.replaybuffer[index] for index in indices])

                    states_batch = torch.from_numpy(np.stack(states_batch)).type(torch.float32).to(self.gpu)  # (B, L, ds)
                    actions_batch = torch.from_numpy(np.stack(actions_batch)).type(torch.float32).to(self.gpu)  # (B, L)
                    next_states_batch = torch.from_numpy(np.stack(next_states_batch)).type(torch.float32).to(self.gpu)  # (B, L, ds)
                    rewards_batch = torch.from_numpy(np.stack(rewards_batch)).type(torch.float32).to(self.gpu)  # (B, L)

                    state_batch = states_batch[:, 0, :]  # (B, ds)
                    action_batch = actions_batch[:, 0]  # (B,)
                    next_state_batch = next_states_batch[:, 0, :]  # (B, ds)
                    reward_batch = rewards_batch[:, 0]  # (B,)

                    ds = state_batch.size(-1)
                    da = self.env.action_space.n

                    # Policy Evaluation
                    q_loss = None
                    if self.policy_evaluation == 'td':
                        q_loss = self.__update_critic_td(
                            state_batch=state_batch,
                            action_batch=action_batch,
                            next_state_batch=next_state_batch,
                            reward_batch=reward_batch
                        )
                    if self.policy_evaluation == 'retrace':
                        q_loss = self.__update_critic_retrace(
                            states_batch=states_batch,
                            actions_batch=actions_batch,
                            next_states_batch=next_states_batch,
                            rewards_batch=rewards_batch
                        )
                    if q_loss is None:
                        raise RuntimeError('invalid policy evaluation')
                    mean_q_loss.append(q_loss)

                    with torch.no_grad():
                        actions = torch.arange(da)[..., None].expand(da, B).to(self.gpu)  # (da, B)
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
                        return η * self.ε + np.mean(max_q) \
                            + η * np.mean(np.log(np.sum(b_prob_np * np.exp((target_q_np - max_q) / η), 0)))

                    bounds = [(1e-6, None)]
                    res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
                    self.η = res.x[0]

                    qij = torch.softmax(target_q / self.η, dim=0)  # (da, B)

                    # M-step
                    # update policy based on lagrangian
                    for _ in range(self.lagrange_iteration_num):
                        π_p = self.actor.forward(state_batch)  # (B, da)
                        π = Categorical(probs=π_p)  # (B,)
                        loss_pi = torch.mean(qij * π.expand((da, B)).log_prob(actions))

                        kl = categorical_kl(p1=π_p, p2=b_p)

                        # Update lagrange multipliers by gradient descent
                        self.η_kl -= self.α * (self.ε_kl - kl).detach().item()

                        if self.η_kl < 0:
                            self.η_kl = 0

                        self.actor_optimizer.zero_grad()
                        loss_policy = -(loss_pi + self.η_kl * (self.ε_kl - kl))
                        mean_lagrange.append(loss_policy.item())
                        loss_policy.backward()
                        clip_grad_norm_(self.actor.parameters(), 0.1)
                        self.actor_optimizer.step()

            self._update_param()

            mean_q_loss = np.mean(mean_q_loss)
            mean_lagrange = np.mean(mean_lagrange)

            print('Iteration :', it + 1)
            print('  Mean reward : ', mean_reward)
            print('  Mean return : ', mean_return)
            print('  Mean Q loss : ', mean_q_loss)
            print('  Mean Lagrange : ', mean_lagrange)
            print('  η : ', self.η)
            print('  η_kl : ', self.η_kl)

            # saving and logging
            self.save_model(os.path.join(log_dir, 'mpo_model.pt'))
            writer.add_scalar('reward', mean_reward, it + 1)
            writer.add_scalar('return', mean_return, it + 1)
            writer.add_scalar('lagrange', mean_lagrange, it + 1)
            writer.add_scalar('qloss', mean_q_loss, it + 1)
            writer.flush()

        # end training
        if writer is not None:
            writer.close()

    def eval(self, episodes, episode_length, render=True):
        """
        method for evaluating current model (mean reward for a given number of
        episodes and episode length)
        :param episodes: (int) number of episodes for the evaluation
        :param episode_length: (int) length of a single episode
        :param render: (bool) flag if to render while evaluating
        :return: (float) meaned reward achieved in the episodes
        """

        summed_rewards = 0
        for episode in range(episodes):
            reward = 0
            observation = self.env.reset()
            for step in range(episode_length):
                action = self.target_actor.eval_step(observation)
                new_observation, rew, done, _ = self.env.step(action)
                reward += rew
                if render:
                    self.env.render()
                observation = new_observation if not done else self.env.reset()

            summed_rewards += reward
        return summed_rewards/episodes

    def load_model(self, path=None):
        """
        loads a model from a given path
        :param path: (str) file path (.pt file)
        """
        load_path = path if path is not None else self.save_path
        checkpoint = torch.load(load_path)
        self.episode = checkpoint['epoch']
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
