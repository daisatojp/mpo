import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
import gym
from mpo_continuous.actor import Actor
from mpo_continuous.critic import Critic
from mpo_continuous.replaybuffer import ReplayBuffer


def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def gaussian_kl(mean1, mean2, cholesky1, cholesky2):
    """
    calculates the KL between the old and new policy assuming a gaussian distribution
    :param mean1: mean of the actor
    :param mean2: mean of the target actor
    :param cholesky1: ([[float]]) cholesky matrix of the actor covariance
    :param cholesky2: ([[float]]) cholesky matrix of the target actor covariance
    :return: C_μ, C_Σ: ([float],[[float]])mean and covariance terms of the KL
    """
    if mean1.dim() == 2:
        mean1 = mean1.unsqueeze(-1)
    if mean2.dim() == 2:
        mean2 = mean2.unsqueeze(-1)
    Σ1 = cholesky1 @ bt(cholesky1)
    Σ2 = cholesky2 @ bt(cholesky2)
    Σ1_inv = Σ1.inverse()
    inner_Σ = btr(Σ1_inv @ Σ2) - Σ1.size(-1) + torch.log(Σ1.det() / Σ2.det())
    inner_μ = ((mean1 - mean2) @ Σ1_inv @ (mean1 - mean2)).squeeze()
    C_μ = 0.5 * torch.mean(inner_Σ)
    C_Σ = 0.5 * torch.mean(inner_μ)
    return C_μ, C_Σ


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
    def __init__(self, env,
                 dual_constraint=0.1,
                 mean_constraint=0.1,
                 var_constraint=1e-4,
                 discount_factor=0.99,
                 alpha=10,
                 sample_episode_num=30,
                 sample_episode_maxlen=200,
                 rerun_num=5,
                 mb_size=64,
                 lagrange_iteration_num=5,
                 add_act=64):
        self.env = env

        self.ε = dual_constraint  # hard constraint for the KL
        self.ε_μ = mean_constraint  # hard constraint for the KL
        self.ε_Σ = var_constraint  # hard constraint for the KL
        self.γ = discount_factor
        self.α = alpha  # scaling factor for the update step of η_μ

        self.sample_episode_num = sample_episode_num
        self.sample_episode_maxlen = sample_episode_maxlen
        self.rerun_num = rerun_num
        self.mb_size = mb_size
        self.lagrange_iteration_num = lagrange_iteration_num
        self.M = add_act
        self.action_shape = env.action_space.shape[0]
        self.action_range = torch.from_numpy(env.action_space.high)

        self.actor = Actor(env)
        self.critic = Critic(env)
        self.target_actor = Actor(env)
        self.target_critic = Critic(env)

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
        self.η_μ = np.random.rand()
        self.η_Σ = np.random.rand()

        # control/log variables
        self.iteration = 0

        self.replaybuffer = ReplayBuffer()

    def __sample_trajectory(self, sample_episode_num, sample_episode_maxlen, render):
        self.replaybuffer.clear()
        for i in range(sample_episode_num):
            state = self.env.reset()
            for steps in range(sample_episode_maxlen):
                action = self.target_actor.action(
                    torch.from_numpy(state).type(torch.float32)
                ).numpy().flatten()
                next_state, reward, done, _ = self.env.step(action)
                if render and i == 0:
                    self.env.render()
                self.replaybuffer.store(state, action, next_state, reward)
                if done:
                    break
                else:
                    state = next_state
            self.replaybuffer.done_episode()

    def __critic_update_td(self, states, actions, next_states, rewards):
        """
        Updates the critics
        :param states: ([State]) mini-batch of states
        :param actions: ([Action]) mini-batch of actions
        :param rewards: ([Reward]) mini-batch of rewards
        :param mean_next_q: ([State]) target Q values
        :return: (float) q-loss
        """
        next_target_μ, next_target_A = self.target_actor.forward(next_states)
        next_target_μ.detach()
        next_target_A.detach()
        next_action_distribution = MultivariateNormal(next_target_μ, scale_tril=next_target_A)
        additional_target_next_q = []
        for i in range(self.M):
            next_action = next_action_distribution.sample()
            additional_target_next_q.append(
                self.target_critic.forward(
                    next_states, next_action).detach())
        additional_target_next_q = torch.stack(additional_target_next_q).squeeze()  # (M, B)
        y = rewards + self.γ * torch.mean(additional_target_next_q, dim=0)
        self.critic_optimizer.zero_grad()
        target = self.critic(states, actions)
        loss_critic = self.mse_loss(y, target.squeeze())
        loss_critic.backward()
        self.critic_optimizer.step()
        return loss_critic.item()

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

    def train(self, iteration_num=100, save_path='model.pt', log=False, log_dir=None, render=False):
        """
        train a model based on MPO
        :param iteration_num: (int)
        :param log: (bool)
        :param log_dir:
        :param save_path
        """

        if log:
            writer = SummaryWriter(log_dir)
        else:
            writer = None

        for it in range(self.iteration, iteration_num):
            self.__sample_trajectory(
                self.sample_episode_num, self.sample_episode_maxlen, render)
            buff_sz = len(self.replaybuffer)

            mean_reward = self.replaybuffer.mean_reward()
            mean_q_loss = []
            mean_lagrange = []

            # Find better policy by gradient descent
            for _ in range(self.rerun_num):
                for indices in tqdm(BatchSampler(SubsetRandomSampler(range(buff_sz)), self.mb_size, False)):
                    B = len(indices)

                    state_batch, action_batch, next_state_batch, reward_batch = zip(
                        *[self.replaybuffer[index] for index in indices])

                    state_batch = torch.from_numpy(np.stack(state_batch)).type(torch.float32)
                    action_batch = torch.from_numpy(np.stack(action_batch)).type(torch.float32)
                    next_state_batch = torch.from_numpy(np.stack(next_state_batch)).type(torch.float32)
                    reward_batch = torch.from_numpy(np.stack(reward_batch)).type(torch.float32)

                    # sample M additional action for each state
                    target_μ, target_A = self.target_actor.forward(state_batch)
                    target_μ.detach()
                    target_A.detach()
                    action_distribution = MultivariateNormal(target_μ, scale_tril=target_A)
                    additional_action = []
                    additional_target_q = []
                    for i in range(self.M):
                        action = action_distribution.sample()
                        additional_action.append(action)
                        additional_target_q.append(
                            self.target_critic.forward(
                                state_batch, action).detach().numpy())
                    additional_action = torch.stack(additional_action).squeeze()  # (M, B)
                    additional_target_q = np.array(additional_target_q).squeeze()  # (M, B)

                    # Update Q-function
                    q_loss = self.__critic_update_td(
                        states=state_batch,
                        actions=action_batch,
                        next_states=next_state_batch,
                        rewards=reward_batch
                    )
                    mean_q_loss.append(q_loss)

                    # E-step
                    # Update Dual-function
                    def dual(η):
                        """
                        Dual function of the non-parametric variational
                        g(η) = η*ε + η \sum \log (\sum \exp(Q(a, s)/η))
                        """
                        max_q = np.max(additional_target_q, 0)
                        return η * self.ε + np.mean(max_q) \
                            + η * np.mean(np.log(np.mean(np.exp((additional_target_q - max_q) / η), 0)))

                    bounds = [(1e-6, None)]
                    res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
                    self.η = res.x[0]

                    # calculate the new q values
                    qij = torch.softmax(torch.tensor(additional_target_q) / self.η, dim=0)

                    # M-step
                    # update policy based on lagrangian
                    for _ in range(self.lagrange_iteration_num):
                        μ, A = self.actor.forward(state_batch)
                        π = MultivariateNormal(μ, scale_tril=A)  # (B,)
                        additional_logprob = qij * π.expand((self.M, B)).log_prob(additional_action[..., None])

                        C_μ, C_Σ = gaussian_kl(
                            mean1=μ, mean2=target_μ,
                            cholesky1=A, cholesky2=target_A)

                        # Update lagrange multipliers by gradient descent
                        self.η_μ -= self.α * (self.ε_μ - C_μ).detach().item()
                        self.η_Σ -= self.α * (self.ε_Σ - C_Σ).detach().item()

                        if self.η_μ < 0:
                            self.η_μ = 0
                        if self.η_Σ < 0:
                            self.η_Σ = 0

                        self.actor_optimizer.zero_grad()
                        loss_policy = -(
                                torch.mean(additional_logprob)
                                + self.η_μ * (self.ε_μ - C_μ)
                                + self.η_Σ * (self.ε_Σ - C_Σ)
                        )
                        mean_lagrange.append(loss_policy.item())
                        loss_policy.backward()
                        self.actor_optimizer.step()

            self._update_param()

            mean_q_loss = np.mean(mean_q_loss)
            mean_lagrange = np.mean(mean_lagrange)

            print(
                "\n Iteration:\t", it + 1,
                "\n Mean reward:\t", mean_reward,
                "\n Mean Q loss:\t", mean_q_loss,
                "\n Mean Lagrange:\t", mean_lagrange,
                "\n η:\t", self.η,
                "\n η_μ:\t", self.η_μ,
                "\n η_Σ:\t", self.η_Σ,
            )

            # saving and logging
            self.save_model(save_path)
            if writer is not None:
                writer.add_scalar('reward', mean_reward, it + 1)
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
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict()
        }
        torch.save(data, path)
