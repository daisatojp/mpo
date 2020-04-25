import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


class ActorContinuous(nn.Module):
    """
    Policy network
    :param env: OpenAI gym environment
    """
    def __init__(self, env):
        super(ActorContinuous, self).__init__()
        self.env = env
        self.ds = env.observation_space.shape[0]
        self.da = env.action_space.shape[0]
        self.lin1 = nn.Linear(self.ds, 256)
        self.lin2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, self.da)
        self.cholesky_layer = nn.Linear(256, (self.da * (self.da + 1)) // 2)

    def forward(self, state):
        """
        forwards input through the network
        :param state: (B, ds)
        :return: mean vector (B, da) and cholesky factorization of covariance matrix (B, da, da)
        """
        device = state.device
        B = state.size(0)
        ds = self.ds
        da = self.da
        action_low = torch.from_numpy(self.env.action_space.low)[None, ...].to(device)  # (1, da)
        action_high = torch.from_numpy(self.env.action_space.high)[None, ...].to(device)  # (1, da)
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        mean = torch.sigmoid(self.mean_layer(x))  # (B, da)
        mean = action_low + (action_high - action_low) * mean
        cholesky_vector = self.cholesky_layer(x)  # (B, (da*(da+1))//2)
        cholesky_diag_index = torch.arange(da, dtype=torch.long) + 1
        cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
        cholesky_vector[:, cholesky_diag_index] = F.softplus(cholesky_vector[:, cholesky_diag_index])
        tril_indices = torch.tril_indices(row=da, col=da, offset=0)
        cholesky = torch.zeros(size=(B, da, da), dtype=torch.float32).to(device)
        cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
        return mean, cholesky

    def action(self, state):
        """
        :param state: (ds,)
        :return: an action
        """
        with torch.no_grad():
            mean, cholesky = self.forward(state[None, ...])
            action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
            action = action_distribution.sample()
        return action[0]


class ActorDiscrete(nn.Module):
    """
    :param env: gym environment
    """
    def __init__(self, env):
        super(ActorDiscrete, self).__init__()
        self.env = env
        self.ds = env.observation_space.shape[0]
        self.da = env.action_space.n
        self.lin1 = nn.Linear(self.ds, 256)
        self.lin2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, self.da)

    def forward(self, state):
        """
        :param state: (B, ds)
        :return:
        """
        B = state.size(0)
        h = F.relu(self.lin1(state))
        h = F.relu(self.lin2(h))
        h = self.out(h)
        return torch.softmax(h, dim=-1)

    def action(self, state):
        """
        :param state: (ds,)
        :return: an action
        """
        with torch.no_grad():
            p = self.forward(state[None, ...])
            action_distribution = Categorical(probs=p[0])
            action = action_distribution.sample()
        return action
