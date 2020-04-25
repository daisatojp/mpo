import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal


class Actor(nn.Module):
    """
    Policy network
    :param env: (gym Environment) environment actor is operating on
    :param layer1: (int) size of the first hidden layer (default = 100)
    :param layer2: (int) size of the first hidden layer (default = 100)
    """
    def __init__(self, env):
        super(Actor, self).__init__()
        self.env = env
        self.ds = env.observation_space.shape[0]
        self.da = env.action_space.shape[0]
        self.lin1 = nn.Linear(self.ds, 128)
        self.lin2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, self.da)
        self.cholesky_layer = nn.Linear(128, (self.da * (self.da + 1)) // 2)

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

    def eval_step(self, state):
        """
        approximates an action based on the mean output of the network
        :param state: (State) a state of  the environment
        :return: (float) an action of the action space
        """
        with torch.no_grad():
            action, _ = self.forward(state)
        return action

    def to_cholesky_matrix(self, cholesky_vector):
        """
        computes cholesky matrix corresponding to a vector
        :param cholesky_vector: ([float]) vector with n items
        :return: ([[float]]) Square Matrix containing the entries of the
                 vector
        """
        k = 0
        cholesky = torch.zeros(self.action_shape, self.action_shape)
        for i in range(self.action_shape):
            for j in range(self.action_shape):
                if i >= j:
                    cholesky[i][j] = cholesky_vector.item() if self.action_shape == 1 else cholesky_vector[k].item()
                    k = k + 1
        return cholesky
