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
    def __init__(self, env, layer1=100, layer2=100):
        super(Actor, self).__init__()
        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.shape[0]
        self.action_range = torch.from_numpy(env.action_space.high)
        self.lin1 = nn.Linear(self.state_shape, layer1)
        self.lin2 = nn.Linear(layer1, layer2)
        self.mean_layer = nn.Linear(layer2, self.action_shape)
        self.cholesky_layer = nn.Linear(layer2, self.action_shape)
        self.action_shape_eye = torch.eye(self.action_shape)

    def forward(self, states):
        """
        forwards input through the network
        :param states: ([State]) a (batch of) state(s) of the environment
        :return: ([float])([float]) mean and cholesky factorization chosen by policy at given state
        """
        B = states.size(0)

        device = self.lin1.weight.device
        if self.action_range.device != device:
            self.action_range = self.action_range.to(device)
        if self.action_shape_eye.device != device:
            self.action_shape_eye = self.action_shape_eye.to(device)

        x = F.relu(self.lin1(states))
        x = F.relu(self.lin2(x))
        mean = self.action_range * torch.tanh(self.mean_layer(x))
        cholesky_vector = F.softplus(self.cholesky_layer(x))
        cholesky = self.action_shape_eye[None, ...].repeat(B, 1, 1) @ cholesky_vector[..., None]
        return mean, cholesky

    def action(self, state):
        """
        approximates an action by going forward through the network
        :param state: (State) a state of the environment
        :return: (float) an action of the action space
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
