import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.distributions import Categorical


class Actor(nn.Module):
    """
    :param env: gym environment
    :param layer1: hidden size of layer1
    :param layer2: hidden size of layer2
    """
    def __init__(self, env, layer1=100, layer2=100):
        super(Actor, self).__init__()
        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.n
        self.lin1 = nn.Linear(self.state_shape, layer1)
        self.lin2 = nn.Linear(layer1, layer2)
        self.out = nn.Linear(layer2, self.action_shape)

    def forward(self, states):
        """
        :param states: (B, ds)
        :return:
        """
        B = states.size(0)

        h = F.relu(self.lin1(states))
        h = F.relu(self.lin2(h))
        h = self.out(h)
        return torch.softmax(h, dim=-1)

    def action(self, state):
        """
        approximates an action by going forward through the network
        :param state: (State) a state of the environment
        :return: (float) an action of the action space
        """
        with torch.no_grad():
            p = self.forward(state[None, ...])
            action_distribution = Categorical(probs=p[0])
            action = action_distribution.sample()
        return action

    def eval_step(self, state):
        """
        approximates an action based on the mean output of the network
        :param state: (State) a state of  the environment
        :return: (float) an action of the action space
        """
        with torch.no_grad():
            action, _ = self.forward(state)
        return action
