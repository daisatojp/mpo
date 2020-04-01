import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.tensor


class Critic(nn.Module):
    """
    Critic class for Q-function network
    :param env: (gym Environment) environment newtwork is operating on
    :param layer1: (int) size of the first hidden layer (default = 200)
    :param layer2: (int) size of the first hidden layer (default = 200)
    """
    def __init__(self, env, layer1=200, layer2=200):
        super(Critic, self).__init__()
        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.shape[0]
        self.lin1 = nn.Linear(self.state_shape, layer1)
        self.lin2 = nn.Linear(layer1 + self.action_shape, layer2)
        self.lin3 = nn.Linear(layer2, 1)

    def forward(self, state, action):
        """
        Forward function forwarding an input through the network
        :param state: (State) a state of the environment
        :param action: (Action) an action of the environments action-space
        :return: (float) Q-value for the given state-action pair
        """
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(torch.cat((x, action), 1)))
        x = self.lin3(x)
        return x
