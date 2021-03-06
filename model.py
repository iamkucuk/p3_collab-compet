import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):

    def __init__(self, state_size, action_size, fc_units=None):
        """
        Actor model for the actor part of the agent
        :param state_size: Input size of the model. Probably same as the environment state size
        :param action_size: Output size of the model. Probably same as the environment action size
        :param fc_units: (List with length of 2) Determines the number of neurons in each hidden layer
        """
        super(Actor, self).__init__()
        if fc_units is None:
            fc_units = [512, 256]
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.fc3 = nn.Linear(fc_units[1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):

    def __init__(self, state_size, action_size, fc_units=None):
        """
        Critic model for the critic part of the agent
        :param state_size: Input size of the model. Probably same as the environment state size
        :param action_size: Should be at the same size as the Actor model.
        :param fc_units: (List with length of 2) Determines the number of neurons in each hidden layer
        """
        super(Critic, self).__init__()
        if fc_units is None:
            fc_units = [512, 256]
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0] + action_size, fc_units[1])
        self.fc3 = nn.Linear(fc_units[1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = F.leaky_relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)
