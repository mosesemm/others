from gym import spaces
import torch.nn as nn

import torch.nn.functional as F


# Class inheritance example with Pytorch, can use Tensorflow instead.
class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert type(observation_space) == spaces.Box, 'observation_space must be of type Box'
        assert len(observation_space.shape) == 3, 'observation space must have the form channels x width x height'
        assert type(action_space) == spaces.Discrete, 'action_space must be of type Discrete'

        # TODO Implement CNN layers
        # dont really know this yet
        self.fc1 = nn.Linear(84, 84)
        self.fc2 = nn.Linear(84, 84)
        self.fc3 = nn.Linear(84, action_space.n)

    def forward(self, x):
        # TODO Implement forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
