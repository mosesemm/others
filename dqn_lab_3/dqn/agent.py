from gym import spaces
import numpy as np

from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

#device = "cuda"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete,
                 replay_buffer: ReplayBuffer,
                 use_double_dqn,
                 lr,
                 batch_size,
                 gamma):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        self.memory = replay_buffer
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.gamma = gamma
        self.policy_network = DQN(observation_space, action_space).to(device)
        self.target_network = DQN(observation_space, action_space).to(device)
        self.update_target_network()
        self.target_network.eval()
        self.optimiser = optim.Adam(self.policy_network.parameters(), lr=lr) # TODO Initialise Pytorch/Tensorflow optimiser with learning rate and policy_network.parameters()

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        # TODO
        #   Optimise the TD-error over a single minibatch of transitions
        #   Sample the minibatch from the replay-memory
        #   using done (as a float) instead of if statement
        #   return loss

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        Q_targets_next = self.target_network(next_states).detatch().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1-dones))

        Q_expected = self.policy_network(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss


    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        # TODO update target_network parameters with policy_network parameters
        #??
        for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            #soft updates, maybe later
            target_param.data.copy_(policy_param.data)

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        # TODO Select action greedily from the Q-network given the state
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state)
        self.policy_network.train()

        return np.argmax(action_values.cpu().data.numpy())
