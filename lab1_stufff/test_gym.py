
from gym import wrappers, envs
import matplotlib.pyplot as plt
import gym
from gym import envs
import numpy as np

import os

from  environments.gridworld import GridworldEnv


env = gym.make("MsPacmanNoFrameskip-v4")
plt.imshow(env.render('rgb_array'))
plt.grid(False)
plt.show()

print("Observation space: ", env.observation_space)
print("Action space: ", env.action_space)