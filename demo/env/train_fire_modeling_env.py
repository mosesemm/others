import gym
import json
import datetime as dt
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DQN, A2C
from stable_baselines.deepq.policies import DQNPolicy

from fire_modeling_env import FireModelingEnv
from dataset_generator import data_set, WIDTH, HEIGHT
from analyse import display_scatter_plot, calculate_correlation
from commons import MODEL_PATH

ndvi_feature = [item[2] for item in data_set]
lst_feature = [item[3] for item in data_set]

#little bit of analyses on the features
display_scatter_plot(ndvi_feature, lst_feature)
print("Covariance between the two features: {}".format(calculate_correlation(ndvi_feature, lst_feature)))

#env = FireModelingEnv(data_set)

env = DummyVecEnv([lambda : FireModelingEnv(data_set)])

model = PPO2(MlpPolicy, env, verbose=1)
#model = DQN(DQNPolicy, env, verbose=1)
#model = A2C(MlpPolicy, env, verbose=1)

model.learn(total_timesteps=20000)
model.save(MODEL_PATH)

'''
obs = env.reset()
env.render()
for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

'''