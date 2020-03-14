import numpy as np
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from fire_modeling_env import FireModelingEnv
from dataset_generator import data_set

from commons import argmax
from evaluate import evaluate_model_free


def hash_state(array):
    return "{}_{}".format(array[0], array[1])

#tried epsilon greedy, dont seem to work nice
def make_epsilon_greedy_policy(Q, epsilon, nA, observation):
    policy_fn = np.ones(nA) * epsilon / nA
    max_val = argmax(Q[hash_state(observation)])

    policy_fn[max_val] += (1 - epsilon)

    return policy_fn

def make_greedy_policy(Q, nA, observation):
    policy_fn = np.zeros(nA)
    max_val = argmax(Q[hash_state(observation)])

    policy_fn[max_val] += 1

    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, epsilon=0.05, alpha=0.5, print_in_train=False):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    rewards = []

    for k in range(num_episodes):

        observation = env.reset()

        if k % 200 == 0 and print_in_train:
            env.render()

        while True:
            policy = make_greedy_policy(Q, nA, observation)
            action = np.random.choice(np.arange(nA), p=policy)

            next_state, reward, done, _ = env.step(action)

            best_action = argmax(Q[hash_state(next_state)])

            td = reward + discount_factor * Q[hash_state(next_state)][best_action]
            td_err = td - Q[hash_state(observation)][action]
            Q[hash_state(observation)][action] = Q[hash_state(observation)][action] + alpha * td_err
            rewards.append(reward)

            if done:
                if k % 200 == 0 and print_in_train:
                    env.render()
                break

            observation = next_state

    return Q, rewards



num_episodes = 1000
discount_factor = 1.0
epsilon = 0.1
alpha = 0.5

env = FireModelingEnv(data_set, attempts=20)


print('TD learning \n')
q_values, rewards = q_learning(env, num_episodes=num_episodes,
                                           discount_factor=discount_factor,
                                           epsilon=epsilon,
                                           alpha=alpha)
print("done...", len(rewards))
print(q_values)

evaluate_model_free(q_values)

