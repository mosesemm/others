import gym
import numpy as np

env = gym.make('FrozenLake-v0')

states = env.observation_space.n
actions = env.action_space.n

def compute_value(env, policy, gamma=1.0, threshold=1e-20):
    value_table =  np.zeros((states, 1))

    while True:
        new_valuetable = np.copy(value_table)
        for state in range(states):
            action = int(policy[state])
            for next_state_parameters in env.env.P[state][action]:
                transition_prob, next_state, reward_prob, _ = next_state_parameters
                value_table[state] = transition_prob*(reward_prob+gamma*new_valuetable[next_state])

        if np.sum(np.fabs(new_valuetable - value_table)) <= threshold:
            break

    return value_table


def extract_policy(value_table, gamma = 1.0):
    policy = np.zeros(states)
    for state in range(states):
        Q_table = np.zeros(actions)
        for action in range(actions):
            for next_sr in env.env.P[state][action]:
                transition_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (transition_prob*(reward_prob+gamma*value_table[next_state]))

        policy[state] = np.argmax(Q_table)

    return policy

n_iterations = 200000
random_policy =  np.zeros((states, 1))
for i in range(n_iterations):
    new_value_table = compute_value(env, random_policy)
    new_policy = extract_policy(new_value_table)
    random_policy = new_policy

print(random_policy)