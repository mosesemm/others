import gym
import numpy as np

env = gym.make('FrozenLake-v0')

states = env.observation_space.n

actions = env.action_space.n

print("Just to confirm the values: {} - {} ".format(states, actions))

value_table = np.zeros((states, 1))

def value_iterations(env, n_iterations, gamma = 1.0, threshold = 1e-30):

    for i in range(n_iterations):
        new_valuetable = np.copy(value_table)

        for state in range(states):
            q_value = []
            for action in range(actions):
                next_state_reward = []
                for next_state_parameters in env.env.P[state][action]:
                    transition_prob, next_state, reward_prob, _ = next_state_parameters
                    reward = transition_prob*(reward_prob+gamma*new_valuetable[next_state])
                    next_state_reward.append(reward)


                q_value.append((np.sum(next_state_reward)))
            value_table[state] = max(q_value)

        if np.sum(np.fabs(new_valuetable - value_table)) <= threshold :
            break

    return value_table


def extract_policy(value_table, gamma = 1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_str in env.env.P[state][action]:
                transition_prob, next_state, reward_prob, _ = next_str
                Q_table[action] += (transition_prob*(reward_prob+gamma*value_table[next_state]))
        policy[state] = np.argmax(Q_table)


    return policy

value_table = value_iterations(env, 10000)
policy = extract_policy(value_table)

print(value_table)
print("*****************")
print(policy)