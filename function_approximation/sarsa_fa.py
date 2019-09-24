import gym
from value_function import ValueFunction
import numpy as np
import itertools
import matplotlib.pyplot as plt



def sarsa_fa(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1):

    episode_rewards = np.zeros(num_episodes)
    episode_lengths = np.zeros(num_episodes)

    for i_episode in range(num_episodes):
        current_state = env.reset()

        action = estimator.act(current_state, epsilon)

        for t in itertools.count():
            next_state, reward, done, _ = env.step(action)

            #choose next action
            next_action = estimator.act(next_state, epsilon)

            #Evaluate Q
            td_target = reward + discount_factor * estimator(next_state,next_action)
            #delta = td_target - estimator(current_state, action)

            #improve policy
            estimator.update(td_target, current_state, action)

            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            if done:
                break
            else:
                current_state = next_state
                action = next_action

    return episode_lengths, episode_rewards


def plotNoStepsPerEpisode(episode_lengths):
    a = [pow(10, i) for i in episode_lengths]

    plt.plot(a, color='blue', lw=2)
    plt.yscale('log')
    plt.savefig("steps_per_episode.png")

def recordActionsOffQValues(env, estimator):
    env = gym.wrappers.Monitor(env, "recording", force=True)
    state = env.reset()
    for i in range(1000):
        env.render()
        action = estimator.act(state)
        state, reward, done, _ = env.step(action)
        if done:
            print("done in {} steps...".format(i))
            break

    env.close()

env = gym.make("MountainCar-v0")
alpha = 0.9

estimator = ValueFunction(alpha, env.action_space.n)

average_returns = np.zeros(500)
for i in np.arange(100):
    episode_lengths, episode_rewards = sarsa_fa(env, estimator, num_episodes=500)
    average_returns += episode_lengths

#1
plotNoStepsPerEpisode(average_returns/100)
#2
recordActionsOffQValues(env, estimator)


