import numpy as np
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib
import matplotlib.animation as animation

import gym
from gym import envs

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    Returns:
        A function that takes the state as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(state):
        # 1 / epsilon for non-greedy actions
        probs = (epsilon / nA) * np.ones(nA)

        greedy_action = Q[state].argmax()
        # (1 / epsilon + (1 - epsilon)) for greedy action
        probs[greedy_action] += 1.0 - epsilon

        return probs

    return policy_fn

def sarsa(env, num_episodes=200, discount_factor=1.0, alpha=0.5, epsilon=0.1, decay_rate=0, type= 'accumulate'):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    E = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # Keeps track of useful statistics
    episode_rewards = np.zeros(num_episodes)
    episode_lengths = np.zeros(num_episodes)

    episodes_qs = []

    for i_episode in range(num_episodes):
        current_state = env.reset()
        # choose the action based on epsilon greedy policy
        probs = policy(current_state)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        episodes_qs.append(Q)
        # keep track number of time-step per episode only for plotting
        for t in itertools.count():
            next_state, reward, done, _ = env.step(action)

            # choose next action
            next_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_probs)), p=next_probs)
            # evaluate Q using estimated action value of (next_state, next_action)
            td_target = reward + discount_factor * Q[next_state][next_action]
            delta = td_target - Q[current_state][action]

            E[current_state][action] = +1

            for s, _ in Q.items():
                Q[s][:] += alpha * (delta)*E[s][:]
                if type == 'accumulate':
                    E[s][:] *=decay_rate*discount_factor
                elif type == 'replace':
                    if s == current_state:
                        E[s][:] = 1
                    else:
                        E[s][:] *= discount_factor*decay_rate

            # improve policy using new evaluate Q
            policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

            # Update statistics
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            if done:
                break
            else:
                current_state = next_state
                action = next_action

    return Q, episode_rewards, episode_lengths, episodes_qs


def displayErrorBar(x,y):

    fig2 = plt.figure()
    yerr = y - np.mean(y)
    plt.figure()
    plt.errorbar(x, y, yerr=yerr)
    plt.title("Average returns error bars")
    plt.savefig("errorbar.png")

env = gym.make("CliffWalking-v0")

def maxValueFunc(Q):
    return np.array([np.max(Q[key]) for key in np.arange(env.observation_space.n)])


lambda_values = [0, 0.3, 0.5]
average_returns = []

lambda_to_q_values = {}
for Lamda in lambda_values:
    Q, episode_rewards, episode_lengths, episodes_qs = sarsa(env, decay_rate=Lamda)
    lambda_to_q_values[Lamda] = episodes_qs
    average_returns.append(sum(episode_rewards)/100)


Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist="Me"), bitrate=1800)

def recordAnimation2(q_values, lambda_v):
    fig = plt.figure()
    ims = []
    for q_v in q_values:
        im = plt.imshow(maxValueFunc(q_v).reshape((4,12)), cmap=matplotlib.cm.coolwarm, animated=True)
        ims.append([im])

    ani = matplotlib.animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save("animation_for_" + str(lambda_v) + ".mp4", writer=writer)

def recordAnimation(q_values, lambda_v):

    fig = plt.figure()
    def animate(i):
        plt.imshow(maxValueFunc(q_values[i]).reshape((4,12)), cmap=matplotlib.cm.coolwarm)
        #sns.heatmap(maxValueFunc(q_values[i]).reshape((4,12)))

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(q_values), repeat=False)
    ani.save("animation_for_"+str(lambda_v)+".mp4", writer=writer)


#generate the required animations
for i in lambda_to_q_values:
    recordAnimation2(lambda_to_q_values[i], i)

#the errorbar
displayErrorBar(lambda_values, average_returns)
