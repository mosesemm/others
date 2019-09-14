#!/usr/bin/env python
'''
READ ME

Student#: 1313437

prepare environment
pip install -r requirements.txt

to run:

./lab_1.py
'''

import numpy as np
from environments.gridworld import GridworldEnv, UP, RIGHT, DOWN, LEFT
import timeit
import matplotlib.pyplot as plt
import sys

from timeit import default_timer as timer
from datetime import timedelta


def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:

        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        policy: [S, A] shaped matrix representing the policy.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """
    V = np.zeros(env.observation_space.n)

    while True:
        V_converged = True
        # iterate over state
        for s in range(env.observation_space.n):
            new_v = 0
            # iterate over action
            for a in range(env.action_space.n):
                # iterate over next state given transition probability
                for prob, next_state, reward, done in env.P[s][a]:
                    # Bellman expectation backup
                    new_v += policy[s][a] * prob * (reward + discount_factor * V[next_state])
            # if the updates are no long significant
            if np.abs(new_v - V[s]) > theta:
                V_converged = False
            # update new value
            V[s] = new_v
        # stop if V_converged
        if V_converged:
            return np.array(V)


def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_evaluation_fn: Policy Evaluation function that takes 3 arguments:
            env, policy, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        action_values = np.zeros(env.action_space.n)
        # iterate over action
        for a in range(env.action_space.n):
            # iterate over next state given transition probability
            for prob, next_state, reward, done in env.P[state][a]:
                action_values[a] += prob * (reward + discount_factor * V[next_state])

        return action_values

    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

    while True:
        # evalute how new policy performs
        V = policy_evaluation_fn(env, policy, discount_factor)
        # prepare for new policy, since new policy will be deterministic
        # we init probability for all actions as 0.0 and give only the best 1.0
        new_policy = np.zeros_like(policy)

        is_policy_optimized = True
        # iterate over state
        for s in range(env.observation_space.n):
            action_taken = policy[s].argmax()
            # calculate the value of all actions in the current state
            action_values = one_step_lookahead(s, V)
            # choose best action based on which action give us max value
            best_action = action_values.argmax()
            # if previous choosen action base on last policy does not equal to new action
            # based on max action value, then we didn't obtain optimal policy
            if action_taken != best_action:
                is_policy_optimized = False
            # update new policy no matter what
            new_policy[s][best_action] = 1.0
        if is_policy_optimized:
            return policy, V

        policy = new_policy



def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        action_values = np.zeros(env.action_space.n)
        # use one-step lookahead and update v to best action value
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][a]:
                action_values[a] += prob * (reward + discount_factor * V[next_state])

        return action_values

    V = np.zeros(env.observation_space.n)

    while True:
        V_converged = True
        # init a empty policy every time, we can choose the best policy along with
        # find optimal action value
        policy = np.zeros([env.observation_space.n, env.action_space.n])
        for s in range(env.observation_space.n):
            action_values = one_step_lookahead(s, V)
            max_v = action_values.max()

            # converged only when V reach optimal (max action value doesn't change anymore)
            if np.abs(max_v - V[s]) > theta:
                V_converged = False
            # update v and corresponding policy
            V[s] = max_v
            policy[s][action_values.argmax()] = 1.0

        if V_converged:
            return policy, V


def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)
    state = env.reset()


    print("")
    env.render()
    print("")

    # TODO: generate random policy

    random_policy = np.ones((env.observation_space.n, env.action_space.n))/env.action_space.n

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    # TODO: evaluate random policy
    v = policy_evaluation(env, random_policy)

    # TODO: print state value for each state, as grid shape

    print(v.reshape((5, 5)))

    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")
    # TODO: use  policy improvement to compute optimal policy and state values
    policy, v = policy_iteration(env)  # call policy_iteration

    # TODO Print out best action for each state in grid shape

    best_actions = [np.argmax(s) for s in policy]

    displayActions(best_actions)
    print()

    # TODO: print state value for each state, as grid shape
    print(v.reshape((5,5)))

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])

    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    # TODO: use  value iteration to compute optimal policy and state values
    policy, v = value_iteration(env)

    # TODO Print out best action for each state in grid shape

    best_actions = [np.argmax(s) for s in policy]

    displayActions(best_actions)
    print()

    # TODO: print state value for each state, as grid shape
    print(v.reshape((5,5)))

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    # required plots
    discountRatesToRunningTimesPlot(env)


def discountRatesToRunningTimesPlot(env):
    discountRates = np.logspace(-0.2, 0, num=30)
    #print(discountRates)

    policy_iteration_times = []
    value_iteration_times = []
    for i in discountRates:

        start = timer()
        _, _ = policy_iteration(env=env, discount_factor=i)
        end = timer()

        policy_iteration_times.append(timedelta(seconds=end-start).total_seconds()/10)

        start = timer()
        _, _ = value_iteration(env=env, discount_factor=i)
        end = timer()

        value_iteration_times.append(timedelta(seconds=end-start).total_seconds()/10)

    print(policy_iteration_times)
    print(value_iteration_times)

    plt.plot(discountRates, policy_iteration_times, label="Policy iteration run times (s)")
    plt.plot(discountRates, value_iteration_times, label="Value iteration run times (s)")

    plt.ylabel("iterations to convergence")
    plt.xlabel("discount rates")
    plt.legend()
    plt.savefig("discount_iterations.png")
    plt.close()

def displayActions(best_actions):
    outfile = sys.stdout
    grid = np.arange(len(best_actions)).reshape((5,5))
    it = np.nditer(grid, flags=['multi_index'])
    terminal_states = [24]
    while not it.finished:
        s = it.iterindex
        y, x = it.multi_index

        if s in terminal_states:
            output = " X "
        else:

            if best_actions[s] == UP:
                output = u' \u2191 '
            if best_actions[s] == LEFT:
                output = u' \u2190 '
            if best_actions[s] == RIGHT:
                output = u' \u2192 '
            if best_actions[s] == DOWN:
                output = u' \u2193 '

        if x == 0:
            output = output.lstrip()
        if x == 4:
            output = output.rstrip()

        outfile.write(output)

        if x == 4:
            outfile.write("\n")

        it.iternext()




if __name__ == "__main__":
    main()
