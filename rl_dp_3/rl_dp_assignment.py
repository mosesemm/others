import matplotlib
import matplotlib.pyplot as plt

import numpy as np;
import seaborn as sns; sns.set()

matplotlib.use('Agg')

class GridWorldMDP:

    SIZE = 4
    # left, up, right, down
    ACTIONS = [np.array([0, -1]),
               np.array([-1, 0]),
               np.array([0, 1]),
               np.array([1, 0])]

    def is_terminal(self, state):
        x, y = state
        return (x == 0 and y == 0)

    def take_action(self, state, action):
        if self.is_terminal(state):
            return state, 0

        next_state = (np.array(state) + action).tolist()
        x, y = next_state

        if x < 0 or x >= self.SIZE or y < 0 or y >= self.SIZE:
            next_state = state

        reward = -1
        return next_state, reward


ACTION_PROB = 0.25
THETA = 1e-2

world = GridWorldMDP()


def calculate_state_value(in_place=True, discount=1.0):
    new_state_values = np.zeros((GridWorldMDP.SIZE, GridWorldMDP.SIZE))
    iteration = 0
    while True:
        if in_place:
            state_values = new_state_values
        else:
            state_values = new_state_values.copy()
        old_state_values = state_values.copy()

        for i in range(GridWorldMDP.SIZE):
            for j in range(GridWorldMDP.SIZE):
                value = 0
                for action in GridWorldMDP.ACTIONS:
                    (next_i, next_j), reward = world.take_action([i, j], action)
                    value += ACTION_PROB * (reward + discount * state_values[next_i, next_j])
                new_state_values[i, j] = value

        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < THETA:
            break

        iteration += 1

    return new_state_values, iteration

def draw_heatmap():
    #calculate in-place
    _, asycn_iteration = calculate_state_value()
    #calculate with 2 arrays
    values, sync_iteration = calculate_state_value(in_place=False)

    print('Number of iterations for in-place option: {}'.format(asycn_iteration))
    print('Number of iterations for Synchronous option: {}'.format(sync_iteration))

    sns.heatmap(values)

    plt.savefig('./heatmap_plot.png')
    plt.close()

def discountRatestToIterationPlot():
    discountRates = np.logspace(- 0.2, 0, num=20)
    print(discountRates)

    asycn_iterations = []
    sync_iterations = []
    for i in discountRates:
        _, asycn_i = calculate_state_value(discount=i)
        asycn_iterations.append(asycn_i)
        _, sync_i = calculate_state_value(in_place=False, discount=i)
        sync_iterations.append(sync_i)

    plt.plot(discountRates, asycn_iterations, label="In-place")
    plt.plot(discountRates, sync_iterations, label="Two array")

    plt.ylabel("iterations to convergence")
    plt.xlabel("discount rates")
    plt.legend()
    plt.savefig("discount_iterations.png")
    plt.close()

if __name__ == '__main__':
    draw_heatmap()
    discountRatestToIterationPlot()

