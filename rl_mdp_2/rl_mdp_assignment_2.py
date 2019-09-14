#!/usr/bin/env python
'''
READ ME

Student#: 1313437

prepare environment
pip install -r requirements.txt

to run:

./rl_mdp_assignment_2.py
'''

import numpy as np
import matplotlib.pyplot as plt
import sys

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

SIZE = 7

def is_start(state):
    return (0,0) == state

def is_goal(state):
    x, y = state
    return (x == 0 and y == 6)

def is_obstacle(state):
        x, y = state
        return x in [0,1,2,3,4,5] and y == 4


def get_possible_action(optimal_v, x,y):
    action = np.zeros(4)
    action[0] = optimal_v[y+1, x] if y != 6 else 0 #up
    action[1] = optimal_v[y-1, x] if y != 0 else 0 #down
    action[2] = optimal_v[y, x+1] if x != 6 else 0 #right
    action[3] = optimal_v[y, x-1] if x != 0 else 0#left

    return np.argmax(action)

def displayTrajectoryTakenByAgent(agent_tragectory):
    outfile = sys.stdout
    grid = np.zeros((7,7))
    it = np.nditer(grid, flags=['multi_index'])

    while not it.finished:
        s = it.iterindex
        y, x = it.multi_index

        output = " * "
        if is_start((x,y)):
            output = " S "
        elif is_goal((x,y)):
            output = " G "
        elif is_obstacle((x,y)):
            output = u" \u2205 "
        else:
            if (x,y) in agent_tragectory:
                output = u' \u21A8 '

        if x == 0:
            output = output.lstrip()
        if x == 6:
            output = output.rstrip()

        outfile.write(output)

        if x == 6:
            outfile.write("\n")


        it.iternext()

def take_action(state, action):
    if is_goal(state):
        #already added the reward just before the goal (-1)
        return state, 20+1

    if is_obstacle(state):
        return state, -1

    x,y = state

    if action == UP:
        y = y + 1 if y < 6 else 6
    if action == LEFT:
        x = x - 1 if x > 0 else 0
    if action == RIGHT:
        x = x + 1 if x < 6 else 6
    if action == DOWN:
        y = y - 1 if y > 0 else 0

    reward = -1
    next_state = (x,y)
    return next_state, reward

def get_tragectory(optimal_value, random = False):

    x = 0
    y = 0
    counter = 0
    MAX_ITERATIONS = 50-1

    agent_tragectory = []
    agent_reward = []

    #terminate when get ultimate reward
    reward = -1

    while reward is not 21 and counter <= MAX_ITERATIONS:

        if random:
            direction = np.random.randint(0,4)
        else:
            direction = get_possible_action(optimal_value, x,y)

        next_state, reward = take_action((x,y), direction)
        agent_reward.append(reward)

        agent_tragectory.append(next_state)
        x, y = next_state

        counter +=1

    return agent_reward, agent_tragectory



#calculated manually
optimal_value = np.array(([2,3,4,5,6,7,8],
                            [3,4,5,6,7,8,9],
                            [4,5,6,7,8,9,10],
                            [5,6,7,8,9,10,11],
                            [0,0,0,0,0,0,12],
                            [19,18,17,16,15,14,13],
                            [20,19,18,17,16,15,14]))
optimal_agent_reward, optimal_agent_tragectory = get_tragectory(optimal_value, random=False)
random_agent_reward, random_agent_tragectory = get_tragectory(optimal_value, random=True)

print("\n\nOptimal agent tragectory:\n ")
displayTrajectoryTakenByAgent(optimal_agent_tragectory)
print("\n\nRandom agent tragectory:\n ")
displayTrajectoryTakenByAgent(random_agent_tragectory)


#averaged over 20 runs
print(sum(optimal_agent_reward)/20)
print(sum(random_agent_reward)/20)

agents = ["Optimal", "Random"]
y_pos = np.arange(len(agents))
rewards = [sum(optimal_agent_reward)/20, sum(random_agent_reward)/20]

plt.bar(1, rewards[0], 1, label="Optimal")
plt.bar(2, rewards[1], 1, label="Random")
plt.xticks(y_pos, agents)
plt.ylabel("Rewards")
plt.title("Agents rewards")
plt.legend()
plt.savefig("agents_rewards.png")