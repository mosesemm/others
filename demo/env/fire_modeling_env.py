import numpy as np
import random
import gym
from gym import spaces

from visualize import render_current
from commons import NORTH, SOUTH, EAST, WEST


#render current state
#render_current(data_set, (random.randint(0,9), random.randint(0,9)),WIDTH, HEIGHT, visited=[(0,0), (0,1), (0,3)])

#my environment stuff
MAX_VALUES = 4000


class FireModelingEnv(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, map, store_images=False, attempts=10):
        super(FireModelingEnv, self).__init__()
        self.map = map
        self.max_ = int(np.sqrt(len(self.map)))

        self.rewards = []
        self.visited_states = set()

        # to record rendered frames to animate later
        self.rendered_frames = []

        _, dim = np.shape(map)

        self.num_attempts = attempts

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(low=0, high=MAX_VALUES, shape=(dim,), dtype=np.float16)


    def reset(self):

        self.visited_states = set()
        self.rewards = []
        #make sure start point is where fire can occur
        applicable_points = [curr for curr in self.map if curr[2] > 0.4]
        self.current_step = applicable_points[random.randint(0, len(applicable_points) - 1)]
        self.visited_states.add((self.current_step[0], self.current_step[1]))

        return self.current_step

    # should be using both ndvi and lst to calculate reward
    # but now just using ndvi
    def calculate_reward(self, ndvi, lst, newState):

        reward = -1
        # you cant move into a state where ndvi is less than 0.4
        if ndvi < 0.4:
            newState = self.current_step
        else:
            reward = 1 if (self.current_step[0], self.current_step[1]) not in self.visited_states else -1

        return reward, newState

    def step(self, action):

        # get current reading
        x,y,ndvi,lst,_ = self.current_step
        reward = -1
        newState = self.current_step


        if NORTH == action:
            m_y = y-1
            if m_y > 0:
                #move into new state
                newState = [curr for curr in self.map if (curr[0], curr[1]) == (x,m_y)][0]

                #get rewarded for it
                x_n, y_n, ndvi_n, lst_n, _ = newState
                reward, newState = self.calculate_reward(ndvi_n, lst_n, newState)

        elif SOUTH == action:
            m_y = y + 1
            if m_y < self.max_ - 1:
                newState = [curr for curr in self.map if (curr[0], curr[1]) == (x, m_y)][0]

                x_n, y_n, ndvi_n, lst_n, _ = newState
                reward, newState = self.calculate_reward(ndvi_n, lst_n, newState)

        elif WEST == action:
            m_x = x -1
            if m_x > 0:
                newState = [curr for curr in self.map if (curr[0], curr[1]) == (m_x, y)][0]

                x_n, y_n, ndvi_n, lst_n, _ = newState
                reward, newState = self.calculate_reward(ndvi_n, lst_n, newState)

        elif EAST == action:
            m_x = x + 1
            if m_x < self.max_ -1:
                newState = [curr for curr in self.map if (curr[0], curr[1]) == (m_x, y)][0]

                x_n, y_n, ndvi_n, lst_n, _ = newState

                reward, newState = self.calculate_reward(ndvi_n, lst_n, newState)


        self.rewards.append(reward)
        self.current_step = newState
        self.visited_states.add((self.current_step[0], self.current_step[1]))

        # if not rewarded in the last 4 moves, maybe no longer possible to expand
        done = 1 not in self.rewards[-(self.num_attempts-1):] if len(self.rewards) > self.num_attempts else False

        return newState,reward,done, {}


    def render(self, mode='human', close=False):
        return render_current(self.map, (self.current_step[0], self.current_step[1]), self.max_, self.max_,
                       visited=self.visited_states)

