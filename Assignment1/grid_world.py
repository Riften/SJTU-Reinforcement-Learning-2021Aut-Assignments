# The problem environment for Small Grid World Problem
import math

import numpy as np


class GridWorld:
    def __init__(self, width=4, height=4, accuracy=1e-4):
        self.width = width
        self.height = height
        self.accuracy = accuracy

        # State values for different states
        # The state values are initialized as zeros
        # tmp_state_values serves as the intermediate variable when update state values synchronous backup updating
        self.state_values = np.zeros([width, height], dtype=float)
        self.tmp_state_values = np.zeros([width, height], dtype=float)

        # The policy
        # policy[x, y, {0,1,2,3}] denotes the probability of action {n, e, s, w} at state (x,y)
        self.policy = np.zeros([width, height, 4], dtype=float)
        self.policy += 0.25
        # We would move to nowhere at terminate state
        self.policy[0, 0] *= 0
        self.policy[width - 1, height - 1] *= 0
        # The movement is limited at border
        for i in range(1, width - 1):
            self.policy[i, 0, :] = [1 / 3, 1 / 3, 1 / 3, 0]
            self.policy[i, height - 1, :] = [1 / 3, 0, 1 / 3, 1 / 3]
        for i in range(1, height - 1):
            self.policy[0, i, :] = [0, 1 / 3, 1 / 3, 1 / 3]
            self.policy[width - 1, i, :] = [1 / 3, 1 / 3, 0, 1 / 3]
        self.policy[0, height - 1] = [0, 0, 1 / 2, 1 / 2]
        self.policy[width - 1, 0] = [1 / 2, 1 / 2, 0, 0]

        # Define new variable for convenient
        self.policy_north = self.policy[:, :, 0]
        self.policy_east = self.policy[:, :, 1]
        self.policy_south = self.policy[:, :, 2]
        self.policy_west = self.policy[:, :, 3]
        # print(self.policy_north)
        # print(self.policy_east)
        # print(self.policy_south)
        # print(self.policy_west)

        # print(np.sum(self.policy_north + self.policy_east + self.policy_south + self.policy_west)
        #        - width*height +2)
        # Assert that the sum of policy at each state is 1
        assert np.abs((np.sum(self.policy_north + self.policy_east + self.policy_south + self.policy_west)
                       - width * height + 2)) < 1e-3
        # The reward function
        self.step_reward = -1

    # Policy Evaluation method
    # return: the delta between V(k+1) and V(k)
    def evaluation(self):
        north_values = np.zeros([self.width, self.height], dtype=float)
        east_values = np.zeros([self.width, self.height], dtype=float)
        south_values = np.zeros([self.width, self.height], dtype=float)
        west_values = np.zeros([self.width, self.height], dtype=float)

        north_values[1:self.width, :] = self.state_values[0:self.width - 1, :]
        east_values[:, 0:self.height - 1] = self.state_values[:, 1:self.height]
        south_values[0:self.width - 1, :] = self.state_values[1:self.width, :]
        west_values[:, 1:self.height] = self.state_values[:, 0:self.height - 1]
        # print('policy north')
        # print(self.policy_north)
        # print('north values')
        # print(north_values)
        self.tmp_state_values = self.policy_north * (self.step_reward + north_values) + \
                                self.policy_south * (self.step_reward + south_values) + \
                                self.policy_east * (self.step_reward + east_values) + \
                                self.policy_west * (self.step_reward + west_values)
        delta = np.abs(np.sum(np.abs(self.tmp_state_values) - np.abs(self.state_values)))
        self.state_values = self.tmp_state_values
        return delta

    def iterative_evaluation(self):
        delta = self.accuracy + 1
        ite = 1
        while delta >= self.accuracy:
            delta = self.evaluation()
            print('\rIteration: %d, Delta: %f' % (ite, delta), end='')
            ite+=1
        print('')

    # Get the state value at (x, y)
    # Return -inf if out of range
    # This method is used to determine the argmax_pi
    def get_value(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return -math.inf
        return self.state_values[x, y]

    def improve_policy(self):
        old_policy = self.policy
        self.policy = np.zeros([self.width, self.height, 4], dtype=float)
        for x in range(0, self.width):
            for y in range(0, self.height):
                tmp_reward = np.array([self.get_value(x-1, y),
                                       self.get_value(x, y+1),
                                       self.get_value(x+1, y),
                                       self.get_value(x, y-1)])
                self.policy[x, y, np.argmax(tmp_reward)] = 1
        self.policy[0, 0] *= 0
        self.policy[self.width - 1, self.height - 1] *= 0
        self.policy_north = self.policy[:, :, 0]
        self.policy_east = self.policy[:, :, 1]
        self.policy_south = self.policy[:, :, 2]
        self.policy_west = self.policy[:, :, 3]
        return np.sum(np.abs(self.policy - old_policy))

    def print_policy(self):
        word_map = {
            0: '^',
            1: '>',
            2: 'v',
            3: '<'
        }

        output = np.chararray([self.width, self.height])
        for x in range(0, self.width):
            for y in range(0, self.height):
                output[x, y] = word_map[np.argmax(self.policy[x, y, :])]
        output[0,0] = '-'
        output[self.width-1, self.height-1] = '-'
        print(output)
