import numpy as np


class CliffState:
    def __init__(self, line, column):
        self.line = line
        self.column = column

    # Return the moved state as a new state instead change current state directly
    def move(self, action):
        if action == 0:  # Move upward
            return CliffState(self.line + 1, self.column)
        elif action == 1:  # Move rightward
            return CliffState(self.line, self.column + 1)
        elif action == 2:  # Move downward
            return CliffState(self.line - 1, self.column)
        elif action == 3:  # Move leftward
            return CliffState(self.line, self.column - 1)
        else:
            print("Unknown action %d" % action)
            exit(1)


# Action-Value function
# Index {0, 1, 2, 3}  represents {^, >, v, <}
# It provides greedy and e-greedy policy on each state.
class CliffActionValue:
    def __init__(self, width=12, height=4, epsilon=1e-1):
        self.width = width
        self.height = height
        self.is_arbitrary = True  # Policy is arbitrary when initialized
        self.action_values = np.zeros([height, width, 4], dtype=float)
        self.epsilon = epsilon

    # Return the action at <state> with greedy policy
    def greedy(self, state: CliffState):
        if np.sum(self.action_values[state.line, state.column, :]) == 0:
            return np.random.randint(0, 4)
        return np.argmax(self.action_values[state.line, state.column, :])

    # Return the action at <state> with e-greedy policy
    def e_greedy(self, state: CliffState):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 4)
        else:
            return self.greedy(state)

    # Return Q(state, action)
    def get(self, state: CliffState, action):
        return self.action_values[state.line, state.column, action]

    def set(self, state: CliffState, action, value):
        self.action_values[state.line, state.column, action] = value

    def print_policy(self):
        symbol_map = {
            0: '^',
            1: '>',
            2: 'v',
            3: '<',
        }
        # output = np.chararray([self.height, self.width])
        for x in range(0, self.height):
            for y in range(0, self.width):
                # output[self.height - x - 1, y] = symbol_map[np.argmax(self.action_values[x, y, :])]
                print(symbol_map[np.argmax(self.action_values[self.height - x - 1, y, :])], end=' ')
            print()
        # print(output)

class CliffWalking:
    def __init__(self, width=12, height=4, epsilon=1e-1, gamma=1, alpha=1):
        self.width = width
        self.height = height
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

    # Input: Current state and action
    # Output: New state and reward
    def move(self, state: CliffState, action):
        new_state = state.move(action)
        if new_state.line < 0:
            new_state.line = 0
        if new_state.line >= self.height:
            new_state.line = self.height - 1
        if new_state.column < 0:
            new_state.column = 0
        if new_state.column >= self.width:
            new_state.column = self.width - 1

        # Reach The Cliff
        if new_state.line == 0 and new_state.column != self.width - 1:
            new_state.column = 0
            return new_state, -100

        return new_state, -1

    def is_terminate(self, state: CliffState):
        return state.line == 0 and state.column == self.width - 1

    def q_learning_sample(self):
        # Initialize State at the start point
        current_state = CliffState(0, 0)
        total_rewards = 0
        while not self.is_terminate(current_state):
            action = self.action_value.e_greedy(current_state)
            new_state, reward = self.move(current_state, action)
            greedy_action = self.action_value.greedy(new_state)
            new_value = self.action_value.get(current_state, action) \
                        + self.alpha * (reward
                                        + self.gamma * self.action_value.get(new_state, greedy_action)
                                        - self.action_value.get(current_state, action))
            self.action_value.set(current_state, action, new_value)
            current_state = new_state
            total_rewards += reward
        return total_rewards

    # Return two list: episode number and rewards
    def q_learning(self, episodes: int = 100):
        self.action_value = CliffActionValue(self.width, self.height, self.epsilon)
        episode_num = np.arange(episodes)
        rewards = np.zeros([episodes, ], dtype=int)
        for episode in range(episodes):
            rewards[episode] = self.q_learning_sample()
        return episode_num, rewards

    def sarsa_sample(self):
        # Initialize State at the start point
        current_state = CliffState(0, 0)
        total_rewards = 0
        while not self.is_terminate(current_state):
            action = self.action_value.e_greedy(current_state)
            new_state, reward = self.move(current_state, action)
            greedy_action = self.action_value.e_greedy(new_state)
            new_value = self.action_value.get(current_state, action) \
                        + self.alpha * (reward
                                        + self.gamma * self.action_value.get(new_state, greedy_action)
                                        - self.action_value.get(current_state, action))
            self.action_value.set(current_state, action, new_value)
            current_state = new_state
            total_rewards += reward
        return total_rewards

    def sarsa(self, episodes: int = 100):
        self.action_value = CliffActionValue(self.width, self.height, self.epsilon)
        episode_num = np.arange(episodes)
        rewards = np.zeros([episodes, ], dtype=int)
        for episode in range(episodes):
            rewards[episode] = self.sarsa_sample()
        return episode_num, rewards


