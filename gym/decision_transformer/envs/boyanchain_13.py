""" Boyan Chain (13 states)
- https://link.springer.com/content/pdf/10.1023/A:1017936530646.pdf
"""

import numpy as np
import sys

# Constants
RIGHT:int = 0
JUMP:int = 1

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=sys.maxsize)


###############################################################
# Boyan Chain environment
###############################################################


class Boyan13():
    def __init__(self):
        self.num_states:int = 13
        self.curr_state:int = 12 # starting with the leftmost state 12

    def get_state(self) -> int:
        return self.curr_state

    def reset(self) -> None:
        self.curr_state = 12

    def step(self, action:int):
        if self.curr_state == 0:
            print("BOYAN-ENV Error: No action should be taken at state 0... Exiting now.")
            exit()

        reward = -3
        terminal = False

        if action == JUMP and self.curr_state <= 1:
            print("BOYAN-ENV Error: JUMP action is unavailable in state 1... Exiting now.")
            exit()

        if action == RIGHT:
            if self.curr_state == 1:
                reward = -2
            self.curr_state = self.curr_state - 1
        elif action == JUMP:
            self.curr_state = self.curr_state - 2

        if self.curr_state == 0:
            terminal = True

        return (reward, self.curr_state, terminal)

    def get_exact_value_function(self):
        # using Bellman equation
        #   v(s) = sum_a pi(a|s) sum_{s',r} p(s',r | s,a) [ r + gamma * v(s') ]
        #   gamma = 1
        v = [0] * (self.num_states)
        v[0] = 0    # terminal state
        v[1] = -2    # 1 * 1 [ -2 + 0 ] = -2
        # v[2] = -4   # 0.5 * [-3 + -2] + 0.5 * [-3 + 0] = -4
        for i in range(2, self.num_states):
            v[i] = 0.5 * (-3 + v[i-1]) + 0.5 * (-3 + v[i-2])
        return v


###############################################################
# Boyan Chain feature representation
###############################################################


def generate_descending_map():
    # this one works from state 12 to state 0
    NUM_FEATURES = 4
    NUM_STATES = 13
    res = np.zeros(NUM_FEATURES)
    res[0] = 1

    counter = 0
    map = []
    for i in range(NUM_STATES):
        map.append(res.tolist())

        if (res[counter] == 0):
            counter += 1
        if (counter + 1 >= NUM_FEATURES):
            break
        res[counter] -= 0.25
        res[counter+1] += 0.25

    map = np.array(map) # convert to np.array
    return map


def generate_ascending_map(dim:int):
    # this one works from state 0 to state 12
    l = [1] + (dim-1)*[0]
    arr = np.array([])
    arr = np.hstack((arr, np.array(l)))

    n = len(l)-1
    for i in range(n):
        j = i + 1
        while (l[i] != 0):
            l[i] = l[i] - 0.25
            l[j] = l[j] + 0.25
            arr = np.vstack((arr, np.array(l)))
    rev = arr[::-1]
    return rev


class BoyanRep13:
    def __init__(self, is_descending:bool=True):
        self.dim:int = 4
        if is_descending is True:
            self.map = generate_descending_map()
        else:
            self.map = generate_ascending_map(self.dim)

    def getmap(self):
        return self.map

    def encode(self, s):
        return self.map[s]

    def features(self) -> int:
        return self.dim


##  For testing
# a = Boyan13()
# print(a.get_exact_value_function())
# a = BoyanRep13()
# print('-------------------------')
# print(a.getmap())
# print('-------------------------')
# print(a.encode(1))

# End of File