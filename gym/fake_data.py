import pickle
import numpy as np
import random
from decision_transformer.envs.boyanchain_13 import Boyan13


def test_output():
    with open('/home/mitacs/github/decision-transformer/gym/data/hopper-medium-v2.pkl', 'rb') as handle:
        temp = pickle.load(handle)
        first_traj = temp[0]
        del temp
        print(type(first_traj['observations']))
        print(len(first_traj['observations']))
        print('--------------------------------------------')
        print(first_traj)
        print('--------------------------------------------')
        print(first_traj['observations'])
        print('--------------------------------------------')
        # print(type(first_traj['next_observations']))
        # print(len(first_traj['next_observations']))
        # print(type(first_traj['actions']))
        # print(len(first_traj['actions']))
        # print(type(first_traj['rewards']))
        # print(len(first_traj['rewards']))
        # print(type(first_traj['terminals']))
        # print(len(first_traj['terminals']))

def generate_trajectory(jump_prob: float):
    assert (0 <= jump_prob and jump_prob <= 1)
    env = Boyan13()
    env.reset()

    # TODO: should we include the starting state in the observations? 
    #       I assume no because other dataset has same dimension for actions & observations
    observations = []
    actions = []
    # next_obs = []
    rewards = []
    terminals = []

    while env.get_state() != 0:
        if random.random() < jump_prob:
            action = 1 # jump
        else:
            action = 0 # right

        if env.curr_state == 1: # no jump at state 1
            action = 0

        actions.append(action)
        reward, observation, terminal = env.step(action=action)
        rewards.append(reward)
        observations.append([observation])  # our observation is just state (an int), the other env's observation is a list of float
        terminals.append(terminal)
    
    res_traj = {}
    res_traj['observations'] = np.array(observations)
    res_traj['actions'] = np.array(actions)
    res_traj['rewards'] = np.array(rewards)
    res_traj['terminals'] = np.array(terminals)
    return res_traj


if __name__ == "__main__":
    test_output()

    NUM_TRAJ = 10
    JUMP_PROB = 0.7
    OUTPUT_FILE = './data/boyan13-medium-v2.pkl'

    output_trajs = []
    for _ in range(NUM_TRAJ):
        traj = generate_trajectory(jump_prob=JUMP_PROB)
        print(traj)
        # print(traj['rewards'].shape[0])
        output_trajs.append(traj)

    with open(OUTPUT_FILE, 'wb') as handle:
        pickle.dump(output_trajs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print(output_trajs == b)