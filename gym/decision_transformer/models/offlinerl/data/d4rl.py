import os
import pickle

import gym
import d4rl
import numpy as np
from loguru import logger

from offlinerl.utils.data import SampleBatch


def addRtgToState(stateArray, nextStateArray, rewardArray, gamma, scale):
    rtgArray = np.zeros_like(rewardArray)
    rtgArray[-1] = rewardArray[-1]
    for t in reversed(range(stateArray.shape[0]-1)):
        rtgArray[t] = rewardArray[t] + gamma * rtgArray[t+1]
    # rtgArray/=scale
    # print(stateArray.shape)
    # print(nextStateArray.shape)
    # print(rtgArray.shape)
    # for t in range(stateArray.shape[0]-1):
    #     print(stateArray[t].shape)
    #     print(nextStateArray[t].shape)
    #     print(rtgArray[t].shape)
    #     stateArray[t] = np.concatenate((stateArray[t], rtgArray[t].reshape(-1,1)), axis=-1)
    #     nextStateArray[t] = np.concatenate((nextStateArray[t], rtgArray[t+1].reshape(-1,1)), axis=-1)
    # stateArray[-1] = np.concatenate((stateArray[-1], rtgArray[-1].reshape(-1,1)), axis=-1)
    # nextStateArray[-1] = np.concatenate((nextStateArray[-1], np.zeros(shape=(1,1))), axis=-1)

    newStateArray = np.concatenate((stateArray, rtgArray.reshape(-1,1)), axis=1)
    rtgArrayForNextStateArray = np.concatenate((rtgArray[1:],np.array([0.0])), axis=0)
    # print(rtgArrayForNextStateArray.shape)
    newNextStateArray = np.concatenate((nextStateArray, rtgArrayForNextStateArray.reshape(-1,1)), axis=1)

    # print(newStateArray.shape)
    # print(newNextStateArray.shape)
    # print(rtgArray.shape)
    # for i, j, k, l,m in zip(newStateArray, newNextStateArray, rtgArray, rtgArrayForNextStateArray,rewardArray):
    #     print(i,"\n",j,"\n",k, "\n",l,'\n', m)
    #     print("*" * 60)
    #     from time import sleep
    #     sleep(10)
    return newStateArray, newNextStateArray, rtgArray


def load_d4rl_buffer(variants):
    # env = gym.make(task[5:])
    env_name = variants["env"]
    dataset = variants["dataset"]

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
    else:
        raise NotImplementedError

    print(os.getcwd())
    # load dataset
    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variants.get('mode', 'normal')
    states, traj_lens, returns, rtgs = [], [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        if variants["model_type"] in ('cqlR', 'cqlR2'):
            path['observations'], path['next_observations'], path['rtg'] = addRtgToState(path['observations'], path['next_observations'], path['rewards'], 1.0, 1000)
        else:
            path['rtg'] = path['rewards']
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
        # [print("%.2f \t %.2f" % (rtg, reward)) for rtg, reward in zip(list(path['rtg']), list(path['rewards']))]
        # from time import sleep
        # sleep(10)


    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    trajectories_all = {}
    for single_traj in trajectories:
        for k,v in single_traj.items():
            if k not in trajectories_all:
                trajectories_all[k] = v
            else:
                trajectories_all[k] = np.concatenate((trajectories_all[k],v), axis=0)

    dataset = d4rl.qlearning_dataset(env, trajectories_all)
    # dataset = d4rl.qlearning_dataset(env, trajectories[0])

    buffer = SampleBatch(
        obs=dataset['observations'],
        obs_next=dataset['next_observations'],
        act=dataset['actions'],
        rew=np.expand_dims(np.squeeze(dataset['rewards']), 1),
        rtg=np.expand_dims(np.squeeze(dataset['rtgs']), 1),
        done=np.expand_dims(np.squeeze(dataset['terminals']), 1),
    )

    logger.info('obs shape: {}', buffer.obs.shape)
    logger.info('obs_next shape: {}', buffer.obs_next.shape)
    logger.info('act shape: {}', buffer.act.shape)
    logger.info('rew shape: {}', buffer.rew.shape)
    logger.info('rtg shape: {}', buffer.rtg.shape)
    logger.info('done shape: {}', buffer.done.shape)
    logger.info('Episode reward: {}', buffer.rew.sum() /np.sum(buffer.done) )
    logger.info('Number of terminals on: {}', np.sum(buffer.done))
    return buffer, state_mean, state_std
