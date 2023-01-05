import torch
import numpy as np
import gym
import time
import os
import mujoco_py
# Set the LD_LIBRARY_PATH environment variable
os.environ['LD_LIBRARY_PATH'] = '/home/nkolln/.mujoco/mujoco210/bin'

def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=100000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=1000,
        mode='normal',
        tests=1,
    ):

    model.eval()
    model.to(device=device)

    #states = np.concatenate(observation, axis=0)
    state_mean = np.array([ 1.3490015,  -0.11208222, -0.5506444,  -0.13188992, -0.00378754, 2.6071432, 0.02322114,-0.01626922,-0.06840388,-0.05183131, 0.04272673])
    
    # state_mean = np.array([ 1.2384834e+00, 1.9578537e-01, -1.0475016e-01, -1.8579608e-01, 2.3003316e-01, 2.2800924e-02, -3.7383768e-01,3.3779100e-01, 3.9250960e+00, -4.7428459e-03, 
    # 2.5267061e-02,-3.9287535e-03,-1.7367510e-02, -4.8212224e-01, 3.5432147e-04, -3.7124525e-03, 2.6285544e-03])        
    state_std = np.array([0.06664903, 0.16980624, 0.17309439, 0.21843709, 0.74599105, 0.02410989, 0.3729872, 0.6226182, 0.9708009, 0.72936815, 1.504065, 2.495893, 3.511518,5.3656907,0.79503316,4.317483,
    6.1784487 ])
    state_std = np.array([0.15980862, 0.0446214, 0.14307782, 0.17629202, 0.5912333, 0.5899924, 1.5405099, 0.8152689, 2.0173461, 2.4107876, 5.8440027 ])
    #state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    print(state_mean.shape,state_std.shape)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    #print('State type: {}\tState shape: {}\n State: {}\nState 0:{}'.format(type(state),len(state),state,state[0]))
    states = torch.from_numpy(state[0]).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    #print('State transformed to dim shape: {}\tState_dim: {}'.format(states.shape,state_dim))
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for i in range(tests):
        env.reset()
        print(i)
        t1 = time.time()
        for t in range(max_ep_len):

            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()
            #print('env.step return shape: {}'.format(len(env.step(action))))
            state, reward, done, truncated,_= env.step(action)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            if mode != 'delayed':
                pred_return = target_return[0,-1] - (reward/scale)
            else:
                pred_return = target_return[0,-1]
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done or t ==950:
                print(f'done with time {time.time()-t1:.2f}')
                break

    return episode_return, episode_length


model = torch.load('saved/hopper-expert-both_lin_iter_5000v2.pt')

env = gym.make("Hopper-v3",render_mode ='human')

state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]


for _ in range(10):
    evaluate_episode_rtg(env,
            state_dim,
            act_dim,
            model)
