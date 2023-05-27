import gym
import numpy as np
from collections import OrderedDict

from offlinerl.utils.env import get_env

def gym_policy_eval(task, eval_episodes=100):
    env = get_env(task)
    
    def policy_eval(policy):
        episode_rewards = []
        episode_lengths = []
        for _ in range(eval_episodes):
            state, done = env.reset(), False
            rewards = 0
            lengths = 0
            while not done:
                state = state[np.newaxis]
                action = policy.get_action(state).reshape(-1)
                state, reward, done, _ = env.step(action)
                rewards += reward
                lengths += 1

            episode_rewards.append(rewards)
            episode_lengths.append(lengths)


        rew_mean = np.mean(episode_rewards)
        len_mean = np.mean(episode_lengths)
       
        
        res = OrderedDict()
        res["Reward_Mean"] = rew_mean
        res["Length_Mean"] = len_mean

        return res
    
    return policy_eval


def gym_env_eval(task, eval_episodes=100):
    env = get_env(task)
    
    def env_eval(policy, obs_scaler=None, act_scaler=None):
        env_mae = []
        for _ in range(eval_episodes):
            state, done = env.reset(), False
            rewards = 0
            lengths = 0
            while not done:
                state = state[np.newaxis]              
                action = env.action_space.sample()
                
                obs = state.reshape(1,-1)
                act = action.reshape(1,-1)
                if obs_scaler is not None:
                    obs = obs_scaler.transform(obs)        
                if act_scaler is not None:   
                    act = act_scaler.transform(act)
                    
                policy_state = policy.get_action(np.concatenate([obs,act], axis=1))
                
                if obs_scaler is not None:
                    policy_state = obs_scaler.inverse_transform(policy_state)
                
                state, reward, done, _ = env.step(action)
                
                env_mae.append(np.mean(np.abs(policy_state -state)))

        env_mae = np.mean(env_mae)
       
        
        res = OrderedDict()
        res["Env_Mae"] = env_mae

        return res
    
    return env_eval 