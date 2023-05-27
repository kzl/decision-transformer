import gym
import neorl
import numpy as np
from typing import Tuple


def get_env(task : str) -> gym.Env:
    try:
        if task.startswith("HalfCheetah-v3"):
            env = neorl.make("HalfCheetah-v3")
        elif task.startswith("Hopper-v3"):
            env = neorl.make("Hopper-v3")
        elif task.startswith("Walker2d-v3"):   
            env = neorl.make("Walker2d-v3")
        elif task.startswith('d4rl'):
            import d4rl
            env = gym.make(task[5:])
            # hack to add terminal function 
            if 'hopper' in task:
                def terminal_function(data : dict):
                    obs = data["obs"]
                    action = data["action"]
                    obs_next = data["next_obs"]

                    singel_done = False
                    if len(obs.shape) == 1:
                        singel_done = True
                        obs = obs.reshape(1, -1)
                    if len(action.shape) == 1:
                        action = action.reshape(1, -1)
                    if len(obs_next.shape) == 1:
                        obs_next = obs_next.reshape(1, -1)

                    if isinstance(obs, np.ndarray):
                        array_type = np
                    else:
                        import torch
                        array_type = torch

                    z = obs_next[:, 0:1]
                    angle = obs_next[:, 1:2]
                    states = obs_next[:, 1:]

                    min_state, max_state = (-100.0, 100.0)
                    min_z, max_z = (0.7, float('inf'))
                    min_angle, max_angle = (-0.2, 0.2)

                    healthy_state = array_type.all(array_type.logical_and(min_state < states, states < max_state), axis=-1, keepdim=True)
                    healthy_z = array_type.logical_and(min_z < z, z < max_z)
                    healthy_angle = array_type.logical_and(min_angle < angle, angle < max_angle)

                    is_healthy = array_type.logical_and(array_type.logical_and(healthy_state, healthy_z), healthy_angle)

                    done = array_type.logical_not(is_healthy)

                    if singel_done:
                        done = done
                    else:
                        done = done.reshape(-1, 1)
                    return done

                env.get_done_func = lambda: terminal_function
            elif 'walker' in task:
                def terminal_function(data : dict):

                    obs = data["obs"]
                    action = data["action"]
                    obs_next = data["next_obs"]

                    singel_done = False
                    if len(obs.shape) == 1:
                        singel_done = True
                        obs = obs.reshape(1, -1)
                    if len(action.shape) == 1:
                        action = action.reshape(1, -1)
                    if len(obs_next.shape) == 1:
                        obs_next = obs_next.reshape(1, -1)

                    if isinstance(obs, np.ndarray):
                        array_type = np
                    else:
                        import torch
                        array_type = torch

                    min_z, max_z = (0.8, 2.0)
                    min_angle, max_angle = (-1.0, 1.0)
                    min_state, max_state = (-100.0, 100.0)
                    
                    z = obs_next[:, 0:1]
                    angle = obs_next[:, 1:2]
                    state = obs_next[:, 2:]
                    
                    healthy_state = array_type.all(array_type.logical_and(min_state < state, state < max_state), axis=-1, keepdim=True)
                    healthy_z = array_type.logical_and(min_z < z, z < max_z)
                    healthy_angle = array_type.logical_and(min_angle < angle, angle < max_angle)
                    is_healthy = array_type.logical_and(array_type.logical_and(healthy_state, healthy_z), healthy_angle)
                    done = array_type.logical_not(is_healthy)

                    if singel_done:
                        done = done
                    else:
                        done = done.reshape(-1, 1)
                        
                    return done

                env.get_done_func = lambda: terminal_function
        else:
            task_name = task.strip().split("-")[0]
            env = neorl.make(task_name)
    except:
            raise NotImplementedError

    return env

def get_env_shape(task : str) -> Tuple[int, int]:
    env = get_env(task)
    obs_dim = env.observation_space.shape
    action_space = env.action_space
    
    if len(obs_dim) == 1:
        obs_dim = obs_dim[0]
        
    if hasattr(env.action_space, 'n'):
        act_dim = env.action_space.n
    else:
        act_dim = action_space.shape[0]
    
    return obs_dim, act_dim

def get_env_obs_act_spaces(task : str):
    env = get_env(task)
    obs_space = env.observation_space
    act_space = env.action_space
    return obs_space, act_space

def get_env_action_range(task : str) -> Tuple[float, float]:
    env = get_env(task)
    act_max = float(env.action_space.high[0])
    act_min = float(env.action_space.low[0])
    
    return act_max, act_min  
    
def get_env_state_range(task : str) -> Tuple[float, float]:
    env = get_env(task)
    obs_max = float(env.observation_space.high[0])
    obs_min = float(env.observation_space.low[0])
    
    return obs_max, obs_min