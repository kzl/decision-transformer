# import fire
import argparse, pickle
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch, numpy as np, gym
from offlinerl.algo import algo_select
from offlinerl.data.d4rl import load_d4rl_buffer
from offlinerl.evaluation import OnlineCallBackFunction

from offlinerl.config.algo import cql_config, plas_config, mopo_config, moose_config, bcqd_config, bcq_config, bc_config, crr_config, combo_config, bremen_config, maple_config
from offlinerl.utils.config import parse_config
from offlinerl.algo.modelfree import cql, plas, bcqd, bcq, bc, crr
from offlinerl.algo.modelbase import mopo, moose, combo, bremen, maple

import matplotlib.pyplot as plt


def algo_select(command_args, root_path, iter=300):
    algo = cql
    algo_config_module = cql_config
    algo_config = parse_config(algo_config_module)
    for k, v in algo_config.items():
        command_args[k] = v

    algo_init = algo.algo_init(command_args)
    algo_trainer = algo.AlgoTrainer

    algo_trainer = algo_trainer(algo_init, algo_config)

    env_name = command_args["env"]
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        env_targets = [3600, 1800]  # evaluation conditioning targets
        max_ep_len = 1000
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        env_targets = [12000, 6000]
        max_ep_len = 1000
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        env_targets = [5000, 2500]
        max_ep_len = 1000
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        scale = 10.
    else:
        raise NotImplementedError


    env_name, dataset = command_args['env'], command_args['dataset']
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = os.path.dirname(os.path.dirname(os.getcwd()))+f'/data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = command_args.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    return algo_trainer, env, state_mean, state_std

def q_infer(root_path, iter):
    # algo_trainer.load_q(root_path + "/../../../gym/saved_para/CQL/walker2d/medium/normal/", iter)
    algo_trainer.load_q(root_path + "/../../../gym/saved_para/CQL/halfcheetah/medium/normal/", iter)

    get_q = algo_trainer.get_q
    return get_q

def pi_infer(root_path, iter):
    # algo_trainer.load_pi(root_path + "/../../../gym/saved_para/CQL/walker2d/medium/normal/", iter)
    algo_trainer.load_pi(root_path + "/../../../gym/saved_para/CQL/halfcheetah/medium/normal/", iter)
    get_action = algo_trainer.get_action
    return get_action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--algo_name', type=str, default='cql')
    parser.add_argument('--task', type=str, default='d4rl-halfcheetah-medium-v0')
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--model_type', type=str, default='CQL')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    root_path = os.getcwd()
    print(os.getcwd())
    algo_trainer, env, state_mean, state_std = algo_select(vars(args), root_path, 300)
    get_action = pi_infer(root_path, 300)

    state = env.reset()
    # env._paused = True
    # env._paused_time = 0.01

    done = None
    time_step = 0
    rtg_record = [0]
    q_record = [0]
    while not done:
        time_step+=1
        # for iter in range(90, 301, 30):
        env.render(mode='human')
        state = (state - state_mean)/(state_std+1e-6)
        state = torch.FloatTensor(state).reshape(1,-1)
        action, _ = get_action(state, False)
        action = action.detach().numpy()
        observation, reward, done, info = env.step(action)

        rtg_record.insert(0,reward+rtg_record[0])

        # 看看Q吧
        max_q = -1000
        for iter in range(90, 301, 30):
            get_q = q_infer(root_path, iter)
            qm, q1, q2 = get_q(state, action)
            max_q = max(max_q, qm)
        q_record.insert(0, max_q)
        print("time_step: %d \t rtg: %.4f \t q: %.4f"%(time_step, rtg_record[0], q_record[0]))

        state = observation

    plt.plot(range(0,len(q_record)),q_record, c='blue')
    plt.plot(range(0, len(rtg_record)), [i/100 for i in rtg_record], c='red')
    plt.title("trajecrory RTG")
    plt.show()




    # drwn q with timestep in trajectory
    # import pickle
    # dataset_path = root_path + '/../../data/hopper-expert-v2.pkl'
    # with open(dataset_path, 'rb') as f:
    #     trajectories = pickle.load(f)
    # num = len(trajectories)
    #
    # for traj_ind in range(1,21):
    #     # traj_ind = 0
    #     tuple_ind = 0
    #
    #     results = []
    #     length = len(trajectories[traj_ind]["observations"])
    #     for i in range(length):
    #         # traj_ind = np.random.randint(0,num)
    #         # traj_ind = 3
    #
    #         length = len(trajectories[traj_ind]["observations"])
    #         tuple_ind = i
    #         # tuple_ind += 1
    #
    #         state = trajectories[traj_ind]["observations"][tuple_ind]
    #         action = trajectories[traj_ind]["actions"][tuple_ind]
    #
    #         state = torch.FloatTensor(state).reshape(1,-1)
    #         action = torch.FloatTensor(action).reshape(1,-1)
    #         max_q = -1000
    #         for iter in range(90, 301, 30):
    #             get_q = q_infer(root_path, iter)
    #             qm, q1, q2 = get_q(state, action)
    #             max_q = max(max_q, qm)
    #         results.append(max_q)
    #         print("traj:%d \t tuple:%d  \t"%(traj_ind, tuple_ind), max_q)
    #     print(max_q)
    #     plt.plot(range(0,len(results)),results)
    #     plt.title("trajecrory %d"%(traj_ind))
    #     plt.show()
