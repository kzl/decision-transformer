import argparse, pickle
import os, sys

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


    return algo_trainer

def q_infer(root_path, iter):
    algo_trainer.load_q(root_path + "/../../../gym/saved_para/CQL/walker2d/medium/normal/", iter)
    get_q = algo_trainer.get_q
    return get_q



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
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
    algo_trainer = algo_select(vars(args), root_path, iter)



    # state = [1.24992, 0.0019745931, 5.2146876e-05, -0.0013884939, 0.0009075383, 0.004452167, -0.003140935, 0.002659766,
    #  0.002327715, 0.0031283426, 0.0012417827]
    # action = [0.934155, -0.71524495, -0.10588102]
    # state = [1.2496322, 0.00055432064, 0.00050323014, -0.0052256254, 0.00049323996, -0.08161253, -0.06903866, -0.37237298,
    #  0.091809765, -0.9598944, -0.10461019]
    # action =  [0.35850546, -0.71467435, -0.15425174]
    # state = [0.8889891, 0.0236572, -1.3141984, -0.43941054, -0.75154144, 3.112586, -2.7534444, 0.63725746, -0.2491002,
    #  -4.0848465, 0.77194554]
    # action = [0.26374677, -0.1162667, 0.30589324]

    import pickle
    # dataset_path = root_path + '/../../data/hopper-expert-v2.pkl'
    dataset_path = root_path + '/../../data/walker2d-medium-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    num = len(trajectories)

    for traj_ind in range(1,21):
        # traj_ind = 0
        tuple_ind = 0

        results = []
        length = len(trajectories[traj_ind]["observations"])
        for i in range(length):
            # traj_ind = np.random.randint(0,num)
            # traj_ind = 3

            length = len(trajectories[traj_ind]["observations"])
            tuple_ind = i
            # tuple_ind += 1

            state = trajectories[traj_ind]["observations"][tuple_ind]
            action = trajectories[traj_ind]["actions"][tuple_ind]

            state = torch.FloatTensor(state).reshape(1,-1)
            action = torch.FloatTensor(action).reshape(1,-1)
            max_q = -1000
            for iter in range(90, 301, 30):
                get_q = q_infer(root_path, iter)
                qm, q1, q2 = get_q(state, action)
                max_q = max(max_q, qm)
            results.append(max_q)
            print("traj:%d \t tuple:%d  \t"%(traj_ind, tuple_ind), max_q)
        print(max_q)
        plt.plot(range(0,len(results)),results)
        plt.title("trajecrory %d"%(traj_ind))
        plt.show()
