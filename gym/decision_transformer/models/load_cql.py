# import fire
import argparse, pickle
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch, numpy as np, gym
from decision_transformer.models.offlinerl.algo import algo_select
from decision_transformer.models.offlinerl.data.d4rl import load_d4rl_buffer
from decision_transformer.models.offlinerl.evaluation import OnlineCallBackFunction

from decision_transformer.models.offlinerl.config.algo import cql_config, plas_config, mopo_config, moose_config, bcqd_config, bcq_config, bc_config, crr_config, combo_config, bremen_config, maple_config
from decision_transformer.models.offlinerl.utils.config import parse_config
from decision_transformer.models.offlinerl.algo.modelfree import cql, plas, bcqd, bcq, bc, crr
from decision_transformer.models.offlinerl.algo.modelbase import mopo, moose, combo, bremen, maple

def load_cql_q_network(env_name, dataset, mode, device, state_dim=None, action_dim=None):
    root_path = os.getcwd()

    algo = cql
    algo_config_module = cql_config
    algo_config = parse_config(algo_config_module)
    algo_config['device'] = device
    # for k, v in algo_config.items():
    #     command_args[k] = v
    algo_config['env'] = env_name
    algo_config['dataset'] = dataset
    algo_config['mode'] = mode
    algo_config['state_dim'] = state_dim
    algo_config['act_dim'] = action_dim


    algo_init = algo.algo_init(algo_config)
    algo_trainer = algo.AlgoTrainer

    algo_trainer = algo_trainer(algo_init, algo_config)

    algo_trainer.load_q(root_path + "/saved_para/CQL/%s/%s/%s/"%(env_name, dataset, mode), 300, device)

    get_q = algo_trainer.get_q
    return get_q

def load_cql_actor(env_name, dataset, mode, device):
    root_path = os.getcwd()

    algo = cql
    algo_config_module = cql_config
    algo_config = parse_config(algo_config_module)
    algo_config['device'] = device
    # for k, v in algo_config.items():
    #     command_args[k] = v
    algo_config['env'] = env_name
    algo_config['dataset'] = dataset
    algo_config['mode'] = mode

    algo_init = algo.algo_init(algo_config)
    algo_trainer = algo.AlgoTrainer

    algo_trainer = algo_trainer(algo_init, algo_config)

    algo_trainer.load_pi(root_path + "/saved_para/CQL/%s/%s/%s/"%(env_name, dataset, mode), 300, device=device)

    get_action = algo_trainer.get_action
    return get_action
