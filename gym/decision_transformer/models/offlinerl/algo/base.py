import os
import uuid
import json
from abc import ABC, abstractmethod

import torch
from collections import OrderedDict
from loguru import logger
# from offlinerl.utils.exp import init_exp_logger
from offlinerl.utils.io import create_dir, download_helper, read_json
from offlinerl.utils.logger import log_path


class BaseAlgo(ABC):
    def __init__(self, args):        
        logger.info('Init AlgoTrainer')
        if "exp_name" not in args.keys():
            exp_name = str(uuid.uuid1()).replace("-","")
        else:
            exp_name = args["exp_name"]
        
        # if "aim_path" in args.keys():
        #     if os.path.exists(args["aim_path"]):
        #         repo = args["aim_path"]
        # else:
        #     repo = None
        
        # self.repo = repo
        # self.exp_logger = init_exp_logger(repo = repo, experiment_name = exp_name)
        # if self.exp_logger.repo is not None:  # a naive fix of aim exp_logger.repo is None
        # self.index_path = self.exp_logger.repo.index_path
        # else:

        repo = os.path.join(log_path(),"./.aim")
        if not os.path.exists(repo):
            logger.info('{} dir is not exist, create {}',repo, repo)
            os.system(str("cd " + os.path.join(repo,"../") + "&& aim init"))
        self.index_path = repo
        # end else

        self.models_save_dir = os.path.join(self.index_path, "models")
        self.metric_logs = OrderedDict()
        self.metric_logs_path = os.path.join(self.index_path, "metric_logs.json")
        create_dir(self.models_save_dir)

        # self.exp_logger.set_params(args, name='hparams')
        
    
    def log_res(self, epoch, result):
        logger.info('Epoch : {}', epoch)
        for k,v in result.items():
            logger.info('{} : {}',k, v)
            self.exp_logger.track(v, name=k.split(" ")[0], epoch=epoch,)
        
        self.metric_logs[str(epoch)] = result
        with open(self.metric_logs_path,"w") as f:
            json.dump(self.metric_logs,f)
        self.save_model(os.path.join(self.models_save_dir, str(epoch) + ".pt"))
            
    
    @abstractmethod
    def train(self, 
              history_buffer,
              eval_fn=None,):
        pass
    
    def _sync_weight(self, net_target, net, soft_target_tau = 5e-3):
        for o, n in zip(net_target.parameters(), net.parameters()):
            o.data.copy_(o.data * (1.0 - soft_target_tau) + n.data * soft_target_tau)
    
    @abstractmethod
    def get_policy(self,):
        pass
    
    #@abstractmethod
    def save_model(self, model_path):
        torch.save(self.get_policy(), model_path)
        
    #@abstractmethod
    def load_model(self, model_path):
        model = torch.load(model_path)
        
        return model