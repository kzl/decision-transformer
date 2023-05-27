#Discrete Batch-Constrained deep Q-Learning (BCQ)
import copy

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.net.bcq_net import Conv_Q, FC_Q
from offlinerl.utils.exp import setup_seed


def algo_init(args):
    logger.info('Run algo_init function')

    setup_seed(args['seed'])
    
    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
    elif "task" in args.keys():
        from offlinerl.utils.env import get_env_shape
        obs_shape, action_shape = get_env_shape(args['task'])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    else:
        raise NotImplementedError
        
    if isinstance(args["obs_shape"], int):
        state_dim = (
            4,
            84,
            84
        ) 
        
        critic = Conv_Q(state_dim[0], args["action_shape"]).to(args['device'])
    else:
        critic = FC_Q(np.prod(args["obs_shape"]), args["action_shape"]).to(args['device'])
        
    critic_opt = optim.Adam(critic.parameters(), **args["optimizer_parameters"])
    
        
    nets =  {
        "critic" : {"net" : critic, "opt" : critic_opt},
        
    }
        
    return nets


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args
        
        self.Q = algo_init["critic"]["net"]
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = algo_init["critic"]["opt"]
        
        self.discount = self.args["discount"]

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if self.args["polyak_target_update"] else self.copy_target_update
        self.target_update_frequency = self.args["target_update_frequency"]
        self.tau = self.args["tau"]

        # Decay for eps
        self.initial_eps = self.args["initial_eps"]
        self.end_eps = self.args["end_eps"]
        self.slope = (self.end_eps - self.initial_eps) / self.args["eps_decay_period"]

        # Evaluation hyper-parameters
        self.state_shape = (-1,) + self.args["obs_shape"] if isinstance(self.args["obs_shape"], int) else (-1, self.args["obs_shape"])
        self.eval_eps = self.args["eval_eps"]
        self.num_actions = self.args["action_shape"]

        # Threshold for "unlikely" actions
        self.threshold = self.args["BCQ_threshold"]

        # Number of training iterations
        self.iterations = 0

    def train(self, train_buffer, val_buffer, callback_fn):
        training_iters = 0
        while training_iters < self.args["max_timesteps"]:
            
            # Sample replay buffer
            batch = train_buffer.sample(self.args["batch_size"])
            batch = batch.to_torch(dtype=torch.float32, device=self.args["device"])
            reward = batch.rew
            done = batch.done
            state = batch.obs
            action = batch.act.to(torch.int64)
            next_state = batch.obs_next

            # Compute the target Q value
            with torch.no_grad():
                q, imt, i = self.Q(next_state)
                imt = imt.exp()
                imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()

                # Use large negative number to mask actions from argmax
                next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

                q, imt, i = self.Q_target(next_state)
                target_Q = reward + done * self.discount * q.gather(1, next_action).reshape(-1, 1)

            # Get current Q estimate
            current_Q, imt, i = self.Q(state)

            current_Q = current_Q.gather(1, action)

            # Compute Q loss
            q_loss = F.smooth_l1_loss(current_Q, target_Q)
            i_loss = F.nll_loss(imt, action.reshape(-1))

            Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

            # Optimize the Q
            self.Q_optimizer.zero_grad()
            Q_loss.backward()
            self.Q_optimizer.step()

            # Update target network by polyak or full copy every X iterations.
            self.maybe_update_target()
            training_iters += 1
            #print(training_iters ,self.args["eval_freq"])
            if training_iters % self.args["eval_freq"] == 0:
                res = callback_fn(self.get_policy())
                
                self.log_res(training_iters // self.args["eval_freq"], res)


    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
             self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "_Q")
        torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")


    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename + "_Q"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))
        
    def get_policy(self,):
        return self.Q
    
    def save_model(self):
        pass