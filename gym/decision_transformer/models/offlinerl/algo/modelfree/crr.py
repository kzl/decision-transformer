# Critic regularized regression
# Paper: https://arxiv.org/abs/2006.15134

import torch
from copy import deepcopy
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.net.common import Net
from offlinerl.utils.net.continuous import DistributionalCritic
from offlinerl.utils.net.tanhpolicy import TanhGaussianPolicy
from offlinerl.utils.exp import setup_seed

def algo_init(args):
    logger.info('Run algo_init function')

    setup_seed(args['seed'])
    
    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
        max_action = args["max_action"]
    elif "task" in args.keys():
        from offlinerl.utils.env import get_env_shape, get_env_action_range
        obs_shape, action_shape = get_env_shape(args['task'])
        max_action, _ = get_env_action_range(args["task"])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    else:
        raise NotImplementedError

    net_a = Net(layer_num=args['hidden_layers'], 
                state_shape=obs_shape, 
                hidden_layer_size=args['hidden_features'])

    actor = TanhGaussianPolicy(preprocess_net=net_a,
                               action_shape=action_shape,
                               hidden_layer_size=args['hidden_features'],
                               conditioned_sigma=True).to(args['device'])

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args['lr'])

    critic = DistributionalCritic(obs_shape, action_shape, args['atoms'], 
                                  args['hidden_features'], args['hidden_layers'],
                                  None, None).to(args['device'])
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args['lr'])

    return {
        "actor" : {"net" : actor, "opt" : actor_optim},
        "critic" : {"net" : critic, "opt" : critic_optim},
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.actor = algo_init['actor']['net']
        self.actor_target = deepcopy(self.actor)
        self.actor_target.requires_grad_(False)
        self.actor_optim = algo_init['actor']['opt']

        self.critic = algo_init['critic']['net']
        self.critic_target = deepcopy(self.critic)
        self.critic_target.requires_grad_(False)
        self.critic_optim = algo_init['critic']['opt']

        self.batch_size = self.args['batch_size']
        self.gamma = self.args['gamma']
        self.beta = self.args['beta']
        self.m = self.args['advantage_samples']
        self.advantage_mode = self.args['advantage_mode']
        self.weight_mode = self.args['weight_mode']
        self.device = self.args['device']
        
    def train(self, train_buffer, val_buffer, callback_fn):
        rewards = train_buffer['rew']
        self.critic.set_interval(rewards.min() / (1 - self.gamma), rewards.max() / (1 - self.gamma))
        self.critic_target.set_interval(rewards.min() / (1 - self.gamma), rewards.max() / (1 - self.gamma))
        for epoch in range(self.args['max_epoch']):
            for i in range(self.args['steps_per_epoch']):
                batch_data = train_buffer.sample(self.batch_size)
                batch_data.to_torch(device=self.device)
                obs = batch_data['obs']
                action = batch_data['act']
                next_obs = batch_data['obs_next']
                reward = batch_data['rew']
                done = batch_data['done'].float()

                # update critic
                p = self.critic(obs, action)
                next_action = self.actor_target.get_action(next_obs)
                target_p = self.critic_target.get_target(next_obs, next_action, reward, self.gamma * (1 - done))
                critic_loss = - (target_p * torch.log(p + 1e-8)).mean()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # update actor
                action_dist = self.actor(obs)
                log_prob = action_dist.log_prob(action)
                actions = torch.stack([action_dist.sample() for _ in range(self.m)], dim=0)
                repeat_obs = torch.repeat_interleave(obs.unsqueeze(0), self.m, 0)
                _, values = self.critic(repeat_obs, actions, with_q=True)
                _, value = self.critic(obs, action, with_q=True)
                
                if self.advantage_mode == 'mean':
                    advantage = value - values.mean(dim=0)
                elif self.advantage_mode == 'max':
                    advantage = value - values.max(dim=0)[0]
                
                if self.weight_mode == 'exp':
                    weight = torch.exp(advantage / self.beta)
                elif self.weight_mode == 'binary':
                    weight = (advantage > 0).float()
                    
                weight = torch.clamp_max(weight, 20).detach()
                actor_loss = - torch.mean(weight * log_prob)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                if i % self.args['update_frequency']:
                    self._sync_weight(self.critic_target, self.critic, 1.0)
                    self._sync_weight(self.actor_target, self.actor, 1.0)
            print("actor_loss: ", actor_loss.item())
            res = callback_fn(self.get_policy())
            
            self.log_res(epoch, res)

        return self.get_policy()
    
    def get_policy(self):
        return self.actor