import torch
from copy import deepcopy
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.net.continuous import GaussianActor
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

    actor = GaussianActor(obs_shape, action_shape, args['actor_features'], args['actor_layers']).to(args['device'])

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args['actor_lr'])

    return {
        "actor" : {"net" : actor, "opt" : actor_optim},
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.actor = algo_init['actor']['net']
        self.actor_optim = algo_init['actor']['opt']

        self.batch_size = self.args['batch_size']
        self.device = self.args['device']

        self.best_actor = deepcopy(self.actor)
        self.best_loss = float('inf')
        
    def train(self, train_buffer, val_buffer, callback_fn):
        for epoch in range(self.args['max_epoch']):
            for i in range(self.args['steps_per_epoch']):
                batch_data = train_buffer.sample(self.batch_size)
                batch_data.to_torch(device=self.device)
                obs = batch_data['obs']
                action = batch_data['act']

                action_dist = self.actor(obs)
                loss = - action_dist.log_prob(action).mean()

                self.actor_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()

            with torch.no_grad():
                val_loss = 0
                for i in range(len(val_buffer) // self.batch_size + (len(val_buffer) % self.batch_size > 0)):
                    batch_data = val_buffer[i*self.batch_size:(i+1)*self.batch_size]
                    batch_data.to_torch(device=self.device)
                    obs = batch_data['obs']
                    action = batch_data['act']

                    action_dist = self.actor(obs)
                    val_loss += ((action_dist.mean - action) ** 2).mean().item()

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_actor.load_state_dict(self.actor.state_dict())

            res = callback_fn(self.get_policy())
            res['loss'] = val_loss
            
            self.log_res(epoch, res)

        return self.get_policy()
    
    def get_policy(self):
        return self.best_actor
