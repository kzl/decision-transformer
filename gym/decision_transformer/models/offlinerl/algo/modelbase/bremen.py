# Deployment-Efficient Reinforcement Learning via Model-based Offline Optimization
# Paper: https://openreview.net/forum?id=3hGNqpI4WS
# Code: https://github.com/matsuolab/BREMEN

import warnings

import torch
import gym
import numpy as np
from copy import deepcopy
from loguru import logger
from torch.distributions import kl_divergence

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.data import Batch
from offlinerl.utils.net.common import MLP
from offlinerl.utils.net.continuous import GaussianActor
from offlinerl.utils.exp import setup_seed
# from offlinerl.utils.env import get_env

from offlinerl.utils.net.model.ensemble import EnsembleTransition

def cg(Ax, b, cg_iters : int = 10):
    """
    Conjugate gradient algorithm
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
    """
    x = torch.zeros_like(b)
    r = b.clone() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
    p = r.clone()
    r_dot_old = torch.dot(r, r)
    for _ in range(cg_iters):
        z = Ax(p)
        with torch.no_grad():
            alpha = r_dot_old / (torch.dot(p, z) + 1e-8)
            x = x + alpha * p
            r = r - alpha * z
            r_dot_new = torch.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
    return x

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
    
    transition = EnsembleTransition(obs_shape, action_shape, args['transition_hidden_size'], args['transition_hidden_layers'], args['transition_init_num']).to(args['device'])
    transition_optim = torch.optim.AdamW(transition.parameters(), lr=args['transition_lr'], weight_decay=1e-5)

    behavior_actor = GaussianActor(obs_shape, action_shape, args['actor_hidden_size'], args['actor_hidden_layers']).to(args['device'])
    behavior_actor_optim = torch.optim.Adam(behavior_actor.parameters(), lr=args['bc_lr'])

    actor = GaussianActor(obs_shape, action_shape, args['actor_hidden_size'], args['actor_hidden_layers']).to(args['device'])

    value_net = MLP(obs_shape, 1, args['value_hidden_size'], args['value_hidden_layers']).to(args['device'])
    value_net_optim = torch.optim.Adam(value_net.parameters(), lr=args['value_lr'], weight_decay=1e-3)

    return {
        "transition" : {"net" : transition, "opt" : transition_optim},
        "behavior_actor" : {"net" : behavior_actor, "opt" : behavior_actor_optim},
        "actor" : {"net" : actor},
        "value_net" : {"net" : value_net, "opt" : value_net_optim},
    }

class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.gamma = self.args['gamma']
        self.lam = self.args['lam']

        self.transition = algo_init['transition']['net']
        self.transition_optim = algo_init['transition']['opt']

        self.behavior_actor = algo_init['behavior_actor']['net']
        self.behavior_actor_optim = algo_init['behavior_actor']['opt']

        self.actor = algo_init['actor']['net']

        self.value_net = algo_init['value_net']['net']
        self.value_net_optim = algo_init['value_net']['opt']

        self.device = args['device']

        # try to get the terminal function from env interface
        env_name = args["env"]
        try:
            if env_name == 'hopper':
                env = gym.make('Hopper-v3')
                max_ep_len = 1000
                scale = 1000.  # normalization for rewards/returns
            elif env_name == 'halfcheetah':
                env = gym.make('HalfCheetah-v3')
                max_ep_len = 1000
                scale = 1000.
            elif env_name == 'walker2d':
                env = gym.make('Walker2d-v3')
                max_ep_len = 1000
                scale = 1000.
            elif env_name == 'reacher2d':
                from decision_transformer.envs.reacher_2d import Reacher2dEnv
                env = Reacher2dEnv()
                max_ep_len = 100
                scale = 10.
            else:
                raise NotImplementedError
            # env = get_env()
            self.terminal_func = env.get_done_func()
        except:
            warnings.warn('Cannot get terminal function from env, assume infinite environment.')
            self.terminal_func = None
        
    def train(self, train_buffer, val_buffer, callback_fn):
        if self.args['dynamics_path'] is not None:
            self.transition = torch.load(self.args['dynamics_path'], map_location='cpu').to(self.device)
        else:
            self.train_transition(train_buffer)
        self.transition.requires_grad_(False)   
        if self.args['behavior_path'] is not None:
            self.behavior_actor = torch.load(self.args['behavior_path'], map_location='cpu').to(self.device)
        else:
            self.clone_policy(train_buffer)
        self.behavior_actor.requires_grad_(False)
        if self.args['bc_init']:
            self.actor.load_state_dict(self.behavior_actor.state_dict())
        self.train_policy(train_buffer, val_buffer, callback_fn)
    
    def get_policy(self):
        return self.actor

    def train_transition(self, buffer):
        data_size = len(buffer)
        val_size = min(int(data_size * 0.2) + 1, 1000)
        train_size = data_size - val_size
        train_splits, val_splits = torch.utils.data.random_split(range(data_size), (train_size, val_size))
        train_buffer = buffer[train_splits.indices]
        valdata = buffer[val_splits.indices]
        batch_size = self.args['transition_batch_size']

        val_losses = [float('inf') for i in range(self.transition.ensemble_size)]

        epoch = 0
        cnt = 0
        while True:
            epoch += 1
            idxs = np.random.randint(train_buffer.shape[0], size=[self.transition.ensemble_size, train_buffer.shape[0]])
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                batch = train_buffer[batch_idxs]
                self._train_transition(self.transition, batch, self.transition_optim)
            new_val_losses = self._eval_transition(self.transition, valdata)
            print(new_val_losses)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(val_losses)), new_val_losses, val_losses):
                if new_loss < old_loss:
                    indexes.append(i)
                    val_losses[i] = new_loss

            if len(indexes) > 0:
                self.transition.update_save(indexes)
                cnt = 0
            else:
                cnt += 1

            if cnt >= 5:
                break
        
        indexes = self._select_best_indexes(val_losses, n=self.args['transition_select_num'])
        self.transition.set_select(indexes)
        return self.transition

    def clone_policy(self, buffer):
        data_size = len(buffer)
        val_size = min(int(data_size * 0.2) + 1, 1000)
        train_size = data_size - val_size
        train_splits, val_splits = torch.utils.data.random_split(range(data_size), (train_size, val_size))
        train_buffer = buffer[train_splits.indices]
        valdata = buffer[val_splits.indices]
        batch_size = self.args['bc_batch_size']

        best_val_loss = float('inf')
        best_actor = deepcopy(self.behavior_actor)

        epoch = 0
        cnt = 0
        while True:
            epoch += 1
            idxs = np.random.randint(train_buffer.shape[0], size=[self.transition.ensemble_size, train_buffer.shape[0]])
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                batch = train_buffer[batch_idxs]
                batch.to_torch(device=self.device)
                
                action_dist = self.behavior_actor(batch.obs)
                actor_loss = - action_dist.log_prob(batch.act).sum(dim=-1).mean()

                self.behavior_actor_optim.zero_grad()
                actor_loss.backward()
                self.behavior_actor_optim.step()

            with torch.no_grad():
                valdata.to_torch(device=self.device)
                action_dist = self.behavior_actor(valdata.obs)
                val_loss = ((action_dist.mean - valdata.act) ** 2).mean().item()               
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_actor.load_state_dict(self.behavior_actor.state_dict())
                cnt = 0
            else:
                cnt += 1

            if cnt >= 5:
                break
        
        self.behavior_actor.load_state_dict(best_actor.state_dict())
        return self.behavior_actor

    def train_policy(self, train_buffer, val_buffer, callback_fn):
        max_obs = torch.as_tensor(train_buffer['obs_next'].max(axis=0), device=self.device)
        min_obs = torch.as_tensor(train_buffer['obs_next'].min(axis=0), device=self.device)
        max_reward = train_buffer['rew'].max()
        min_reward = train_buffer['rew'].min()

        for epoch in range(self.args['max_epoch']):
            for step in range(self.args['trpo_steps_per_epoch']):
                # collect data
                with torch.no_grad():
                    obs = train_buffer.sample(int(self.args['data_collection_per_epoch'] / self.args['horizon']))['obs']
                    obs = torch.tensor(obs, device=self.device)
                    traj = []
                    for t in range(self.args['horizon']):
                        action_dist = self.actor(obs)
                        if self.args['explore_mode'] == 'sample':
                            action = action_dist.sample()
                        elif self.args['explore_mode'] == 'static':
                            action = action_dist.mean
                            action += torch.randn_like(action) * self.args['static_noise']
                        value = self.value_net(obs)
                        obs_action = torch.cat([obs, action], dim=-1)
                        next_obs_dists = self.transition(obs_action)
                        next_obses = next_obs_dists.sample()
                        rewards = next_obses[:, :, -1:]
                        next_obses = next_obses[:, :, :-1]

                        model_indexes = np.random.randint(0, next_obses.shape[0], size=(obs.shape[0]))
                        next_obs = next_obses[model_indexes, np.arange(obs.shape[0])]
                        reward = rewards[model_indexes, np.arange(obs.shape[0])]
                        # reward = rewards.mean(dim=0)

                        next_obs = torch.max(torch.min(next_obs, max_obs), min_obs)
                        reward = torch.clamp(reward, min_reward, max_reward)

                        next_value = self.value_net(next_obs)

                        if self.terminal_func is not None:
                            dones = self.terminal_func({'obs' : obs, 'action' : action, 'next_obs' : next_obs})
                            dones = dones.to(reward)
                        else:
                            dones = torch.zeros_like(reward)

                        done_traj = int(torch.sum(dones))
                        if done_traj > 0:
                            replace_obs = train_buffer.sample(done_traj)['obs']
                            replace_obs = torch.tensor(replace_obs, device=self.device)
                            next_obs[torch.where(dones > 0)[0]] = replace_obs

                        traj.append(Batch({
                            "obs" : obs,
                            "obs_next" : next_obs,
                            "act" : action,
                            "rew" : reward,
                            "done" : dones,
                            "value" : value,
                            "next_value" : next_value,
                        }))

                        obs = next_obs

                traj = Batch.stack(traj, axis=0)

                self._trpo_update(traj)

            res = callback_fn(self.get_policy())
            res['reward'] = traj.rew.mean().item()
            self.log_res(epoch, res)

        return self.get_policy()

    def _trpo_update(self, traj):
        obs = traj['obs']
        action = traj['act']
        next_obs = traj['obs_next']
        reward = traj['rew']
        done = traj['done']
        value = traj['value']
        next_value = traj['next_value']

        # compute GAE
        advantages = torch.zeros_like(reward)
        td_error = torch.zeros_like(reward)
        pre_adv = 0
        for t in reversed(range(reward.shape[0])):
            mask = self.gamma * (1 - done[t])
            td_error[t] = reward[t] + mask * next_value[t] - value[t]
            advantages[t] = td_error[t] + mask * self.lam * pre_adv
            pre_adv = advantages[t]
        returns = value + advantages
        advantages = (advantages - advantages.mean(axis=0)) / (advantages.std(axis=0) + 1e-4)

        ''' update actor '''
        with torch.no_grad():
            old_action_dist = self.actor(obs)
            action_log_prob_old = old_action_dist.log_prob(action)

        action_dist = self.actor(obs)
        log_prob = action_dist.log_prob(action)
        ratio = (log_prob - action_log_prob_old).sum(dim=-1, keepdims=True).exp()
        p_loss = - (ratio * advantages).mean()

        p_grad = torch.cat([grad.view(-1) for grad in torch.autograd.grad(p_loss, self.actor.parameters(), create_graph=True)])
        kl = kl_divergence(action_dist, old_action_dist).mean()
        kl_grad = torch.cat([grad.view(-1) for grad in torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)])
        def hvp(x):
            return torch.cat([grad.view(-1) for grad in torch.autograd.grad(torch.sum(kl_grad * x), self.actor.parameters(), create_graph=True)]) + \
                self.args['damping_coeff'] * x

        x = cg(hvp, p_grad, cg_iters=self.args['cg_iters'])
        total_grad = torch.sqrt(2 * self.args['trpo_step_size'] / (torch.dot(x, hvp(x)) + 1e-8)) * x

        old_parameters = torch.cat([param.view(-1) for param in self.actor.parameters()])
        @torch.no_grad()
        def set_and_eval(new_paramters):
            try:
                parameter_index = 0
                for parameter in self.actor.parameters():
                    total_shape = np.prod(parameter.shape)
                    _param = new_paramters[parameter_index:parameter_index+total_shape].view(parameter.shape)
                    parameter_index += total_shape
                    parameter.data = _param
                new_action_dist = self.actor(obs)
                kl = kl_divergence(new_action_dist, old_action_dist).mean()
                return kl.item()
            except:
                return float('inf')

        update = False
        for j in range(self.args['backtrack_iters']):
            alpha = self.args['backtrack_coeff'] ** j
            new_parameters = old_parameters - alpha * total_grad
            new_kl = set_and_eval(new_parameters)
            if new_kl < self.args['trpo_step_size']: 
                update = True
                break

        if not update: 
            print('linear search fail, keep old parameter')
            set_and_eval(old_parameters)

        ''' update critic '''
        for _ in range(self.args['train_v_iters']):
            value = self.value_net(obs)
            v_loss = torch.mean((value - returns) ** 2)

            self.value_net_optim.zero_grad()
            v_loss.backward()
            self.value_net_optim.step()

    def _select_best_indexes(self, metrics, n):
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        selected_indexes = [pairs[i][1] for i in range(n)]
        return selected_indexes

    def _train_transition(self, transition, data, optim):
        data.to_torch(device=self.device)
        dist = transition(torch.cat([data['obs'], data['act']], dim=-1))
        loss = - dist.log_prob(torch.cat([data['obs_next'], data['rew']], dim=-1))
        loss = loss.mean()

        loss = loss + 0.01 * transition.max_logstd.mean() - 0.01 * transition.min_logstd.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        
    def _eval_transition(self, transition, valdata):
        with torch.no_grad():
            valdata.to_torch(device=self.device)
            dist = transition(torch.cat([valdata['obs'], valdata['act']], dim=-1))
            loss = ((dist.mean - torch.cat([valdata['obs_next'], valdata['rew']], dim=-1)) ** 2).mean(dim=(1,2))
            return list(loss.cpu().numpy())