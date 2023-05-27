# COMBO: Conservative Offline Model-Based Policy Optimization
# http://arxiv.org/abs/2102.08363
# No available code

import torch
import numpy as np
from copy import deepcopy
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.data import Batch
from offlinerl.utils.net.common import MLP, Net
from offlinerl.utils.net.tanhpolicy import TanhGaussianPolicy
from offlinerl.utils.exp import setup_seed

from offlinerl.utils.data import ModelBuffer
from offlinerl.utils.net.model.ensemble import EnsembleTransition

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

    args['target_entropy'] = - float(np.prod(action_shape))
    
    transition = EnsembleTransition(obs_shape, action_shape, args['hidden_layer_size'], args['transition_layers'], args['transition_init_num']).to(args['device'])
    transition_optim = torch.optim.Adam(transition.parameters(), lr=args['transition_lr'], weight_decay=0.000075)

    net_a = Net(layer_num=args['hidden_layers'], 
                state_shape=obs_shape, 
                hidden_layer_size=args['hidden_layer_size'])

    actor = TanhGaussianPolicy(preprocess_net=net_a,
                               action_shape=action_shape,
                               hidden_layer_size=args['hidden_layer_size'],
                               conditioned_sigma=True).to(args['device'])

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args['actor_lr'])

    log_alpha = torch.zeros(1, requires_grad=True, device=args['device'])
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args["critic_lr"])

    q1 = MLP(obs_shape + action_shape, 1, args['hidden_layer_size'], args['hidden_layers'], norm=None, hidden_activation='swish').to(args['device'])
    q2 = MLP(obs_shape + action_shape, 1, args['hidden_layer_size'], args['hidden_layers'], norm=None, hidden_activation='swish').to(args['device'])
    critic_optim = torch.optim.Adam([*q1.parameters(), *q2.parameters()], lr=args['critic_lr'])

    log_beta = torch.zeros(1, requires_grad=True, device=args['device'])
    beta_optimizer = torch.optim.Adam([log_beta], lr=args["critic_lr"])

    return {
        "transition" : {"net" : transition, "opt" : transition_optim},
        "actor" : {"net" : actor, "opt" : actor_optim},
        "log_alpha" : {"net" : log_alpha, "opt" : alpha_optimizer},
        "critic" : {"net" : [q1, q2], "opt" : critic_optim},
        "log_beta" : {"net" : log_beta, "opt" : beta_optimizer},
    }

class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.transition = algo_init['transition']['net']
        self.transition_optim = algo_init['transition']['opt']
        self.transition_optim_secheduler = torch.optim.lr_scheduler.ExponentialLR(self.transition_optim, gamma=0.99)
        self.selected_transitions = None

        self.actor = algo_init['actor']['net']
        self.actor_optim = algo_init['actor']['opt']

        self.log_alpha = algo_init['log_alpha']['net']
        self.log_alpha_optim = algo_init['log_alpha']['opt']

        self.q1, self.q2 = algo_init['critic']['net']
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.critic_optim = algo_init['critic']['opt']

        self.log_beta = algo_init['log_beta']['net']
        self.log_beta_optim = algo_init['log_beta']['opt']

        self.device = args['device']
        
    def train(self, train_buffer, val_buffer, callback_fn):
        if self.args['dynamics_path'] is not None:
            self.transition = torch.load(self.args['dynamics_path'], map_location='cpu').to(self.device)
        else:
            self.train_transition(train_buffer)
        self.transition.requires_grad_(False)   
        policy = self.train_policy(train_buffer, val_buffer, self.transition, callback_fn)
    
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

            self.transition_optim_secheduler.step()
        
        indexes = self._select_best_indexes(val_losses, n=self.args['transition_select_num'])
        self.transition.set_select(indexes)
        return self.transition

    def train_policy(self, train_buffer, val_buffer, transition, callback_fn):
        real_batch_size = int(self.args['policy_batch_size'] * self.args['real_data_ratio'])
        model_batch_size = self.args['policy_batch_size']  - real_batch_size
        
        model_buffer = ModelBuffer(self.args['buffer_size'])

        for epoch in range(self.args['max_epoch']):
            # collect data
            with torch.no_grad():
                obs = train_buffer.sample(int(self.args['data_collection_per_epoch']))['obs']
                obs = torch.tensor(obs, device=self.device)
                for t in range(self.args['horizon']):
                    action = self.actor(obs).sample()
                    obs_action = torch.cat([obs, action], dim=-1)
                    next_obs_dists = transition(obs_action)
                    next_obses = next_obs_dists.sample()
                    rewards = next_obses[:, :, -1:]
                    next_obses = next_obses[:, :, :-1]

                    next_obses_mode = next_obs_dists.mean[:, :, :-1]
                    next_obs_mean = torch.mean(next_obses_mode, dim=0)
                    diff = next_obses_mode - next_obs_mean
                    disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]
                    aleatoric_uncertainty = torch.max(torch.norm(next_obs_dists.stddev, dim=-1, keepdim=True), dim=0)[0]

                    model_indexes = np.random.randint(0, next_obses.shape[0], size=(obs.shape[0]))
                    next_obs = next_obses[model_indexes, np.arange(obs.shape[0])]
                    reward = rewards[model_indexes, np.arange(obs.shape[0])]
                    
                    print('average reward:', reward.mean().item())

                    dones = torch.zeros_like(reward)

                    batch_data = Batch({
                        "obs" : obs.cpu(),
                        "act" : action.cpu(),
                        "rew" : reward.cpu(),
                        "done" : dones.cpu(),
                        "obs_next" : next_obs.cpu(),
                    })

                    model_buffer.put(batch_data)

                    obs = next_obs

            # update
            for _ in range(self.args['steps_per_epoch']):
                batch = train_buffer.sample(real_batch_size)
                model_batch = model_buffer.sample(model_batch_size)
                batch = Batch.cat([batch, model_batch], axis=0)
                batch.to_torch(device=self.device)

                self._cql_update(batch)

            res = callback_fn(self.get_policy())
            
            res['disagreement_uncertainty'] = disagreement_uncertainty.mean().item()
            res['aleatoric_uncertainty'] = aleatoric_uncertainty.mean().item()
            res['beta'] = torch.exp(self.log_beta.detach()).item()
            res['reward'] = reward.mean().item()
            self.log_res(epoch, res)

        return self.get_policy()

    def _cql_update(self, batch_data):
        obs = batch_data['obs']
        action = batch_data['act']
        next_obs = batch_data['obs_next']
        reward = batch_data['rew']
        done = batch_data['done']
        batch_size = done.shape[0]

        '''update critic'''

        # normal bellman backup loss
        obs_action = torch.cat([obs, action], dim=-1)
        _q1 = self.q1(obs_action)
        _q2 = self.q2(obs_action)

        with torch.no_grad():
            next_action_dist = self.actor(next_obs)
            next_action = next_action_dist.sample()
            log_prob = next_action_dist.log_prob(next_action).sum(dim=-1, keepdim=True)
            next_obs_action = torch.cat([next_obs, next_action], dim=-1)
            _target_q1 = self.target_q1(next_obs_action)
            _target_q2 = self.target_q2(next_obs_action)
            alpha = torch.exp(self.log_alpha)
            y = reward + self.args['discount'] * (1 - done) * (torch.min(_target_q1, _target_q2) - alpha * log_prob)

        critic_loss = ((y - _q1) ** 2).mean() + ((y - _q2) ** 2).mean()

        # attach the value penalty term
        random_actions = torch.rand(self.args['num_samples'], batch_size, action.shape[-1]).to(action) * 2 - 1
        action_dist = self.actor(obs)
        sampled_actions = torch.stack([action_dist.rsample() for _ in range(self.args['num_samples'])], dim=0)

        random_next_actions = torch.rand(self.args['num_samples'], batch_size, action.shape[-1]).to(action) * 2 - 1
        next_action_dist = self.actor(next_obs)
        sampled_next_actions = torch.stack([next_action_dist.rsample() for _ in range(self.args['num_samples'])], dim=0)

        sampled_actions = torch.cat([random_actions, sampled_actions], dim=0)
        repeated_obs = torch.repeat_interleave(obs.unsqueeze(0), sampled_actions.shape[0], 0)
        sampled_q1 = self.q1(torch.cat([repeated_obs, sampled_actions], dim=-1))
        sampled_q2 = self.q2(torch.cat([repeated_obs, sampled_actions], dim=-1))

        sampled_next_actions = torch.cat([random_next_actions, sampled_next_actions], dim=0)
        repeated_next_obs = torch.repeat_interleave(next_obs.unsqueeze(0), sampled_next_actions.shape[0], 0)
        sampled_next_q1 = self.q1(torch.cat([repeated_next_obs, sampled_next_actions], dim=-1))
        sampled_next_q2 = self.q2(torch.cat([repeated_next_obs, sampled_next_actions], dim=-1))

        sampled_q1 = torch.cat([sampled_q1, sampled_next_q1], dim=0)
        sampled_q2 = torch.cat([sampled_q2, sampled_next_q2], dim=0)        

        if self.args['with_important_sampling']:
            # perform important sampling
            _random_log_prob = torch.ones(self.args['num_samples'], batch_size, 1).to(sampled_q1) * action.shape[-1] * np.log(0.5)
            _log_prob = action_dist.log_prob(sampled_actions[self.args['num_samples']:]).sum(dim=-1, keepdim=True)
            _next_log_prob = next_action_dist.log_prob(sampled_next_actions[self.args['num_samples']:]).sum(dim=-1, keepdim=True)
            is_weight = torch.cat([_random_log_prob, _log_prob, _random_log_prob, _next_log_prob], dim=0)
            sampled_q1 = sampled_q1 - is_weight
            sampled_q2 = sampled_q2 - is_weight

        q1_penalty = (torch.logsumexp(sampled_q1, dim=0) - _q1) * self.args['base_beta']
        q2_penalty = (torch.logsumexp(sampled_q2, dim=0) - _q2) * self.args['base_beta']

        if self.args['learnable_beta']:
            # update beta
            beta_loss = - torch.mean(torch.exp(self.log_beta) * (q1_penalty - self.args['lagrange_thresh']).detach()) - \
                torch.mean(torch.exp(self.log_beta) * (q2_penalty - self.args['lagrange_thresh']).detach())

            self.log_beta_optim.zero_grad()
            beta_loss.backward()
            self.log_beta_optim.step()

        q1_penalty = q1_penalty * torch.exp(self.log_beta)
        q2_penalty = q2_penalty * torch.exp(self.log_beta)

        critic_loss = critic_loss + torch.mean(q1_penalty) + torch.mean(q2_penalty)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # soft target update
        self._sync_weight(self.target_q1, self.q1, soft_target_tau=self.args['soft_target_tau'])
        self._sync_weight(self.target_q2, self.q2, soft_target_tau=self.args['soft_target_tau'])


        '''update actor'''
        if self.args['learnable_alpha']:
            # update alpha
            alpha_loss = - torch.mean(self.log_alpha * (log_prob + self.args['target_entropy']).detach())

            self.log_alpha_optim.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optim.step()

        # norm actor loss
        action_dist = self.actor(obs)
        new_action = action_dist.rsample()
        action_log_prob = action_dist.log_prob(new_action)
        new_obs_action = torch.cat([obs, new_action], dim=-1)
        q = torch.min(self.q1(new_obs_action), self.q2(new_obs_action))
        actor_loss = - q.mean() + torch.exp(self.log_alpha) * action_log_prob.sum(dim=-1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

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