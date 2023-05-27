# Off-policy deep reinforcement learning without exploration
# https://arxiv.org/abs/1812.02900
# https://github.com/sfujim/BCQ

import torch
import numpy as np
from copy import deepcopy
from loguru import logger
from torch.functional import F
from torch.distributions import Normal, kl_divergence

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.net.common import MLP
from offlinerl.utils.exp import setup_seed

class VAE(torch.nn.Module):
    def __init__(self, state_dim, action_dim, vae_features, vae_layers, max_action=1.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = 2 * self.action_dim
        self.max_action = max_action

        self.encoder = MLP(self.state_dim + self.action_dim, 2 * self.latent_dim, vae_features, vae_layers, hidden_activation='relu')
        self.decoder = MLP(self.state_dim + self.latent_dim, self.action_dim, vae_features, vae_layers, hidden_activation='relu')

    def encode(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        mu, logstd = torch.chunk(self.encoder(state_action), 2, dim=-1)
        logstd = torch.clamp(logstd, -4, 15)
        std = torch.exp(logstd)
        return Normal(mu, std)

    def decode(self, state, z=None):
        if z is None:
            param = next(self.parameters())
            z = torch.randn((*state.shape[:-1], self.latent_dim)).to(param)
            z = torch.clamp(z, -0.5, 0.5)

        action = self.decoder(torch.cat([state, z], dim=-1))
        action = self.max_action * torch.tanh(action)
        return action    

    def forward(self, state, action):
        dist = self.encode(state, action)
        z = dist.rsample()
        action = self.decode(state, z)
        return dist, action

class Jitter(torch.nn.Module):
    def __init__(self, state_dim, action_dim, jitter_features, jitter_layers, max_action=1.0, phi=0.05):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.phi = phi

        self.jitter_net = MLP(self.state_dim + self.action_dim, self.action_dim, jitter_features, jitter_layers, hidden_activation='relu')

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        noise = self.jitter_net(state_action)
        noise = self.phi * self.max_action * torch.tanh(noise)

        return torch.clamp(action + noise, -self.max_action, self.max_action)

class BCQPolicy(torch.nn.Module):
    def __init__(self, vae, jitter, q_net):
        super().__init__()
        self.vae = vae
        self.jitter = jitter
        self.q_net = q_net

    def forward(self, state):
        raise NotImplementedError
    
    @torch.no_grad()
    def get_action(self, state : np.ndarray):
        param = next(self.vae.parameters())
        state = torch.as_tensor(state).to(param)
        repeat_state = torch.repeat_interleave(state.unsqueeze(0), 100, 0)
        multiple_actions = self.jitter(repeat_state, self.vae.decode(repeat_state))
        state_action = torch.cat([repeat_state, multiple_actions], dim=-1)
        q = self.q_net(state_action)
        index = torch.argmax(q, dim=0).squeeze(dim=-1)
        action = multiple_actions[index, np.arange(index.shape[0])]
        return action.cpu().numpy()


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
    
    vae = VAE(obs_shape, action_shape, args['vae_features'], args['vae_layers'], max_action).to(args['device'])
    vae_optim = torch.optim.Adam(vae.parameters(), lr=args['vae_lr'])

    jitter = Jitter(obs_shape, action_shape, args['jitter_features'], args['jitter_layers'], max_action, args['phi']).to(args['device'])
    jitter_optim = torch.optim.Adam(jitter.parameters(), lr=args['jitter_lr'])

    q1 = MLP(obs_shape + action_shape, 1, args['value_features'], args['value_layers'], hidden_activation='relu').to(args['device'])
    q2 = MLP(obs_shape + action_shape, 1, args['value_features'], args['value_layers'], hidden_activation='relu').to(args['device'])
    critic_optim = torch.optim.Adam([*q1.parameters(), *q2.parameters()], lr=args['critic_lr'])

    return {
        "vae" : {"net" : vae, "opt" : vae_optim},
        "jitter" : {"net" : jitter, "opt" : jitter_optim},
        "critic" : {"net" : [q1, q2], "opt" : critic_optim},
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.vae = algo_init['vae']['net']
        self.vae_optim = algo_init['vae']['opt']

        self.jitter = algo_init['jitter']['net']
        self.jitter_target = deepcopy(self.jitter)
        self.jitter_optim = algo_init['jitter']['opt']

        self.q1, self.q2 = algo_init['critic']['net']
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.critic_optim = algo_init['critic']['opt']

        self.batch_size = self.args['batch_size']
        self.gamma = self.args['gamma']
        self.lam = self.args['lam']
        self.device = self.args['device']
        
    def train(self, train_buffer, val_buffer, callback_fn):
        for epoch in range(self.args['max_epoch']):
            for i in range(self.args['steps_per_epoch']):
                batch_data = train_buffer.sample(self.batch_size)
                batch_data.to_torch(device=self.device)
                obs = batch_data['obs']
                action = batch_data['act']
                next_obs = batch_data['obs_next']
                reward = batch_data['rew']
                done = batch_data['done'].float()

                # train vae
                dist, _action = self.vae(obs, action)
                kl_loss = kl_divergence(dist, Normal(0, 1)).sum(dim=-1).mean()
                recon_loss = ((action - _action) ** 2).sum(dim=-1).mean()
                vae_loss = kl_loss + recon_loss

                self.vae_optim.zero_grad()
                vae_loss.backward()
                self.vae_optim.step()

                # train critic
                with torch.no_grad():
                    repeat_next_obs = torch.repeat_interleave(next_obs.unsqueeze(0), 10, 0)
                    multiple_actions = self.jitter_target(repeat_next_obs, self.vae.decode(repeat_next_obs))

                    obs_action = torch.cat([repeat_next_obs, multiple_actions], dim=-1)
                    target_q1 = self.target_q1(obs_action)
                    target_q2 = self.target_q2(obs_action)

                    target_q = self.lam * torch.min(target_q1, target_q2) + (1 - self.lam) * torch.max(target_q1, target_q2)
                    target_q = torch.max(target_q, dim=0)[0]

                    target_q = reward + self.gamma * (1 - done) * target_q

                obs_action = torch.cat([obs, action], dim=-1)
                q1 = self.q1(obs_action)
                q2 = self.q2(obs_action)
                critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # train jitter
                action = self.vae.decode(obs)
                action = self.jitter(obs, action)
                obs_action = torch.cat([obs, action], dim=-1)
                jitter_loss = - self.q1(obs_action).mean()

                self.jitter_optim.zero_grad()
                jitter_loss.backward()
                self.jitter_optim.step()

                # soft target update
                self._sync_weight(self.jitter_target, self.jitter, soft_target_tau=self.args['soft_target_tau'])
                self._sync_weight(self.target_q1, self.q1, soft_target_tau=self.args['soft_target_tau'])
                self._sync_weight(self.target_q2, self.q2, soft_target_tau=self.args['soft_target_tau'])

            res = callback_fn(self.get_policy())
            
            res['kl_loss'] = kl_loss.item()
            self.log_res(epoch, res)

        return self.get_policy()

    #def save_model(self, model_save_path):
    #    torch.save(self.get_policy(), model_save_path)
    
    def get_policy(self):
        return BCQPolicy(self.vae, self.jitter, self.q1)