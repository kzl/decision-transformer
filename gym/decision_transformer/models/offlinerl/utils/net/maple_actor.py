import torch.nn
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from offlinerl.utils.net.common import miniblock


class Maple_actor(nn.Module):
    def __init__(self, obs_dim, action_dim, deterministic=False, hidden_sizes=(16,), Guassain_hidden_sizes=(256,256), max_traj_len=5, LOG_MAX_STD=2, LOG_MIN_STD=-20, EPS=1e-8, lstm_hidden_unit=128):
        super(Maple_actor,self).__init__()
        self.obs_dim = obs_dim
        self.deterministic = deterministic
        self.act_dim = action_dim
        self.hidden_sizes = list(hidden_sizes).copy()
        self.Guassain_hidden_sizes = list(Guassain_hidden_sizes).copy()
        self.max_traj_len = max_traj_len
        self.LOG_MAX_STD = LOG_MAX_STD
        self.LOG_MIN_STD = LOG_MIN_STD
        self.EPS = EPS
        self.lstm_hidden_unit = lstm_hidden_unit
        self.mlp = miniblock(lstm_hidden_unit, hidden_sizes[0], None, relu=False)
        if len(hidden_sizes) >= 2:
            for i in range(1,len(hidden_sizes)):
                self.mlp += miniblock(hidden_sizes[i-1], hidden_sizes[i], None)
        self.mlp = nn.Sequential(*self.mlp)
        self.Guassain_input_dim = self.hidden_sizes[-1] + self.obs_dim
        self.Guassain_mlp = miniblock(self.Guassain_input_dim, self.Guassain_hidden_sizes[0], None)
        if len(Guassain_hidden_sizes)>=2:
            for i in range(1,len(Guassain_hidden_sizes)):
                self.Guassain_mlp += miniblock(Guassain_hidden_sizes[i-1], Guassain_hidden_sizes[i], None)
        self.Guassain_mlp = nn.Sequential(*self.Guassain_mlp)
        self.Guassain_mu_mlp = [nn.Linear(self.Guassain_hidden_sizes[-1], action_dim)]
        self.Guassain_logstd_mlp = [nn.Linear(self.Guassain_hidden_sizes[-1], action_dim)]
        self.Guassain_mu_mlp = nn.Sequential(*self.Guassain_mu_mlp)
        self.Guassain_logstd_mlp = nn.Sequential(*self.Guassain_logstd_mlp)
    def gaussian_likelihood(self,x, mu, log_std):
        pre_sum = -0.5 * (((x - mu) / (torch.exp(log_std) + self.EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return torch.sum(pre_sum, dim=-1)

    def forward(self, hidden_policy, obs): 
        policy_out = self.mlp(hidden_policy)
        policy_z = torch.cat([policy_out, obs], dim=-1)
        out = self.Guassain_mlp(policy_z)
        mu = self.Guassain_mu_mlp(out)
        log_std = self.Guassain_logstd_mlp(out)
        log_std = torch.clip(log_std, self.LOG_MIN_STD, self.LOG_MAX_STD)
        std = torch.exp(log_std)
        acts = torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(std)).sample()*std + mu
        log_p_acts = self.gaussian_likelihood(acts, mu, log_std)
        mu, acts, log_p_acts = self.apply_squashing_func(mu, acts, log_p_acts)
        return mu, acts, log_p_acts, std

    def apply_squashing_func(self, mu, pi, logp_pi):
        logp_pi -= torch.sum(2 * (np.log(2) - pi - F.softplus(-2 * pi)), dim=-1)
        # Squash those unbounded actions!
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        return mu, pi, logp_pi









