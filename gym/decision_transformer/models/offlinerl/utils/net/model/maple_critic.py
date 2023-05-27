import torch.nn
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from offlinerl.utils.net.common import miniblock


class Maple_critic(nn.Module):
    def __init__(self, obs_dim, action_dim,deterministic=False,hidden_sizes=(16,),value_hidden_sizes=(256,256),lstm_hidden_unit=128):
        super(Maple_critic,self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.deterministic = deterministic
        self.hidden_sizes = list(hidden_sizes).copy()
        self.value_hidden_sizes = list(value_hidden_sizes).copy()
        self.lstm_hidden_unit = lstm_hidden_unit
        self.mlp = miniblock(self.lstm_hidden_unit, self.hidden_sizes[0], None, relu=False)
        if len(self.hidden_sizes) >= 2:
            for i in range(1,len(self.hidden_sizes)):
                self.mlp += miniblock(self.hidden_sizes[i-1], self.hidden_sizes[i], None)
        self.mlp = nn.Sequential(*self.mlp)
        self.vfs = miniblock(self.hidden_sizes[-1]+self.obs_dim+self.action_dim, self.value_hidden_sizes[0],None)
        if len(self.value_hidden_sizes)>=2:
            for i in range(1, len(self.value_hidden_sizes)):
                self.vfs += miniblock(self.value_hidden_sizes[i-1], self.value_hidden_sizes[i], None)
        self.vfs += [nn.Linear(self.value_hidden_sizes[-1], 1)]
        self.vfs = nn.Sequential(*self.vfs)

    def forward(self, value_hidden, actions, obs):
        out = self.mlp(value_hidden)
        out = torch.cat([out, obs, actions], dim=-1)
        out = self.vfs(out)
        return out
























