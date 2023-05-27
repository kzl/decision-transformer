import torch.nn
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from offlinerl.utils.net.common import miniblock

class GRU_Model(nn.Module):
    def __init__(self, obs_dim, action_dim,device=None, lstm_hidden_units=128):
        super(GRU_Model, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.lstm_hidden_units = lstm_hidden_units
        self.GRU = nn.GRU(self.obs_dim + self.action_dim, lstm_hidden_units, batch_first=True)
    def forward(self, obs, last_acts, pre_hidden, lens):
        sta_acs = torch.cat([obs, last_acts], dim=-1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(sta_acs,lens,batch_first=True, enforce_sorted=False)
        if len(pre_hidden.shape) == 2:
            pre_hidden = torch.unsqueeze(pre_hidden, dim=0)
        output,_ = self.GRU(packed, pre_hidden)
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output
    def get_hidden(self, obs, last_actions, lens):
        pre_hidden = torch.zeros((1,len(lens),self.lstm_hidden_units)).to(self.device)
        return self(obs, last_actions, pre_hidden,lens)
