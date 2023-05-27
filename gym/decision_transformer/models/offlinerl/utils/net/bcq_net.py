import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from offlinerl.utils.net.common import BasePolicy


# Used for Atari
class Conv_Q(nn.Module):
    def __init__(self, frames, num_actions):
        super(Conv_Q, self).__init__()
        self.c1 = nn.Conv2d(frames, 32, kernel_size=8, stride=4)
        self.c2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.q1 = nn.Linear(3136, 512)
        self.q2 = nn.Linear(512, 16)
        self.q3 = nn.Linear(16, num_actions)

        self.i1 = nn.Linear(3136, 512)
        self.i2 = nn.Linear(512, 16)
        self.i3 = nn.Linear(16, num_actions)


    def forward(self, state):
        c = F.relu(self.c1(state))
        c = F.relu(self.c2(c))
        c = F.relu(self.c3(c))

        q = F.relu(self.q1(c.reshape(-1, 3136)))
        q = F.relu(self.q2(q))
        q = self.q3(q)
        
        i = F.relu(self.i1(c.reshape(-1, 3136)))
        i = F.relu(self.i2(i))
        i = self.i3(i)
        return q, F.log_softmax(i, dim=1), i

    def encode(self, state):
        with torch.no_grad():
            c = F.relu(self.c1(state))
            c = F.relu(self.c2(c))
            c = F.relu(self.c3(c))

            q = F.relu(self.q1(c.reshape(-1, 3136)))
            q = F.relu(self.q2(q))

            i = F.relu(self.i1(c.reshape(-1, 3136)))
            i = F.relu(self.i2(i))            
            return i



# Used for Box2D / Toy problems
class FC_Q(nn.Module, BasePolicy):
    def __init__(self, state_dim, num_actions):
        super(FC_Q, self).__init__()
        self.q1 = nn.Linear(state_dim, 256)
        self.q2 = nn.Linear(256, 256)
        self.q3 = nn.Linear(256, num_actions)

        self.i1 = nn.Linear(state_dim, 256)
        self.i2 = nn.Linear(256, 256)
        self.i3 = nn.Linear(256, num_actions)


    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))

        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        i = F.relu(self.i3(i))
        return self.q3(q), F.log_softmax(i, dim=1), i
    
    def policy_infer(self, obs):
    
        q, imt, i = self(obs)
        imt = imt.exp()
        imt = (imt/imt.max(1, keepdim=True)[0] > 0.3).float()
        # Use large negative number to mask actions from argmax

        return (imt * q + (1. - imt) * -1e8).argmax(1)

