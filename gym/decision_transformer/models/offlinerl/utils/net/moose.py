import torch
import torch.nn as nn
import torch.nn.functional as F

from offlinerl.utils.net.common import BasePolicy

class VAE(nn.Module, BasePolicy):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 latent_dim, 
                 max_action,
                 hidden_size=750):
        super(VAE, self).__init__()
        
        self.e1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        self.d1 = nn.Linear(latent_dim, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)
        self.d3 = nn.Linear(hidden_size, state_dim + action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        
        self._actor = None

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(z)

        return u, mean, std

    def decode(self, z):
        a = F.relu(self.d1(z))
        a = F.relu(self.d2(a))
        a = self.d3(a)
        return a

    
    def policy_infer(self, obs):
        return self.decode(obs)