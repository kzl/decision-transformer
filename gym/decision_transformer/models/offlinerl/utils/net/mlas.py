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

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)
        self.d3 = nn.Linear(hidden_size, action_dim)

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

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None, clip=None, raw=False):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(state.device)
            if clip is not None:
                z = z.clamp(-clip, clip)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        a = self.d3(a)
        if raw: 
            return a
        return self.max_action * torch.tanh(a)
    
    def policy_infer(self, obs):
        return self.decode(obs, z=self._actor(obs)[0])
    
class ActorPerturbation(nn.Module, BasePolicy):
    def __init__(self, state_dim, action_dim, latent_action_dim, max_action, max_latent_action=2, phi=0.05):
        super(ActorPerturbation, self).__init__()

        self.hidden_size = (400, 300, 400, 300)

        self.l1 = nn.Linear(state_dim, self.hidden_size[0])
        self.l2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.l3 = nn.Linear(self.hidden_size[1], latent_action_dim)

        self.l4 = nn.Linear(state_dim + action_dim, self.hidden_size[2])
        self.l5 = nn.Linear(self.hidden_size[2], self.hidden_size[3])
        self.l6 = nn.Linear(self.hidden_size[3], action_dim)

        self.max_latent_action = max_latent_action
        self.max_action = max_action
        self.phi = phi
        
        self.vae = None

    def forward(self, state, decoder):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        latent_action = self.max_latent_action * torch.tanh(self.l3(a))

        mid_action = decoder(state, z=latent_action)

        a = F.relu(self.l4(torch.cat([state, mid_action], 1)))
        a = F.relu(self.l5(a))
        a = self.phi * torch.tanh(self.l6(a))
        final_action = (a + mid_action).clamp(-self.max_action, self.max_action)
        return latent_action, mid_action, final_action
    
    def policy_infer(self, obs):
        
        return self(obs, self.vae.decode)[-1]