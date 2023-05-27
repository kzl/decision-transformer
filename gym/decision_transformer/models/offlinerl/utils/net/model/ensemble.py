import torch

from offlinerl.utils.function import soft_clamp
from offlinerl.utils.net.common import Swish

class EnsembleLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=7):
        super().__init__()

        self.ensemble_size = ensemble_size

        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))

        torch.nn.init.trunc_normal_(self.weight, std=1/(2*in_features**0.5))

        self.register_parameter('saved_weight', torch.nn.Parameter(self.weight.detach().clone()))
        self.register_parameter('saved_bias', torch.nn.Parameter(self.bias.detach().clone()))

        self.select = list(range(0, self.ensemble_size))

    def forward(self, x):
        weight = self.weight[self.select]
        bias = self.bias[self.select]

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias

        return x

    def set_select(self, indexes):
        assert len(indexes) <= self.ensemble_size and max(indexes) < self.ensemble_size
        self.select = indexes
        self.weight.data[indexes] = self.saved_weight.data[indexes]
        self.bias.data[indexes] = self.saved_bias.data[indexes]

    def update_save(self, indexes):
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]

class EnsembleTransition(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_features, hidden_layers, ensemble_size=7, mode='local', with_reward=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.mode = mode
        self.with_reward = with_reward
        self.ensemble_size = ensemble_size

        self.activation = Swish()

        module_list = []
        for i in range(hidden_layers):
            if i == 0:
                module_list.append(EnsembleLinear(obs_dim + action_dim, hidden_features, ensemble_size))
            else:
                module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size))
        self.backbones = torch.nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(hidden_features, 2 * (obs_dim + self.with_reward), ensemble_size)
        self.obs_mean = None
        self.obs_std = None
        self.register_parameter('max_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * 1, requires_grad=True))
        self.register_parameter('min_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * -5, requires_grad=True))

    def update_self(self, obs):
        self.obs_mean = obs.mean(dim=0)
        self.obs_std = obs.std(dim=0)

    def forward(self, obs_action):
        # Normalization for obs. If 'normaliza', no residual. 
        # use 'dims' to make forward work both when training and evaluating
        dims = len(obs_action.shape) - 2
        if self.obs_mean is not None:
            if dims:
                obs_mean = self.obs_mean.unsqueeze(0).expand(obs_action.shape[0], -1).to(obs_action.device)
                obs_std = self.obs_std.unsqueeze(0).expand(obs_action.shape[0], -1).to(obs_action.device)
            else:
                obs_mean = self.obs_mean.to(obs_action.device)
                obs_std = self.obs_std.to(obs_action.device)
            if self.mode == 'normalize':
                batch_size = obs_action.shape[dims]
                obs, action = torch.split(obs_action, [self.obs_dim, obs_action.shape[-1] - self.obs_dim], dim=-1)
                if dims:
                    obs = obs - obs_mean.unsqueeze(dims).expand(-1, batch_size, -1)
                    obs = obs / (obs_std.unsqueeze(dims).expand(-1, batch_size, -1) + 1e-8)
                else:
                    obs = obs - obs_mean.unsqueeze(dims).expand(batch_size, -1)
                    obs = obs / (obs_std.unsqueeze(dims).expand(batch_size, -1) + 1e-8)
                output = torch.cat([obs, action], dim=-1)
            else:
                output = obs_action
        else:
            output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mu, logstd = torch.chunk(self.output_layer(output), 2, dim=-1)
        logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
        # 'local': with residual
        if self.mode == 'local' or self.mode == 'normalize':
            if self.with_reward:
                obs, reward = torch.split(mu, [self.obs_dim, 1], dim=-1)
                obs = obs + obs_action[..., :self.obs_dim]
                mu = torch.cat([obs, reward], dim=-1)
            else:
                mu = mu + obs_action[..., :self.obs_dim]
        return torch.distributions.Normal(mu, torch.exp(logstd))

    def set_select(self, indexes):
        for layer in self.backbones:
            layer.set_select(indexes)
        self.output_layer.set_select(indexes)

    def update_save(self, indexes):
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)