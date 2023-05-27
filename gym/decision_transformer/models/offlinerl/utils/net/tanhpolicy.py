import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal

from offlinerl.utils.net.common import BasePolicy
from offlinerl.utils.net.continuous import ActorProb


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X = tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, max_action=1, min_action=-1, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon
        # self.mode = torch.tanh(normal_mean)
        self.max_action = max_action
        self.min_action = min_action

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return (self.max_action-self.min_action)/2*torch.tanh(z)+(self.max_action+self.min_action)/2, z
        else:
            return (self.max_action-self.min_action)/2*torch.tanh(z)+(self.max_action+self.min_action)/2

    def atanh(self,x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5 * torch.log(one_plus_x / one_minus_x)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        unscaled_value = (2*value - (self.max_action+self.min_action))/(self.max_action - self.min_action)  # assume the actual actions have been transformed
        if pre_tanh_value is None:
            pre_tanh_value = self.atanh(unscaled_value)  # get the raw Gaussian distribution output

        # ==== previous calculation of tanh log_prob =====
        # self.normal.log_prob(pre_tanh_value) - torch.log(
        #     1 - value * value + self.epsilon
        # )
        # previous calculation of tanhGaussian log_prob is OK when the action is in (-1,1). To be more general, we need the following revision

        action_scale = (self.max_action-self.min_action)/2.0
        squashed_action = unscaled_value
        log_prob = self.normal.log_prob(pre_tanh_value) - torch.log(action_scale * (1 - squashed_action.pow(2)) + self.epsilon)
        return log_prob

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return (self.max_action-self.min_action)/2*torch.tanh(z)+(self.max_action+self.min_action)/2, z
        else:
            return (self.max_action-self.min_action)/2*torch.tanh(z)+(self.max_action+self.min_action)/2
    


    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                torch.zeros(self.normal_mean.size(), device=self.normal_mean.device),
                torch.ones(self.normal_std.size(), device=self.normal_mean.device)
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return (self.max_action-self.min_action)/2*torch.tanh(z)+(self.max_action+self.min_action)/2, z
        else:
            return (self.max_action-self.min_action)/2*torch.tanh(z)+(self.max_action+self.min_action)/2
        

class TanhGaussianPolicy(ActorProb, BasePolicy):
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -5
    MEAN_MIN = -9.0
    MEAN_MAX = 9.0
    
    def atanh(self,x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5*torch.log(one_plus_x/ one_minus_x)

    def log_prob(self, obs, actions):
        raw_actions = self.atanh(actions)
        logits, h = self.preprocess(obs)
        
        mean = self.mu(logits)
        mean = torch.clamp(mean, self.MEAN_MIN, self.MEAN_MAX)
        if self._c_sigma:
            log_std = torch.clamp(
                self.sigma(logits), min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX
            )
            std = log_std.exp()
        else:
            shape = [1] * len(mean.shape)
            shape[1] = -1
            log_std = (self.sigma.view(shape) + torch.zeros_like(mean))
            std = log_std.exp()

        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        return log_prob.sum(-1)

    def forward(
            self,
            obs,
            state=None,
            infor={},
            reparameterize=True,

    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        logits, h = self.preprocess(obs, state)
        mean = self.mu(logits)
        
        if self._c_sigma:
            log_std = torch.clamp(
                self.sigma(logits), min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX
            )
            std = log_std.exp()
        else:
            shape = [1] * len(mean.shape)
            shape[1] = -1
            log_std = (self.sigma.view(shape) + torch.zeros_like(mean))
            std = log_std.exp()
        
        return TanhNormal(mean, std)

    def get_action(self, states, actions, rewards, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
                             dtype=torch.float32, device=states.device), states], dim=1)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None, **kwargs)
        return actions[0,-1]

    def policy_infer(self, obs):
        print("obs", obs.shape)
        # return self.mode(obs)
        return self(obs).mode
