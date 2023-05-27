import os.path, numpy as np
import time, copy
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from decision_transformer.training.iql import IQTTwinQ, ValueFunction, DeterministicPolicy, asymmetric_l2_loss, soft_update
from decision_transformer.training.trainer import Trainer

torch.autograd.set_detect_anomaly(True)

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

class TdSequenceTrainer(Trainer):
    def __init__(self, model, optimizer, variants, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        device = variants['device']
        # state_dim = variants['state_dim']
        action_dim = variants['act_dim']
        hidden_size = variants['embed_dim']
        max_action = variants['max_action']

        # q_network = IQTTwinQ(hidden_size-1).to(device)
        # v_network = ValueFunction(hidden_size-1).to(device)
        q_network = torch.nn.Linear(hidden_size, 1)
        v_network = torch.nn.Linear(hidden_size, 1)

        self.actor = (
            DeterministicPolicy(hidden_size, action_dim, max_action)
            # if config.iql_deterministic
            # else GaussianPolicy(state_dim, action_dim, max_action)
        ).to(device)
        v_optimizer = torch.optim.Adam(v_network.parameters(), lr=3e-4)
        q_optimizer = torch.optim.Adam(q_network.parameters(), lr=3e-4)
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        iql_tau = 0.7
        beta = 3.0
        max_steps = 1000000
        discount = 0.99
        tau = 0.005

        self.vf = v_network
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)

        self.model = model
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.batch_size = variants['batch_size']
        self.hidden_size = variants['embed_dim']
        self.seq_length = variants['seq_length']

        self.actor_optimizer = actor_optimizer
        self.q_optimizer = q_optimizer
        self.v_optimizer = v_optimizer
        self.optimizer = optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.start_time = time.time()

    def _update_v(self, observations, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v,
        observations,
        rewards,
        terminals,
        log_dict,
    ):
        print(np.shape(rewards))
        print(np.shape(terminals))
        print("next_v: ", np.shape(next_v.detach()))
        targets = rewards + (1.0 - terminals.float()) * self.discount * torch.squeeze(next_v).detach()
        # qs = self.qf.both(observations)
        qs = self.qf(observations).squeeze()
        print(qs.shape)
        # q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        q_loss = F.mse_loss(qs, targets)
        print(q_loss.shape)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(self, adv, observations, actions, log_dict):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=-1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv.squeeze() * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        state_preds, action_preds, reward_preds, hiddenState = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        # ### old code of predict action based on the mse loss ###
        action_target = torch.clone(actions)
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        #
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )
        #
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        # print(self.model.parameters)
        self.optimizer.step()
        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        # return loss.detach().cpu().item()

        with torch.no_grad():
            # 64, 20, 3, 128  -->>  64, 3, 20, 128
            hidden_state_ = hiddenState.reshape(self.batch_size, self.seq_length, 3, self.hidden_size).permute(0, 2, 1,
                                                                                                               3)
            hidden_state = hidden_state_[:, 2, 1:]
            next_hidden_state = hidden_state_[:, 2, :-1]
            print(hidden_state.shape)
            print(next_hidden_state.shape)

        log_dict = {}

        with torch.no_grad():
            # next_v: (64, 19, 1)
            next_v = self.vf(next_hidden_state)
        # Update value function
        adv = self._update_v(hidden_state, log_dict)
        rewards = rewards.squeeze(dim=-1)  # (64, 20)
        dones = dones.squeeze(dim=-1)  # (64, 20)
        # Update Q function
        self._update_q(next_v, hidden_state, rewards[:, :-1], dones[:, :-1], log_dict)
        # Update actor
        self._update_policy(adv, hidden_state, actions[:,:-1], log_dict)

        for k,v in log_dict.items():
            print(k, v)

        return log_dict['q_loss']


    def save_para(self, state_dict, path="saved_para/", iter="unknown"):
        torch.save(state_dict, path + "/iter_%s.pt" % (str(iter)))