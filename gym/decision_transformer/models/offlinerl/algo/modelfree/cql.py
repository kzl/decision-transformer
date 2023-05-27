# Conservative Q-Learning for Offline Reinforcement Learning
# https://arxiv.org/abs/2006.04779
# https://github.com/aviralkumar2907/CQL
import copy, os, sys, gym, time, json
import torch
import numpy as np
from torch import nn
from torch import optim
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.net.common import Net
from offlinerl.utils.net.continuous import Critic
from offlinerl.utils.net.tanhpolicy import TanhGaussianPolicy
from offlinerl.utils.exp import setup_seed


def algo_init(args):
    logger.info('Run algo_init function')

    # setup_seed(args['seed'])
    # print(vars(args.keys()))
    if args["state_dim"] and args["act_dim"]:
        obs_shape, action_shape = args["state_dim"], args["act_dim"]
        args["state_shape"], args["action_shape"] = args["state_dim"], args["act_dim"]
    else:
        raise NotImplementedError

    net_a = Net(layer_num=args['layer_num'],
                state_shape=obs_shape,
                hidden_layer_size=args['hidden_layer_size'])

    actor = TanhGaussianPolicy(preprocess_net=net_a,
                               action_shape=action_shape,
                               hidden_layer_size=args['hidden_layer_size'],
                               conditioned_sigma=True,
                               ).to(args['device'])

    actor_optim = optim.Adam(actor.parameters(), lr=args['actor_lr'])

    net_c1 = Net(layer_num=args['layer_num'],
                 state_shape=obs_shape,
                 action_shape=action_shape,
                 concat=True,
                 hidden_layer_size=args['hidden_layer_size'])
    critic1 = Critic(preprocess_net=net_c1,
                     hidden_layer_size=args['hidden_layer_size'],
                     ).to(args['device'])
    critic1_optim = optim.Adam(critic1.parameters(), lr=args['critic_lr'])

    net_c2 = Net(layer_num=args['layer_num'],
                 state_shape=obs_shape,
                 action_shape=action_shape,
                 concat=True,
                 hidden_layer_size=args['hidden_layer_size'])
    critic2 = Critic(preprocess_net=net_c2,
                     hidden_layer_size=args['hidden_layer_size'],
                     ).to(args['device'])
    critic2_optim = optim.Adam(critic2.parameters(), lr=args['critic_lr'])

    if args["use_automatic_entropy_tuning"]:
        if args["target_entropy"]:
            target_entropy = args["target_entropy"]
        else:
            target_entropy = -np.prod(args["action_shape"]).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=args['device'])
        alpha_optimizer = optim.Adam(
            [log_alpha],
            lr=args["actor_lr"],
        )

    nets = {
        "actor": {"net": actor, "opt": actor_optim},
        "critic1": {"net": critic1, "opt": critic1_optim},
        "critic2": {"net": critic2, "opt": critic2_optim},
        "log_alpha": {"net": log_alpha, "opt": alpha_optimizer, "target_entropy": target_entropy},

    }

    if args["lagrange_thresh"] >= 0:
        target_action_gap = args["lagrange_thresh"]
        log_alpha_prime = torch.zeros(1, requires_grad=True, device=args['device'])
        alpha_prime_optimizer = optim.Adam(
            [log_alpha_prime],
            lr=args["critic_lr"],
        )

        nets.update({"log_alpha_prime": {"net": log_alpha_prime, "opt": alpha_prime_optimizer}})

    return nets


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.actor = algo_init["actor"]["net"]
        self.actor_opt = algo_init["actor"]["opt"]

        self.critic1 = algo_init["critic1"]["net"]
        self.critic1_opt = algo_init["critic1"]["opt"]
        self.critic2 = algo_init["critic2"]["net"]
        self.critic2_opt = algo_init["critic2"]["opt"]
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        if args["use_automatic_entropy_tuning"]:
            self.log_alpha = algo_init["log_alpha"]["net"]
            self.alpha_opt = algo_init["log_alpha"]["opt"]
            self.target_entropy = algo_init["log_alpha"]["target_entropy"]

        if self.args["lagrange_thresh"] >= 0:
            self.log_alpha_prime = algo_init["log_alpha_prime"]["net"]
            self.alpha_prime_opt = algo_init["log_alpha_prime"]["opt"]

        self.critic_criterion = nn.MSELoss()

        self._n_train_steps_total = 0
        self._current_epoch = 0
        #
        # print(self.args)


    def _get_tensor_values(self, obs, actions, network):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        new_obs_actions, new_obs_log_pi = network(
            obs_temp, reparameterize=True, return_log_prob=True,
        )
        if not self.args["discrete"]:
            return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)
        else:
            return new_obs_actions

    def forward(self, obs, reparameterize=True, return_log_prob=True):
        log_prob = None
        tanh_normal = self.actor(obs, reparameterize=reparameterize, )

        if return_log_prob:
            if reparameterize is True:
                action, pre_tanh_value = tanh_normal.rsample(
                    return_pretanh_value=True
                )
            else:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )
            log_prob = tanh_normal.log_prob(
                action,
                pre_tanh_value=pre_tanh_value
            )
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            if reparameterize is True:
                action = tanh_normal.rsample()
            else:
                action = tanh_normal.sample()
        return action, log_prob

    def _train(self, batch):
        self._current_epoch += 1
        batch = batch.to_torch(dtype=torch.float32, device=self.args["device"])
        rewards = batch.rew
        terminals = batch.done
        obs = batch.obs
        actions = batch.act
        next_obs = batch.obs_next

        """
        Policy and Alpha Loss
        """
        new_obs_actions, log_pi = self.forward(obs)

        if self.args["use_automatic_entropy_tuning"]:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        if self._current_epoch < self.args["policy_bc_steps"]:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_log_prob = self.actor.log_prob(obs, actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()
        else:
            q_new_actions = torch.min(
                self.critic1(obs, new_obs_actions),
                self.critic2(obs, new_obs_actions),
            )

            policy_loss = (alpha * log_pi - q_new_actions).mean()
        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

        """
        QF Loss
        """
        q1_pred = self.critic1(obs, actions)
        q2_pred = self.critic2(obs, actions)

        new_next_actions, new_log_pi = self.forward(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        new_curr_actions, new_curr_log_pi = self.forward(
            obs, reparameterize=True, return_log_prob=True,
        )

        if self.args["type_q_backup"] == "max":
            target_q_values = torch.max(
                self.critic1_target(next_obs, new_next_actions),
                self.critic2_target(next_obs, new_next_actions),
            )
            target_q_values = target_q_values - alpha * new_log_pi

        elif self.args["type_q_backup"] == "min":
            target_q_values = torch.min(
                self.critic1_target(next_obs, new_next_actions),
                self.critic2_target(next_obs, new_next_actions),
            )
            target_q_values = target_q_values - alpha * new_log_pi
        elif self.args["type_q_backup"] == "medium":
            target_q1_next = self.critic1_target(next_obs, new_next_actions)
            target_q2_next = self.critic2_target(next_obs, new_next_actions)
            target_q_values = self.args["q_backup_lmbda"] * torch.min(target_q1_next, target_q2_next) \
                              + (1 - self.args["q_backup_lmbda"]) * torch.max(target_q1_next, target_q2_next)
            target_q_values = target_q_values - alpha * new_log_pi

        else:
            """when using max q backup"""
            next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=10, network=self.forward)
            target_qf1_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.critic1).max(1)[
                0].view(-1, 1)
            target_qf2_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.critic2).max(1)[
                0].view(-1, 1)
            target_q_values = torch.min(target_qf1_values, target_qf2_values)

        q_target = self.args["reward_scale"] * rewards + (1. - terminals) * self.args[
            "discount"] * target_q_values.detach()

        qf1_loss = self.critic_criterion(q1_pred, q_target)
        qf2_loss = self.critic_criterion(q2_pred, q_target)

        ## add CQL
        random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.args["num_random"],
                                                  actions.shape[-1]).uniform_(-1, 1).to(self.args["device"])
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.args["num_random"],
                                                                     network=self.forward)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.args["num_random"],
                                                                        network=self.forward)
        q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.critic1)
        q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.critic2)
        q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.critic1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.critic2)
        q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.critic1)
        q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.critic2)

        cat_q1 = torch.cat([q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1)
        cat_q2 = torch.cat([q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1)

        if self.args["min_q_version"] == 3:
            # importance sammpled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(),
                 q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(),
                 q2_curr_actions - curr_log_pis.detach()], 1
            )

        min_qf1_loss = torch.logsumexp(cat_q1 / self.args["temp"], dim=1, ).mean() * self.args["min_q_weight"] * \
                       self.args["temp"]
        min_qf2_loss = torch.logsumexp(cat_q2 / self.args["temp"], dim=1, ).mean() * self.args["min_q_weight"] * \
                       self.args["temp"]

        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.args["min_q_weight"]
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.args["min_q_weight"]

        if self.args["lagrange_thresh"] >= 0:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.args["lagrange_thresh"])
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.args["lagrange_thresh"])

            self.alpha_prime_opt.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_opt.step()

        qf1_loss = self.args["explore"] * qf1_loss + (2 - self.args["explore"]) * min_qf1_loss
        qf2_loss = self.args["explore"] * qf2_loss + (2 - self.args["explore"]) * min_qf2_loss

        """
        Update critic networks
        """
        self.critic1_opt.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        qf2_loss.backward()
        self.critic2_opt.step()

        """
        Soft Updates target network
        """
        self._sync_weight(self.critic1_target, self.critic1, self.args["soft_target_tau"])
        self._sync_weight(self.critic2_target, self.critic2, self.args["soft_target_tau"])

        self._n_train_steps_total += 1
        # print(self._current_epoch)
        if self._current_epoch % 1000 == 0:
            logger.info(
                'Pi loss and critic loss: {}, {}, {}'.format(policy_loss.item(), qf1_loss.item(), qf2_loss.item()))
        return policy_loss.item(), qf1_loss.item(), qf2_loss.item()

    def get_model(self):
        return self.actor

    def save_model(self, model_save_path):
        actor_para_path = model_save_path + "_actor.pt"
        critic1_para_path = model_save_path + "_critic1.pt"
        critic2_para_path = model_save_path + "_critic2.pt"
        torch.save(self.actor, actor_para_path)
        torch.save(self.critic1, critic1_para_path)
        torch.save(self.critic2, critic2_para_path)

    def load_q(self, para_path, iter=300, device='cuda:2'):
        if not torch.cuda.is_available():
            self.critic1 = torch.load(para_path + "iter_%d_critic1.pt" % (iter), map_location=torch.device('cpu'))
            self.critic2 = torch.load(para_path + "iter_%d_critic2.pt" % (iter), map_location=torch.device('cpu'))
        else:
            self.critic1 = torch.load(para_path + "iter_%d_critic1.pt" % (iter), map_location=device)
            self.critic2 = torch.load(para_path + "iter_%d_critic2.pt" % (iter), map_location=device)

    def get_q(self, state, action):
        # print("123")
        # print(self.critic1(state, action).cpu().detach().numpy())
        # print("123")
        q1 = self.critic1(state, action).cpu().detach().numpy().reshape(-1, 1)
        q2 = self.critic2(state, action).cpu().detach().numpy().reshape(-1, 1)
        return (q1+q2)/2, q1, q2

    def load_pi(self, para_path, iter=300,eval_fn=None):
        self.eval_fns = eval_fn
        if not torch.cuda.is_available():
            self.actor = torch.load(para_path + "iter_%d_actor.pt" % (iter), map_location=torch.device('cpu'))
        else:
            self.actor = torch.load(para_path + "iter_%d_actor.pt" % (iter))


    def get_action(self, state, reparameterize=True):
        action, logp = self.forward(state, reparameterize)
        return action, logp

    def get_policy(self):
        return self.actor


    def train(self, train_buffer, val_buffer, eval_fns, print_logs=True, variant=None):
        start_time = time.time()
        for epoch in range(1, self.args["max_epoch"] + 1):
            logs = dict()
            policy_losses = []
            qf_losses = []

            for step in range(1, self.args["steps_per_epoch"] + 1):
                train_data = train_buffer.sample(self.args["batch_size"])
                state_mean = np.mean(train_data.obs, axis=0)
                state_std = np.std(train_data.obs, axis=0) + 1e-6
                train_data.obs = (train_data.obs - state_mean) / state_std
                train_data.obs_next = (train_data.obs_next - state_mean) / state_std

                # print("train_data" + "=" * 80)
                # print(np.shape(train_data.obs))
                # print(np.shape(train_data.rtg))
                # from time import sleep
                # sleep(10)
                # print(np.shape(train_data.obs))
                # print(np.mean(train_data.obs, axis=0))
                # print(np.shape(train_data.obs_next))
                # print(np.mean(train_data.obs_next, axis=0))

                policy_loss, qf1_loss, qf2_loss = self._train(train_data)
                policy_losses.append(policy_loss)
                qf_losses.append(qf1_loss)
                qf_losses.append(qf2_loss)

            logs['time/training'] = time.time() - start_time
            eval_start = time.time()
            for eval_fn in [eval_fns]:
                self.actor.eval()
                outputs = eval_fn(self.actor)
                for k, v in outputs.items():
                    logs[f'evaluation/{k}'] = v

            logs['time/total'] = time.time() - start_time
            logs['time/evaluation'] = time.time() - eval_start
            logs['training/policy_losses_mean'] = np.mean(policy_losses)
            logs['training/policy_losses_std'] = np.std(policy_losses)
            logs['training/qf_losses_mean'] = np.mean(qf_losses)
            logs['training/qf_losses_std'] = np.std(qf_losses)

            self.diagnostics = {}

            for k in self.diagnostics:
                logs[k] = self.diagnostics[k]

            if print_logs:
                print('=' * 80)
                print(f'Iteration {epoch}')
                for k, v in logs.items():
                    print(f'{k}: {v}')
            save_path = r"saved_para/%s/%s/%s/%s" % (variant['model_type'], variant["env"], variant["dataset"], variant["mode"])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open('%s/config.json' % (save_path), 'w') as fp:
                json.dump(variant, fp)

            if epoch % 30 ==0 and epoch!=0 and epoch>=200:
                self.save_model('%s/iter_%s' % (save_path, str(epoch)))
            # print(logs)
            # res = callback_fn(self.get_policy())
            # self.log_res(epoch, res)

        return self.get_policy()

    def evaluate_episode(
            env,
            state_dim,
            act_dim,
            model,
            max_ep_len=1000,
            device='cuda',
            target_return=0,
            mode='normal',
            state_mean=0.,
            state_std=1.,
    ):
        model.eval()
        model.to(device=device)

        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)

        state = env.reset()

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
        sim_states = []

        episode_return, episode_length = 0, 0
        for t in range(max_ep_len):

            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return=target_return,
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, _ = env.step(action)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            episode_return += reward
            episode_length += 1

            if done:
                break

        return episode_return, episode_length

    # def eval_para(self, print_logs=False):
    #     logs = dict()
    #
    #     eval_start = time.time()
    #     # self.actor.eval()
    #
    #     outputs = evaluate_episode()
    #     for k, v in outputs.items():
    #         logs[f'evaluation/{k}'] = v
    #     # print(logs)
    #
    #     logs['time/total'] = time.time() - self.start_time
    #     logs['time/evaluation'] = time.time() - eval_start
    #
    #     for k in self.diagnostics:
    #         logs[k] = self.diagnostics[k]
    #
    #     if print_logs:
    #         print('=' * 80)
    #         # print(f'Iteration {iter_num}')
    #         for k, v in logs.items():
    #             print(f'{k}: {v}')
    #
    #     return logs

    def eval_para(self, print_logs=False):
        logs = dict()

        eval_start = time.time()
        self.actor.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.actor)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start

        # for k in self.diagnostics:
        #     logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            # print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def eval_saved_policy(self, train_buffer, val_buffer, eval_fns, print_logs=True, variant=None):
        start_time = time.time()
        for epoch in range(1, self.args["num_eval_episodes"] + 1):
            logs = dict()

            eval_start = time.time()
            for eval_fn in [eval_fns]:
                self.actor.eval()
                outputs = eval_fn(self.actor)
                for k, v in outputs.items():
                    logs[f'evaluation/{k}'] = v

            logs['time/total'] = time.time() - start_time
            logs['time/evaluation'] = time.time() - eval_start

            self.diagnostics = {}

            for k in self.diagnostics:
                logs[k] = self.diagnostics[k]

            if print_logs:
                print('=' * 80)
                print(f'Iteration {epoch}')
                for k, v in logs.items():
                    print(f'{k}: {v}')
            # save_path = r"saved_para/%s/%s/%s/%s" % (variant['model_type'], variant["env"], variant["dataset"], variant["mode"])
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # with open('%s/config.json' % (save_path), 'w') as fp:
            #     json.dump(variant, fp)

            # if epoch % 30 ==0 and epoch!=0:
            #     self.save_model('%s/iter_%s' % (save_path, str(epoch)))

        return self.get_policy()