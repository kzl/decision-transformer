# import fire
import argparse, pickle
import torch, numpy as np, gym
from offlinerl.algo import algo_select
from offlinerl.data.d4rl import load_d4rl_buffer
from offlinerl.evaluation import OnlineCallBackFunction

# for CQL evaluation
def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=None,
        state_std=None,
        model_type='cql'
):

    model.eval()
    model.to(device=device)

    state = env.reset()
    dimOfEnvState = state.shape[0]
    dimOfInputState = state_mean.shape[0]

    if dimOfEnvState + 1 == dimOfInputState:
        rtgMean = state_mean[-1]
        rtgStd = state_std[-1]
        rtgNow = target_return

        state = np.concatenate((state, np.array([(target_return - rtgMean) / (rtgStd + 1e-6)])), )
    # print(state)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        # env.render()
        # add padding
        # actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        tanh_normal = model((states.to(dtype=torch.float32) - state_mean) / state_std,)
        action = tanh_normal.rsample()
        action = action.detach().cpu().numpy()
        state, reward, done, _ = env.step(action[-1])
        if dimOfEnvState + 1 == dimOfInputState:
            if model_type=='cqlR2':
                rtgNow -= reward
            state = np.concatenate((state, np.array([(rtgNow - rtgMean) / (rtgStd + 1e-6)])), )

        cur_state = torch.from_numpy(state).to(device=device, dtype=torch.float32).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

def run_algo(variant):
    num_eval_episodes = variant['num_eval_episodes']
    model_type = variant['model_type']
    env_name = variant["env"]
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        env_targets = [3600, 1800]  # evaluation conditioning targets
        max_ep_len = 1000
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        env_targets = [12000, 6000]
        max_ep_len = 1000
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        env_targets = [5000, 2500]
        max_ep_len = 1000
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        scale = 10.
    else:
        raise NotImplementedError


    env_name, dataset = variant['env'], variant['dataset']
    if model_type == 'cqlR' or 'cqlR2':
        state_dim = env.observation_space.shape[0] + 1
    else:
        state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    variant['state_dim'] = state_dim
    variant['act_dim']= variant['action_shape'] = act_dim

    algo_init_fn, algo_trainer_obj, algo_config = algo_select(variant, None)
    train_buffer, state_mean, state_std = load_d4rl_buffer(variant)
    algo_init = algo_init_fn
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    print(env_name, state_dim, act_dim)
    mode = algo_config.get('mode', 'normal')
    device = algo_config["device"]
    device = torch.device(device)

    # load dataset
    # dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    # with open(dataset_path, 'rb') as f:
    #     trajectories = pickle.load(f)

    # save all path information into separate lists
    # states, traj_lens, returns = [], [], []
    # for path in trajectories:
    #     if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
    #         path['rewards'][-1] = path['rewards'].sum()
    #         path['rewards'][:-1] = 0.
    #     states.append(path['observations'])
    #     traj_lens.append(len(path['observations']))
    #     returns.append(path['rewards'].sum())
    # traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    # states = np.concatenate(states, axis=0)
    # state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        target_return=target_rew,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        model_type=model_type
                    )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if variant['eval']:
        algo_trainer.eval_saved_policy(train_buffer, None, eval_fns=eval_episodes(env_targets[0]), variant=algo_config)
    else:
        algo_trainer.train(train_buffer, None, eval_fns=eval_episodes(env_targets[0]),variant=algo_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--algo_name', type=str, default='cql')
    parser.add_argument('--task', type=str, default='d4rl-halfcheetah-medium-v0')
    parser.add_argument('--num_eval_episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_type', type=str, default='CQL')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    run_algo(vars(args))
