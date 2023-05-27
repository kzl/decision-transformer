import os
import glob
import pickle
import gzip
import pdb
import numpy as np
import torch
torch.set_default_tensor_type(torch.FloatTensor)

def restore_pool(replay_pool, experiment_root, max_size, save_path=None, adapt=False, maxlen=5,
                 policy_hook=None, value_hook=None, fake_env=None):
    print('[ DEBUG ]: start restoring the pool')
    if 'd4rl' in experiment_root:
        restore_pool_d4rl(replay_pool, experiment_root[5:], adapt, maxlen, policy_hook,value_hook,fake_env=fake_env)
    else:
        assert os.path.exists(experiment_root)
        if os.path.isdir(experiment_root):
            restore_pool_softlearning(replay_pool, experiment_root, max_size, save_path)
        else:
            try:
                restore_pool_contiguous(replay_pool, experiment_root)
            except:
                restore_pool_bear(replay_pool, experiment_root)
    print('[ mbpo/off_policy ] Replay pool has size: {}'.format(replay_pool.size))

def allocate_hidden_state(replay_pool_full_traj, get_action, make_hidden):
    state = replay_pool_full_traj
    pass

def restore_pool_d4rl(replay_pool, name, adapt=True, maxlen=5, policy_hook=None,value_hook=None,device=None,fake_env=None):
    import gym
    import d4rl
    if 'sac_data' in name:
        npz_data = np.load(f'../data/{name[9:]}')
        data = {}
        for k in npz_data.keys():
            data[k] = npz_data[k]
    else:
        env = gym.make(name)
        is_hopper_medium = name == 'hopper-medium-v0'
        is_hopper_med_expert = name == 'hopper-medium-expert-v0'
        illed_idx_hopper_medium = [113488, 171088, 294360, 381282, 389466, 703200, 871770, 995870]
        illed_idx_hopper_medium_exp = get_illed_med_exp()

        mask_steps = env._max_episode_steps - 1
        data = d4rl.qlearning_dataset(gym.make(name))
        if is_hopper_medium:
            for k in data:
                data[k] = np.delete(data[k], illed_idx_hopper_medium, axis=0)
        if is_hopper_med_expert:
            for k in data:
                data[k] = np.delete(data[k], illed_idx_hopper_medium_exp, axis=0)

    get_policy_hidden = policy_hook if policy_hook else None
    get_value_hidden = value_hook if value_hook else None
    data['rewards'] = np.expand_dims(data['rewards'], axis=1)
    data['terminals'] = np.expand_dims(data['terminals'], axis=1)
    data['last_actions'] = np.concatenate((np.zeros((1, data['actions'].shape[1]), dtype=np.float32), data['actions'][:-1, :]), axis=0).copy()
    data['first_step'] = np.zeros_like(data['terminals'])
    data['end_step'] = np.zeros_like(data['terminals'])
    data['valid'] = np.ones_like(data['terminals'])
    max_traj_len = -1
    last_start = 0
    traj_num = 1
    traj_lens = []
    ill_idxs = []
    for i in range(data['observations'].shape[0]):
        non_terminal = True
        if i >= 1:
            non_terminal = (data['observations'][i] == data['next_observations'][i-1]).all()
            if data['terminals'][i-1]:
                non_terminal = False

        if not non_terminal:
            if data['terminals'][i - 1]:
                data['next_observations'][i - 1] = data['observations'][i - 1]
            data['last_actions'][i][:] = 0
            data['first_step'][i][:] = 1
            data['end_step'][i-1][:] = 1
            traj_len = i - last_start
            max_traj_len = max(max_traj_len, traj_len)
            last_start = i
            traj_num += 1
            traj_lens.append(traj_len)
            if traj_len > 1000:
                print('[ DEBUG + WARN ]: trajectory length is too large: current step is ', i, traj_num,)
    print(ill_idxs)
    traj_lens.append(data['observations'].shape[0] - last_start)
    assert len(traj_lens) == traj_num
    if adapt and policy_hook is not None:
        # making init hidden state
        # 1, making state and lst action
        data['policy_hidden'] = None # np.zeros((data['last_actions'].shape[0], policy_hidden.shape[-1]))
        data['value_hidden'] = None # np.zeros((data['last_actions'].shape[0], value_hidden.shape[-1]))
        last_start_ind = 0
        traj_num_to_infer = 400
        for i_ter in range(int(np.ceil(traj_num / traj_num_to_infer))):
            traj_lens_it = traj_lens[traj_num_to_infer * i_ter : min(traj_num_to_infer * (i_ter + 1), traj_num)]
            states = np.zeros((len(traj_lens_it), max_traj_len, data['observations'].shape[-1]), dtype=np.float32)
            actions = np.zeros((len(traj_lens_it), max_traj_len, data['actions'].shape[-1]), dtype=np.float32)
            lst_actions = np.zeros((len(traj_lens_it), max_traj_len, data['last_actions'].shape[-1]), dtype=np.float32)
            start_ind = last_start_ind
            for ind, item in enumerate(traj_lens_it):
                states[ind, :item] = data['observations'][start_ind:(start_ind+item)]
                lst_actions[ind, :item] = data['last_actions'][start_ind:(start_ind+item)]
                actions[ind, :item] = data['actions'][start_ind:(start_ind+item)]
                start_ind += item
            with torch.no_grad():
                policy_hidden_out = get_policy_hidden.get_hidden(torch.from_numpy(states).to(device),\
                                                                 torch.from_numpy(lst_actions).to(device), torch.IntTensor(traj_lens_it))
                value_hidden_out = get_value_hidden.get_hidden(torch.from_numpy(states).to(device),\
                                                               torch.from_numpy(lst_actions).to(device), torch.IntTensor(traj_lens_it))
            max_len = max(traj_lens_it)
            policy_hidden = np.concatenate((np.zeros((len(traj_lens_it), 1, policy_hidden_out.shape[-1]), dtype=np.float32),
                                            policy_hidden_out[:, :-1].cpu().detach().numpy()), axis=-2)
            value_hidden = np.concatenate((np.zeros((len(traj_lens_it), 1, value_hidden_out.shape[-1]), dtype=np.float32),
                                           value_hidden_out[:, :-1].cpu().detach().numpy()), axis=-2)
            start_ind = last_start_ind
            for ind, item in enumerate(traj_lens_it):
                if data['policy_hidden'] is None:
                    data['policy_hidden'] = np.zeros((data['last_actions'].shape[0], policy_hidden.shape[-1]), dtype=np.float32)
                    data['value_hidden'] = np.zeros((data['last_actions'].shape[0], value_hidden.shape[-1]), dtype=np.float32)
                data['policy_hidden'][start_ind:(start_ind + item)] = policy_hidden[ind, :item]
                data['value_hidden'][start_ind:(start_ind + item)] = value_hidden[ind, :item]
                start_ind += item
            last_start_ind = start_ind
    print('[ DEBUG ]: inferring hidden state done')
    data_target = {k: replay_pool.fields[k] for k in replay_pool.fields}
    traj_target_ind = 0
    mini_target_ind = 0
    mini_traj_cum_num = 0
    for k in data_target:
        data_target[k][traj_target_ind, :] = 0
    if adapt:
        for i in range(data['policy_hidden'].shape[0]):
            if mini_target_ind == 0:
                for k in data:
                    if k in data_target:
                        data_target[k][traj_target_ind, :] = 0
            for k in data:
                if k in data_target:
                    data_target[k][traj_target_ind, mini_target_ind, :] = data[k][i]
            mini_target_ind += 1
            if data['end_step'][i] or mini_target_ind == maxlen:
                traj_target_ind += 1
                traj_target_ind = traj_target_ind % replay_pool._max_size
                mini_traj_cum_num += 1
                mini_target_ind = 0
        replay_pool._pointer += mini_traj_cum_num
        replay_pool._size = int(min(mini_traj_cum_num + replay_pool._size, replay_pool._max_size))
        replay_pool._pointer %= replay_pool._max_size
        print('[ DEBUG ] data num: {}, max size: {}'.format(mini_traj_cum_num, replay_pool._max_size))
        return
    replay_pool.add_samples(data)

def reset_hidden_state(replay_pool, name, maxlen=5, policy_hook=None, value_hook = None,device=None, fake_env=None):
    import gym
    import d4rl
    if 'sac_data' in name:
        npz_data = np.load(f'../data/{name[9:]}')
        data = {}
        for k in npz_data.keys():
            data[k] = npz_data[k]
    else:
        env = gym.make(name)
        is_hopper_medium = name == 'hopper-medium-v0'
        is_hopper_med_expert = name == 'hopper-medium-expert-v0'
        illed_idx_hopper_medium = [113488, 171088, 294360, 381282, 389466, 703200, 871770, 995870]
        illed_idx_hopper_medium_exp = get_illed_med_exp()

        # mask_steps = env._max_episode_steps - 1
        data = d4rl.qlearning_dataset(gym.make(name))
        if is_hopper_medium:
            for k in data:
                data[k] = np.delete(data[k], illed_idx_hopper_medium, axis=0)
        if is_hopper_med_expert:
            for k in data:
                data[k] = np.delete(data[k], illed_idx_hopper_medium_exp, axis=0)
    get_policy = policy_hook if policy_hook else None
    get_value = value_hook if value_hook else None
    data['rewards'] = np.expand_dims(data['rewards'], axis=1)
    data['terminals'] = np.expand_dims(data['terminals'], axis=1)
    data['last_actions'] = np.concatenate((np.zeros((1, data['actions'].shape[1]), dtype=np.float32), data['actions'][:-1, :]),
                                          axis=0).copy()
    data['first_step'] = np.zeros_like(data['terminals'])
    data['end_step'] = np.zeros_like(data['terminals'])
    data['valid'] = np.ones_like(data['terminals'])
    print('[ DEBUG ] reset_hidden_state: key in data: {}'.format(list(data.keys())))
    max_traj_len = -1
    last_start = 0
    traj_num = 1
    traj_lens = []
    print('[ DEBUG ] reset_hidden_state, obs shape: ', data['observations'].shape)
    for i in range(data['observations'].shape[0]):
        flag = True
        if i >= 1:
            flag = (data['observations'][i] == data['next_observations'][i - 1]).all()
            if data['terminals'][i - 1]:
                flag = False
        if not flag:
            data['last_actions'][i, :] = 0
            data['first_step'][i, :] = 1
            data['end_step'][i - 1, :] = 1
            if data['terminals'][i - 1]:
                data['next_observations'][i-1] = data['observations'][i-1]
            traj_len = i - last_start
            max_traj_len = max(max_traj_len, traj_len)
            last_start = i
            traj_num += 1
            traj_lens.append(traj_len)
            if traj_len > 1000:
                print('[ DEBUG + WARN ] reset_hidden_state: trajectory length is too large: current step is ', i, traj_num, )
    traj_lens.append(data['observations'].shape[0] - last_start)
    assert len(traj_lens) == traj_num
    data['policy_hidden'] = None # np.zeros((data['last_actions'].shape[0], policy_hidden.shape[-1]))
    data['value_hidden'] = None # np.zeros((data['last_actions'].shape[0], value_hidden.shape[-1]))
    last_start_ind = 0
    traj_num_to_infer = 400

    for i_ter in range(int(np.ceil(traj_num / traj_num_to_infer))):
        traj_lens_it = traj_lens[traj_num_to_infer * i_ter : min(traj_num_to_infer * (i_ter + 1), traj_num)]
        states = np.zeros((len(traj_lens_it), max_traj_len, data['observations'].shape[-1]), dtype=np.float32)
        actions = np.zeros((len(traj_lens_it), max_traj_len, data['actions'].shape[-1]), dtype=np.float32)
        lst_actions = np.zeros((len(traj_lens_it), max_traj_len, data['last_actions'].shape[-1]), dtype=np.float32)
        start_ind = last_start_ind
        for ind, item in enumerate(traj_lens_it):
            states[ind, :item] = data['observations'][start_ind:(start_ind+item)]
            lst_actions[ind, :item] = data['last_actions'][start_ind:(start_ind+item)]
            actions[ind, :item] = data['actions'][start_ind:(start_ind+item)]
            start_ind += item
        print('[ DEBUG ] reset_hidden_state size of total env states: {}, actions: {}'.format(states.shape, actions.shape))
        with torch.no_grad():
            policy_hidden_out = get_policy.get_hidden(torch.from_numpy(states).to(device), \
                                                      torch.from_numpy(lst_actions).to(device), torch.IntTensor(traj_lens_it))
            value_hidden_out = get_value.get_hidden(torch.from_numpy(states).to(device),\
                                                    torch.from_numpy(lst_actions).to(device), torch.IntTensor(traj_lens_it))
        policy_hidden = np.concatenate((np.zeros((len(traj_lens_it), 1, policy_hidden_out.shape[-1]), dtype=np.float32),
                                        policy_hidden_out[:, :-1].cpu().detach().numpy()), axis=-2)
        value_hidden = np.concatenate((np.zeros((len(traj_lens_it), 1, value_hidden_out.shape[-1]), dtype=np.float32),
                                        value_hidden_out[:, :-1].cpu().detach().numpy()), axis=-2)

        start_ind = last_start_ind
        for ind, item in enumerate(traj_lens_it):
            if data['policy_hidden'] is None:
                data['policy_hidden'] = np.zeros((data['last_actions'].shape[0], policy_hidden.shape[-1]), dtype=np.float32)
                data['value_hidden'] = np.zeros((data['last_actions'].shape[0], value_hidden.shape[-1]), dtype=np.float32)
            data['policy_hidden'][start_ind:(start_ind + item)] = policy_hidden[ind, :item]
            data['value_hidden'][start_ind:(start_ind + item)] = value_hidden[ind, :item]
            start_ind += item
        last_start_ind = start_ind
    print('[ DEBUG ] reset_hidden_state: inferring hidden state done')
    data_new = {'policy_hidden': data['policy_hidden'],
            'value_hidden': data['value_hidden']}
    data_adapt = {k: [] for k in data_new}
    it_traj = {k: [] for k in data_new}
    current_len = 0
    data_target = {k: replay_pool.fields[k] for k in data_new}
    traj_target_ind = 0
    mini_target_ind = 0
    for k in data_new:
        data_target[k][traj_target_ind, :] = 0
    for i in range(data_new['policy_hidden'].shape[0]):
        for k in data_new:
            data_target[k][traj_target_ind, mini_target_ind, :] = data_new[k][i]
        mini_target_ind += 1
        if data['end_step'][i] or mini_target_ind == maxlen:
            traj_target_ind += 1
            traj_target_ind = traj_target_ind % replay_pool._max_size
            mini_target_ind = 0
            for k in data_new:
                data_target[k][traj_target_ind, :] = 0

def restore_pool_softlearning(replay_pool, experiment_root, max_size, save_path=None):
    print('[ mopo/off_policy ] Loading SAC replay pool from: {}'.format(experiment_root))
    experience_paths = [
        checkpoint_dir
        for checkpoint_dir in sorted(glob.iglob(
            os.path.join(experiment_root, 'checkpoint_*')))
    ]

    checkpoint_epochs = [int(path.split('_')[-1]) for path in experience_paths]
    checkpoint_epochs = sorted(checkpoint_epochs)
    if max_size == 250e3:
        checkpoint_epochs = checkpoint_epochs[2:]

    for epoch in checkpoint_epochs:
        fullpath = os.path.join(experiment_root, 'checkpoint_{}'.format(epoch), 'replay_pool.pkl')
        print('[ mopo/off_policy ] Loading replay pool data: {}'.format(fullpath))
        replay_pool.load_experience(fullpath)
        if replay_pool.size >= max_size:
            break

    if save_path is not None:
        size = replay_pool.size
        stat_path = os.path.join(save_path, 'pool_stat_{}.pkl'.format(size))
        save_path = os.path.join(save_path, 'pool_{}.pkl'.format(size))
        d = {}
        for key in replay_pool.fields.keys():
            d[key] = replay_pool.fields[key][:size]

        num_paths = 0
        temp = 0
        path_end_idx = []
        for i in range(d['terminals'].shape[0]):
            if d['terminals'][i] or i - temp + 1 == 1000:
                num_paths += 1
                temp = i + 1
                path_end_idx.append(i)
        total_return = d['rewards'].sum()
        avg_return = total_return / num_paths
        buffer_max, buffer_min = -np.inf, np.inf
        path_return = 0.0
        for i in range(d['rewards'].shape[0]):
            path_return += d['rewards'][i]
            if i in path_end_idx:
                if path_return > buffer_max:
                    buffer_max = path_return
                if path_return < buffer_min:
                    buffer_min = path_return
                path_return = 0.0

        print('[ mopo/off_policy ] Replay pool average return is {}, buffer_max is {}, buffer_min is {}'.format(avg_return, buffer_max, buffer_min))
        d_stat = dict(avg_return=avg_return, buffer_max=buffer_max, buffer_min=buffer_min)
        pickle.dump(d_stat, open(stat_path, 'wb'))

        print('[ mopo/off_policy ] Saving replay pool to: {}'.format(save_path))
        pickle.dump(d, open(save_path, 'wb'))


def restore_pool_bear(replay_pool, load_path):
    print('[ mopo/off_policy ] Loading BEAR replay pool from: {}'.format(load_path))
    data = pickle.load(gzip.open(load_path, 'rb'))
    num_trajectories = data['terminals'].sum() or 1000
    avg_return = data['rewards'].sum() / num_trajectories
    print('[ mopo/off_policy ] {} trajectories | avg return: {}'.format(num_trajectories, avg_return))

    for key in ['log_pis', 'data_policy_mean', 'data_policy_logvar']:
        del data[key]

    replay_pool.add_samples(data)


def restore_pool_contiguous(replay_pool, load_path):
    print('[ mopo/off_policy ] Loading contiguous replay pool from: {}'.format(load_path))
    import numpy as np
    data = np.load(load_path)

    state_dim = replay_pool.fields['observations'].shape[1]
    action_dim = replay_pool.fields['actions'].shape[1]
    expected_dim = state_dim + action_dim + state_dim + 1 + 1
    actual_dim = data.shape[1]
    assert expected_dim == actual_dim, 'Expected {} dimensions, found {}'.format(expected_dim, actual_dim)

    dims = [state_dim, action_dim, state_dim, 1, 1]
    ends = []
    current_end = 0
    for d in dims:
        current_end += d
        ends.append(current_end)
    states, actions, next_states, rewards, dones = np.split(data, ends, axis=1)[:5]
    replay_pool.add_samples({
        'observations': states,
        'actions': actions,
        'next_observations': next_states,
        'rewards': rewards,
        'terminals': dones.astype(bool)
    })

def get_illed_med_exp():
    return [9290, 15278, 16276, 19270, 26256, 33242, 34240, 35238, 43869, 46863,
            52851, 55845, 68547, 69545, 70543, 74535, 75533, 79525, 80523, 81521, 82519, 85513,
            86511, 87509, 92499, 95493, 96491, 98487, 99485, 101481, 102479, 109465, 112808, 113806,
            118783, 119781, 129761, 133753, 140739, 149721, 150719, 152715, 154711, 156707, 160345, 165335,
            172618, 174152, 184525, 189831, 193823, 200809, 203803, 208793, 209791, 210789, 220654, 221652,
            226642, 235609, 241597, 244591, 251577, 253573, 255569, 259561, 261557, 263553, 279411, 280409,
            281407, 285399, 287395, 288393, 290389, 291387, 295379, 296377, 298373, 300369, 303363, 319023,
            320021, 326009, 327007, 334715, 337709, 345082, 347078, 352440, 355434, 356432, 370963, 374955,
            375953, 376951, 377949, 382939, 385933, 386931, 389925, 392919, 394915, 402899, 404895, 409885,
            414875, 416871, 417869, 418867, 421861, 423857, 429845, 430843, 433837, 445108, 446106, 450098,
            455999, 457995, 465903, 466901, 468897, 469895, 470893, 473887, 477879, 480873, 483867, 485863,
            489855, 495229, 497965, 501271, 502269, 505263, 507259, 509255, 511251, 512249, 514245, 517239,
            518237, 525223, 532209, 533207, 534205, 538197, 540193, 541191, 547179, 548177, 552169, 556161,
            563147, 564145, 565143, 567139, 570133, 572129, 574125, 578117, 582590, 583588, 586582, 587580,
            591572, 597159, 600154, 608138, 610134, 612130, 615848, 616846, 627824, 629820, 634810, 635808,
            636806, 645612, 646610, 649604, 650602, 655421, 659413, 660411, 663055, 670334, 682873, 690857,
            691855, 698841, 701830, 704824, 707818, 710812, 715529, 721517, 734036, 739026, 743018, 748008,
            760402, 763396, 766390, 767388, 772378, 775691, 776689, 779683, 781679, 786410, 788406, 789404,
            793396, 803376, 805372, 810362, 814354, 823336, 828326, 836310, 838306, 842298, 843296, 850289,
            857275, 858273, 859271, 860269, 863096, 871822, 876812, 877810, 878808, 885359, 886357, 894726,
            900608, 901606, 903602, 904600, 910588, 921906, 926896, 927894, 930888, 936876, 942219, 948207,
            950203, 953197, 955193, 960183, 971309, 972307, 976631, 980623, 981621, 985613, 986611, 989605,
            992599, 993597, 997589, 1090448, 1116281, 1117755, 1118649, 1118888, 1119536, 1119979, 1120853,
            1122660, 1125857, 1127473, 1138991, 1140216, 1140855, 1142213,
            1146000, 1147056, 1147470, 1151830, 1152150, 1154541, 1155304, 1155735, 1156261, 1157092, 1158497,
            1161298, 1162988,
            1163795, 1168075, 1168768, 1171017, 1172762, 1176232, 1178077, 1178961, 1192387]

