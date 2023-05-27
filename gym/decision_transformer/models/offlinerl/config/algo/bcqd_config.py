import torch
from offlinerl.utils.exp import select_free_cuda

task = "Hopper-v3"
task_data_type = "low"
task_train_num = 99

seed = 42

device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None


max_timesteps = 1e6
eval_freq = 1e3

optimizer_parameters = {
    "lr": 3e-4,
    }

BCQ_threshold = 0.3

discount = 0.99
tau = 0.005
polyak_target_update = True
target_update_frequency=1
start_timesteps = 1e3
initial_eps = 0.1
end_eps = 0.1
eps_decay_period = 1
eval_eps = 0.001
buffer_size = 1e6
batch_size = 256
train_freq = 1