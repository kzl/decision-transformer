import torch
from offlinerl.utils.exp import select_free_cuda

task = "Hopper-v3"
task_data_type = "low"
task_train_num = 99

seed = 42
device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None

dynamics_path = None
behavior_path = None

transition_hidden_size = 256
transition_hidden_layers = 4
transition_init_num = 7
transition_select_num = 5

actor_hidden_size = 256
actor_hidden_layers = 2
value_hidden_size = 256
value_hidden_layers = 2

transition_batch_size = 256
data_collection_per_epoch = 50000
max_epoch = 250
trpo_steps_per_epoch = 25

bc_batch_size = 256
bc_init = True

transition_lr = 1e-3
bc_lr = 1e-3
value_lr = 3e-4

cg_iters = 10
damping_coeff = 0.1
backtrack_iters = 10
backtrack_coeff = 0.8
train_v_iters = 50
trpo_step_size = 0.01
explore_mode = 'sample'
static_noise = 0.1

horizon = 250
gamma = 0.99
lam = 0.95

#tune
params_tune = {
    "horizon" : {"type" : "discrete", "value": [250, 500, 1000]}
}

#tune
grid_tune = {
    'horizon' : [250, 1000],
    # 'trpo_step_size' : [0.01, 0.05],
    'explore_mode' : ['sample', 'static'],
}
