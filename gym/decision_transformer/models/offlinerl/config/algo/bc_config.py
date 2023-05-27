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

actor_features = 256
actor_layers = 2

batch_size = 256
steps_per_epoch = 1000
max_epoch = 100

actor_lr = 1e-3

#tune
params_tune = {
    "actor_lr" : {"type" : "continuous", "value": [1e-4, 1e-3]},
}

#tune
grid_tune = {
    "actor_lr" : [1e-3],
}
