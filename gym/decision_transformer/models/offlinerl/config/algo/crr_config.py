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

hidden_features = 256
hidden_layers = 2
atoms = 21

advantage_mode = 'mean'
weight_mode = 'exp'
advantage_samples = 4
beta = 1.0
gamma = 0.99

batch_size = 1024
steps_per_epoch = 1000
max_epoch = 200

lr = 1e-4
update_frequency = 100

#tune
params_tune = {
    "beta" : {"type" : "continuous", "value": [0.0, 10.0]},
}

#tune
grid_tune = {
    "advantage_mode" : ['mean', 'max'],
    "weight_mode" : ['exp', 'binary'],
}
