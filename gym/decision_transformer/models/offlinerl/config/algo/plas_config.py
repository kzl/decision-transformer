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

vae_iterations = 500000
vae_hidden_size = 750
vae_batch_size = 100
vae_kl_weight = 0.5

latent = True
layer_num = 2
actor_batch_size = 100
hidden_layer_size = 256
actor_iterations = 500000
vae_lr = 1e-4
actor_lr = 1e-4
critic_lr = 1e-3
soft_target_tau = 0.005
lmbda = 0.75
discount = 0.99

max_latent_action = 2 
phi = 0.05

#tune
params_tune = {
    "vae_iterations" : {"type" : "discrete", "value":[50000, 100000, 500000,]},
    "actor_lr" : {"type" : "continuous", "value":[1E-4, 1E-3]},
    "vae_lr" : {"type" : "continuous", "value":[1E-4, 1E-3]},
    "actor_batch_size" : {"type": "discrete", "value":[128, 256, 512]},
    "latent" : {"type": "discrete", "value":[True, False]},
    "lmbda" :{"type": "discrete", "value":[0.65, 0.75, 0.85]},
}

#tune
grid_tune = {
    "phi" : [0, 0.05, 0.1, 0.2, 0.4],
}
