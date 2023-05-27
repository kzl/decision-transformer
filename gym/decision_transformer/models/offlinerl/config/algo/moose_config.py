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
#vae_pretrain_model = "/tmp/vae_499999.pkl"


latent = False
layer_num = 3
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
    "vae_iterations" : {"type" : "continuous", "value":[50000, 100000, 500000,]},
    "actor_lr" : {"type" : "continuous", "value":[1E-4, 1E-3]},
    "vae_lr" : {"type" : "continuous", "value":[1E-4, 1E-3]},
    "lmbda" :{"type": "discrete", "value":[0.0, 0.25, 0.5, 0.75, 1.0]},
}
