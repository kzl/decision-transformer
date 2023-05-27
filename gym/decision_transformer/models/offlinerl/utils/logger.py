import os
import uuid
# import aim

from offlinerl.utils.io import create_dir

def log_path():
    import offlinerl
    log_path = os.path.abspath(os.path.join(offlinerl.__file__,"../../","offlinerl_tmp"))

    create_dir(log_path)

    return log_path
    
"""
class exp_logger():
    def __init__(self, experiment_name=None,flush_frequency=1):
        print("experiment_name:",experiment_name)
        self.aim_logger = aim.Session(experiment=experiment_name, flush_frequency=flush_frequency)
    
    def log_hparams(self, hparams_dict):
        self.aim_logger.set_params(hparams_dict, name='hparams')
"""