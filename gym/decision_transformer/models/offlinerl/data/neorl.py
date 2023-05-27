import numpy as np
from loguru import logger

from offlinerl.utils.data import SampleBatch
from offlinerl.utils.data import BufferDataset, BufferDataloader

def load_neorl_buffer(data):
    buffer = SampleBatch(
        obs = data["obs"],
        obs_next = data["next_obs"],
        act = data["action"],
        rew = data["reward"],
        done = data["done"],
    )

    logger.info('obs shape: {}', buffer.obs.shape)
    logger.info('obs_next shape: {}', buffer.obs_next.shape)
    logger.info('act shape: {}', buffer.act.shape)
    logger.info('rew shape: {}', buffer.rew.shape)
    logger.info('done shape: {}', buffer.done.shape)
    logger.info('Episode reward: {}', buffer.rew.sum() /np.sum(buffer.done) )
    logger.info('Number of terminals on: {}', np.sum(buffer.done))
    
    """
    rew_scaler = get_scaler(buffer.rew)
    buffer.rew = rew_scaler.transform(buffer.rew)
    buffer.rew =  buffer.rew * 0.01
    buffer.done[buffer.rew < np.sort(buffer.rew.reshape(-1))[int(len(buffer)*0.01)]] = 1
    
    buffer = BufferDataset(buffer)
    buffer = BufferDataloader(buffer, batch_size=1, collate_fn=lambda x: x[0], num_workers=8)
    """
    
    return buffer
