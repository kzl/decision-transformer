
# Atari

We build our Atari implementation on top of [minGPT](https://github.com/karpathy/minGPT) and benchmark our results on the [DQN-replay](https://github.com/google-research/batch_rl) dataset. 

## Installation

Dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets

Create a directory for the dataset and load the dataset using [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install). Replace `[DIRECTORY_NAME]` and `[GAME_NAME]` accordingly (e.g., `./dqn_replay` for `[DIRECTORY_NAME]` and `Breakout` for `[GAME_NAME]`)
```
mkdir [DIRECTORY_NAME]
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] [DIRECTORY_NAME]
```

## Example usage

Scripts to reproduce our Decision Transformer results can be found in `run.sh`.

```
python run_dt_atari.py --seed 123 --block_size 90 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128 --data_dir_prefix [DIRECTORY_NAME]
```
