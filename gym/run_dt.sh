# DT
CUDA_VISIBLE_DEVICES=0 nohup python -u experiment.py --env hopper --max_iters 30 --dataset medium --model_type dt --device cuda:0 > output/hopper_medium_normal_dt.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1 nohup python -u experiment.py --env halfcheetah --max_iters 30 --dataset medium --model_type dt --device cuda:0 > output/halfcheetah_medium_normal_dt.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2 nohup python -u experiment.py --env walker2d --max_iters 30 --dataset medium --model_type dt --device cuda:1 > output/walker2d_medium_normal_dt.log 2>&1 &

nohup python -u experiment.py --env hopper --max_iters 30 --dataset medium-replay --model_type dt --device cuda:1 > output/hopper_medium_replay_normal_dt.log 2>&1 &
nohup python -u experiment.py --env halfcheetah --max_iters 30 --dataset medium-replay --model_type dt --device cuda:1 > output/halfcheetah_medium_replay_normal_dt.log 2>&1 &
nohup python -u experiment.py --env walker2d --max_iters 30 --dataset medium-replay --model_type dt --device cuda:1 > output/walker2d_medium_replay_normal_dt.log 2>&1 &

nohup python -u experiment.py --env hopper --max_iters 30 --dataset expert --model_type dt --device cuda:3 > output/hopper_expert_normal_dt.log 2>&1 &
nohup python -u experiment.py --env halfcheetah --max_iters 30 --dataset expert --model_type dt --device cuda:4 > output/halfcheetah_expert_normal_dt.log 2>&1 &
nohup python -u experiment.py --env walker2d --max_iters 30 --dataset expert --model_type dt --device cuda:5 > output/walker2d_expert_normal_dt.log 2>&1 &