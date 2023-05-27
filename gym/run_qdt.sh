# QDT
nohup python -u experiment.py --env hopper --max_iters 30 --dataset medium --model_type qdt --device cuda:0 > output/hopper_medium_normal_qdt.log 2>&1 &
nohup python -u experiment.py --env halfcheetah --max_iters 30 --dataset medium --model_type qdt --device cuda:0 > output/halfcheetah_medium_normal_qdt.log 2>&1 &
nohup python -u experiment.py --env walker2d --max_iters 30 --dataset medium --model_type qdt --device cuda:0 > output/walker2d_medium_normal_qdt.log 2>&1 &

nohup python -u experiment.py --env hopper --max_iters 30 --dataset medium-replay --model_type qdt --device cuda:1 > output/hopper_medium_replay_normal_qdt.log 2>&1 &
nohup python -u experiment.py --env halfcheetah --max_iters 30 --dataset medium-replay --model_type qdt --device cuda:1 > output/halfcheetah_medium_replay_normal_qdt.log 2>&1 &
nohup python -u experiment.py --env walker2d --max_iters 30 --dataset medium-replay --model_type qdt --device cuda:1 > output/walker2d_medium_replay_normal_qdt.log 2>&1 &

nohup python -u experiment.py --env hopper --max_iters 30 --dataset expert --model_type qdt --device cuda:2 > output/hopper_expert_normal_qdt.log 2>&1 &
nohup python -u experiment.py --env halfcheetah --max_iters 30 --dataset expert --model_type qdt --device cuda:2 > output/halfcheetah_expert_normal_qdt.log 2>&1 &
nohup python -u experiment.py --env walker2d --max_iters 30 --dataset expert --model_type qdt --device cuda:2 > output/walker2d_expert_normal_qdt.log 2>&1 &