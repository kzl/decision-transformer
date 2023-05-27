# QDT2
nohup python -u experiment.py --env hopper --max_iters 30 --dataset medium --model_type qdt2 --device cuda:3 > output/hopper_medium_normal_qdt2.log 2>&1 &
nohup python -u experiment.py --env halfcheetah --max_iters 30 --dataset medium --model_type qdt2 --device cuda:3 > output/halfcheetah_medium_normal_qdt2.log 2>&1 &
nohup python -u experiment.py --env walker2d --max_iters 30 --dataset medium --model_type qdt2 --device cuda:3 > output/walker2d_medium_normal_qdt2.log 2>&1 &

nohup python -u experiment.py --env hopper --max_iters 30 --dataset medium-replay --model_type qdt2 --device cuda:4 > output/hopper_medium_replay_normal_qdt2.log 2>&1 &
nohup python -u experiment.py --env halfcheetah --max_iters 30 --dataset medium-replay --model_type qdt2 --device cuda:4 > output/halfcheetah_medium_replay_normal_qdt2.log 2>&1 &
nohup python -u experiment.py --env walker2d --max_iters 30 --dataset medium-replay --model_type qdt2 --device cuda:4 > output/walker2d_medium_replay_normal_qdt2.log 2>&1 &

nohup python -u experiment.py --env hopper --max_iters 30 --dataset expert --model_type qdt2 --device cuda:5 > output/hopper_expert_normal_qdt2.log 2>&1 &
nohup python -u experiment.py --env halfcheetah --max_iters 30 --dataset expert --model_type qdt2 --device cuda:5 > output/halfcheetah_expert_normal_qdt2.log 2>&1 &
nohup python -u experiment.py --env walker2d --max_iters 30 --dataset expert --model_type qdt2 --device cuda:5 > output/walker2d_expert_normal_qdt2.log 2>&1 &