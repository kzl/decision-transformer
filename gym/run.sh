nohup python experiment.py --env hopper --dataset medium --model_type dt --max_iters 30 --device cuda:2 > output/hopper_medium_normal_dt.logs 2>&1 &
nohup python experiment.py --env halfcheetah --dataset medium --model_type dt --max_iters 30 --device cuda:3 > output/halfcheetah_medium_normal_dt.logs 2>&1 &
nohup python experiment.py --env walker2d --dataset medium --model_type dt --max_iters 30 --device cuda:4 > output/walker2d_medium_normal_dt.logs 2>&1 &

nohup python experiment.py --env hopper --dataset medium --mode delayed --model_type dt --max_iters 30 --device cuda:2 > output/hopper_medium_normal_dt.logs 2>&1 &
nohup python experiment.py --env halfcheetah --dataset medium --mode delayed --model_type dt --max_iters 30 --device cuda:3 > output/halfcheetah_medium_normal_dt.logs 2>&1 &
nohup python experiment.py --env walker2d --dataset medium --mode delayed --model_type dt --max_iters 30 --device cuda:4 > output/walker2d_medium_normal_dt.logs 2>&1 &

nohup python experiment.py --env hopper --dataset medium-replay --model_type dt --max_iters 30 --device cuda:2 > output/hopper_medium_normal_dt.logs 2>&1 &
nohup python experiment.py --env halfcheetah --dataset medium-replay --model_type dt --max_iters 30 --device cuda:3 > output/halfcheetah_medium_normal_dt.logs 2>&1 &
nohup python experiment.py --env walker2d --dataset medium-replay --model_type dt --max_iters 30 --device cuda:4 > output/walker2d_medium_normal_dt.logs 2>&1 &

nohup python experiment.py --env hopper --dataset medium-replay --mode delayed --model_type dt --max_iters 30 --device cuda:2 > output/hopper_medium_normal_dt.logs 2>&1 &
nohup python experiment.py --env halfcheetah --dataset medium-replay --mode delayed --model_type dt --max_iters 30 --device cuda:3 > output/halfcheetah_medium_normal_dt.logs 2>&1 &
nohup python experiment.py --env walker2d --dataset medium-replay --mode delayed --model_type dt --max_iters 30 --device cuda:4 > output/walker2d_medium_normal_dt.logs 2>&1 &