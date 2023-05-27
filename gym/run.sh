# QDT
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python -u experiment.py --env hopper --max_iters 30 --dataset medium --model_type qdt --device cuda:3 > output/hopper_medium_normal_qdt.logs 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python -u experiment.py --env halfcheetah --max_iters 30 --dataset medium --model_type qdt --device cuda:0 > output/halfcheetah_medium_normal_qdt.logs 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python -u experiment.py --env walker2d --max_iters 30 --dataset medium --model_type qdt --device cuda:0 > output/walker2d_medium_normal_qdt.logs 2>&1 &

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python -u experiment.py --env hopper --max_iters 30 --dataset expert --model_type qdt --device cuda:1 > output/hopper_expert_normal_qdt.logs 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python -u experiment.py --env halfcheetah --max_iters 30 --dataset expert --model_type qdt --device cuda:1 > output/halfcheetah_expert_normal_qdt.logs 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python -u experiment.py --env walker2d --max_iters 30 --dataset expert --model_type qdt --device cuda:1 > output/walker2d_expert_normal_qdt.logs 2>&1 &

# DT
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python -u experiment.py --env hopper --max_iters 30 --dataset medium --model_type dt --device cuda:2 > output/hopper_medium_normal_dt.logs 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python -u experiment.py --env halfcheetah --max_iters 30 --dataset medium --model_type dt --device cuda:2 > output/halfcheetah_medium_normal_dt.logs 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python -u experiment.py --env walker2d --max_iters 30 --dataset medium --model_type dt --device cuda:2 > output/walker2d_medium_normal_dt.logs 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python -u experiment.py --env hopper --max_iters 30 --dataset expert --model_type dt --device cuda:3 > output/hopper_expert_normal_dt.logs 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python -u experiment.py --env halfcheetah --max_iters 30 --dataset expert --model_type dt --device cuda:3 > output/halfcheetah_expert_normal_dt.logs 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python -u experiment.py --env walker2d --max_iters 30 --dataset expert --model_type dt --device cuda:3 > output/walker2d_expert_normal_dt.logs 2>&1 &




#nohup python experiment.py --env hopper --dataset medium --model_type dt --max_iters 30 --device cuda:2 > output/hopper_medium_normal_dt.logs 2>&1 &
#nohup python experiment.py --env halfcheetah --dataset medium --model_type dt --max_iters 30 --device cuda:3 > output/halfcheetah_medium_normal_dt.logs 2>&1 &
#nohup python experiment.py --env walker2d --dataset medium --model_type dt --max_iters 30 --device cuda:4 > output/walker2d_medium_normal_dt.logs 2>&1 &
#
#nohup python experiment.py --env hopper --dataset medium --mode delayed --model_type dt --max_iters 30 --device cuda:2 > output/hopper_medium_normal_dt.logs 2>&1 &
#nohup python experiment.py --env halfcheetah --dataset medium --mode delayed --model_type dt --max_iters 30 --device cuda:3 > output/halfcheetah_medium_normal_dt.logs 2>&1 &
#nohup python experiment.py --env walker2d --dataset medium --mode delayed --model_type dt --max_iters 30 --device cuda:4 > output/walker2d_medium_normal_dt.logs 2>&1 &
#
#nohup python experiment.py --env hopper --dataset medium-replay --model_type dt --max_iters 30 --device cuda:2 > output/hopper_medium_normal_dt.logs 2>&1 &
#nohup python experiment.py --env halfcheetah --dataset medium-replay --model_type dt --max_iters 30 --device cuda:3 > output/halfcheetah_medium_normal_dt.logs 2>&1 &
#nohup python experiment.py --env walker2d --dataset medium-replay --model_type dt --max_iters 30 --device cuda:4 > output/walker2d_medium_normal_dt.logs 2>&1 &
#
#nohup python experiment.py --env hopper --dataset medium-replay --mode delayed --model_type dt --max_iters 30 --device cuda:2 > output/hopper_medium_normal_dt.logs 2>&1 &
#nohup python experiment.py --env halfcheetah --dataset medium-replay --mode delayed --model_type dt --max_iters 30 --device cuda:3 > output/halfcheetah_medium_normal_dt.logs 2>&1 &
#nohup python experiment.py --env walker2d --dataset medium-replay --mode delayed --model_type dt --max_iters 30 --device cuda:4 > output/walker2d_medium_normal_dt.logs 2>&1 &
#
#nohup python experiment.py --env walker2d --dataset medium-replay --mode delayed --model_type dt --max_iters 30 --device cuda:4 > output/walker2d_medium_normal_dt.logs 2>&1 &

#nohup python decision_transformer/models/run_cql.py --algo_name cql --env hopper --dataset medium --mode normal --num_eval_episodes 10 --model_type CQL --device cuda:4 > output/hopper_medium_normal_cql.log 2>&1 &
#nohup python decision_transformer/models/run_cql.py --algo_name cql --env halfcheetah --dataset medium --mode normal --num_eval_episodes 10 --model_type CQL --device cuda:4 > output/halfcheetah_medium_normal_cql.log 2>&1 &
#nohup python decision_transformer/models/run_cql.py --algo_name cql --env walker2d --dataset medium --mode normal --num_eval_episodes 10 --model_type CQL --device cuda:4 > output/walker2d_medium_normal_cql.log 2>&1 &

#nohup python decision_transformer/models/run_cql.py --algo_name cql --env hopper --dataset medium-expert --mode normal --num_eval_episodes 10 --model_type CQL --device cuda:4 > output/hopper_medium_expert_normal_cql.log 2>&1 &
#nohup python decision_transformer/models/run_cql.py --algo_name cql --env halfcheetah --dataset medium-expert --mode normal --num_eval_episodes 10 --model_type CQL --device cuda:4 > output/halfcheetah_medium_expert_normal_cql.log 2>&1 &
#nohup python decision_transformer/models/run_cql.py --algo_name cql --env walker2d --dataset medium-expert --mode normal --num_eval_episodes 10 --model_type CQL --device cuda:4 > output/walker2d_medium_expert_normal_cql.log 2>&1 &

#nohup python decision_transformer/models/run_cql.py --algo_name cql --env hopper --dataset expert --mode normal --num_eval_episodes 10 --model_type CQL --device cuda:4 > output/hopper_expert_normal_cql.log 2>&1 &
#nohup python decision_transformer/models/run_cql.py --algo_name cql --env halfcheetah --dataset expert --mode normal --num_eval_episodes 10 --model_type CQL --device cuda:4 > output/halfcheetah_expert_normal_cql.log 2>&1 &
#nohup python decision_transformer/models/run_cql.py --algo_name cql --env walker2d --dataset expert --mode normal --num_eval_episodes 10 --model_type CQL --device cuda:4 > output/walker2d_expert_normal_cql.log 2>&1 &

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python decision_transformer/models/examples.py --algo_name cql --env hopper --dataset medium-replay --mode normal --num_eval_episodes 10 --model_type CQL --device cuda:1 > output/hopper_medium_replay_normal_cql.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python decision_transformer/models/examples.py --algo_name cql --env halfcheetah --dataset medium-replay --mode normal --num_eval_episodes 10 --model_type CQL --device cuda:1 > output/halfcheetah_medium_replay_normal_cql.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup python decision_transformer/models/examples.py --algo_name cql --env walker2d --dataset medium-replay --mode normal --num_eval_episodes 10 --model_type CQL --device cuda:1 > output/walker2d_medium_replay_normal_cql.log 2>&1 &

