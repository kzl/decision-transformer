nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR2 --env hopper --dataset medium --mode normal --num_eval_episodes 10 --model_type cqlR2 --device cuda:0 > output/hopper_medium_normal_cqlR2.log 2>&1 &
nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR2 --env halfcheetah --dataset medium --mode normal --num_eval_episodes 10 --model_type cqlR2 --device cuda:0 > output/halfcheetah_medium_normal_cqlR2.log 2>&1 &
nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR2 --env walker2d --dataset medium --mode normal --num_eval_episodes 10 --model_type cqlR2 --device cuda:1 > output/walker2d_medium_normal_cqlR2.log 2>&1 &

nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR2 --env hopper --dataset medium-replay --mode normal --num_eval_episodes 10 --model_type cqlR2 --device cuda:1 --num_eval_episodes 1 > output/hopper_medium_replay_normal_cqlR2.log 2>&1 &
nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR2 --env halfcheetah --dataset medium-replay --mode normal --num_eval_episodes 10 --model_type cqlR2 --device cuda:2 > output/halfcheetah_medium_replay_normal_cqlR2.log 2>&1 &
nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR2 --env walker2d --dataset medium-replay --mode normal --num_eval_episodes 10 --model_type cqlR2 --device cuda:2 > output/walker2d_medium_replay_normal_cqlR2.log 2>&1 &

nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR2 --env hopper --dataset expert --mode normal --num_eval_episodes 10 --model_type cqlR2 --device cuda:3 > output/hopper_expert_normal_cqlR2.log 2>&1 &
nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR2 --env halfcheetah --dataset expert --mode normal --num_eval_episodes 10 --model_type cqlR2 --device cuda:3 --num_eval_episodes 1 > output/halfcheetah_expert_normal_cqlR2.log 2>&1 &
nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR2 --env walker2d --dataset expert --mode normal --num_eval_episodes 10 --model_type cqlR2 --device cuda:4 > output/walker2d_expert_normal_cqlR2.log 2>&1 &

