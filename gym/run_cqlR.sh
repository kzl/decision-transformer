nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR --env hopper --dataset medium --mode normal --num_eval_episodes 10 --model_type cqlR --device cuda:4 > output/hopper_medium_normal_cqlR.log 2>&1 &
nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR --env halfcheetah --dataset medium --mode normal --num_eval_episodes 10 --model_type cqlR --device cuda:4 > output/halfcheetah_medium_normal_cqlR.log 2>&1 &
nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR --env walker2d --dataset medium --mode normal --num_eval_episodes 10 --model_type cqlR --device cuda:4 > output/walker2d_medium_normal_cqlR.log 2>&1 &

nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR --env hopper --dataset medium-replay --mode normal --num_eval_episodes 10 --model_type cqlR --device cuda:3 --num_eval_episodes 1 > output/hopper_medium_replay_normal_cqlR.log 2>&1 &
nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR --env halfcheetah --dataset medium-replay --mode normal --num_eval_episodes 10 --model_type cqlR --device cuda:3 > output/halfcheetah_medium_replay_normal_cqlR.log 2>&1 &
nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR --env walker2d --dataset medium-replay --mode normal --num_eval_episodes 10 --model_type cqlR --device cuda:3 > output/walker2d_medium_replay_normal_cqlR.log 2>&1 &

nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR --env hopper --dataset expert --mode normal --num_eval_episodes 10 --model_type cqlR --device cuda:5 > output/hopper_expert_normal_cqlR.log 2>&1 &
nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR --env halfcheetah --dataset expert --mode normal --num_eval_episodes 10 --model_type cqlR --device cuda:5 --num_eval_episodes 1 > output/halfcheetah_expert_normal_cqlR.log 2>&1 &
nohup python -u decision_transformer/models/run_cql.py --algo_name cqlR --env walker2d --dataset expert --mode normal --num_eval_episodes 10 --model_type cqlR --device cuda:5 > output/walker2d_expert_normal_cqlR.log 2>&1 &

