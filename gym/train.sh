# Original 
# python -O experiment.py --env hopper --dataset medium --model_type dt
# python -O experiment.py --env hopper --dataset medium --model_type dt --num_steps_per_iter 100

# K = sequence length
# https://stackoverflow.com/questions/60351790/running-python3-with-debug-set-to-off-with-o
python -O experiment.py --env boyan13 --dataset medium --model_type dt --K 4 --num_steps_per_iter 100