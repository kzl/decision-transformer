# Decision Transformer (DT)
for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 50 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
done

# Behavior Cloning (BC)
for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 50 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
done