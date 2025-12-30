#!/bin/bash

# You can change model_type to blair_large

python finetune_train.py \
    --model_type blair_base \
    --batch_size 64 \
    --negative_sample.in_batch \
    --temperature 0.05 \
    --warmup_steps 100 \
    --data_path datasets/Arts \
    --learning_rate 5e-5 \
    --log_every_n_steps 1
