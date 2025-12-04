#!/bin/bash

# You can change model_type to recformer_large.
# Change the ckpt_path to the corresponding pretrained model checkpoint.

python finetune_train.py \
    --model_type recformer_base \
    --model_kwargs ckpt_path checkpoints/pretrained/recformer_base_pretrained.pt \
    --batch_size 64 \
    --negative_sample.in_batch \
    --temperature 0.05 \
    --warmup_steps 100 \
    --data_path datasets/Arts \
    --learning_rate 5e-5 \
    --log_every_n_steps 1

