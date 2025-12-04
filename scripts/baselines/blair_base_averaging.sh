#!/bin/bash

# You can change model_type to blair_large

python merge_test.py \
    --model_type blair_base \
    --data_paths \
      datasets/Arts \
      datasets/Beauty \
      datasets/Instruments \
      datasets/Office \
      datasets/Pantry \
      datasets/Scientific \
      datasets/Sports \
      datasets/Toys \
    --finetune_checkpoint_paths \
      checkpoints/blair_base/Arts/state_dict.pt \
      checkpoints/blair_base/Beauty/state_dict.pt \
      checkpoints/blair_base/Instruments/state_dict.pt \
      checkpoints/blair_base/Office/state_dict.pt \
      checkpoints/blair_base/Pantry/state_dict.pt \
      checkpoints/blair_base/Scientific/state_dict.pt \
      checkpoints/blair_base/Sports/state_dict.pt \
      checkpoints/blair_base/Toys/state_dict.pt \
    --batch_size 32 \
    --train_data_split test \
    --test_data_split test \
    --merge_type task_vector \
    --learn_type task_wise \
    --weight_file average \
    --metrics_path blair_base_averaging.csv
