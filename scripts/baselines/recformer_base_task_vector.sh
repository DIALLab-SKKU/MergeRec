#!/bin/bash

WEIGHT=$1  # This is the merging weight to be used

# You can change model_type to recformer_large
# Run example: bash scripts/baselines/recformer_base_task_vector.sh 0.3
# Replace `--weight_file task_vector` with ties to use different merging methods

python merge_test.py \
    --model_type recformer_base \
    --model_kwargs ckpt_path checkpoints/pretrained/recformer_base_pretrained.pt \
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
      checkpoints/recformer_base/Arts/42/state_dict.pt \
      checkpoints/recformer_base/Beauty/42/state_dict.pt \
      checkpoints/recformer_base/Instruments/42/state_dict.pt \
      checkpoints/recformer_base/Office/42/state_dict.pt \
      checkpoints/recformer_base/Pantry/42/state_dict.pt \
      checkpoints/recformer_base/Scientific/42/state_dict.pt \
      checkpoints/recformer_base/Sports/42/state_dict.pt \
      checkpoints/recformer_base/Toys/42/state_dict.pt \
    --batch_size 32 \
    --train_data_split test \
    --test_data_split test \
    --merge_type task_vector \
    --learn_type task_wise \
    --weight_file uniform \
    --weight_file_line ${WEIGHT} \
    --metrics_path recformer_base_task_vector_${WEIGHT}.csv

