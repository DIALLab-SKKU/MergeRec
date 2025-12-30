#!/bin/bash

# This is an example of:
# - merging blair_base models
# - with task_vector merge type
# - and task_wise merging coefficient
# You can change `model_type` to blair_large

python merge_train.py \
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
      checkpoints/blair_base/Arts/42/state_dict.pt \
      checkpoints/blair_base/Beauty/42/state_dict.pt \
      checkpoints/blair_base/Instruments/42/state_dict.pt \
      checkpoints/blair_base/Office/42/state_dict.pt \
      checkpoints/blair_base/Pantry/42/state_dict.pt \
      checkpoints/blair_base/Scientific/42/state_dict.pt \
      checkpoints/blair_base/Sports/42/state_dict.pt \
      checkpoints/blair_base/Toys/42/state_dict.pt \
    --item_embeddings_paths \
      checkpoints/blair_base/Arts/item_embedding.pt \
      checkpoints/blair_base/Beauty/item_embedding.pt \
      checkpoints/blair_base/Instruments/item_embedding.pt \
      checkpoints/blair_base/Office/item_embedding.pt \
      checkpoints/blair_base/Pantry/item_embedding.pt \
      checkpoints/blair_base/Scientific/item_embedding.pt \
      checkpoints/blair_base/Sports/item_embedding.pt \
      checkpoints/blair_base/Toys/item_embedding.pt \
    --sequence_embeddings_paths \
      checkpoints/blair_base/Arts/item_embedding.pt \
      checkpoints/blair_base/Beauty/item_embedding.pt \
      checkpoints/blair_base/Instruments/item_embedding.pt \
      checkpoints/blair_base/Office/item_embedding.pt \
      checkpoints/blair_base/Pantry/item_embedding.pt \
      checkpoints/blair_base/Scientific/item_embedding.pt \
      checkpoints/blair_base/Sports/item_embedding.pt \
      checkpoints/blair_base/Toys/item_embedding.pt \
    --batch_size 16 \
    --train_data_split item \
    --test_data_split test \
    --sample_method random \
    --max_steps 500 \
    --loss_type SINGLE_PSEUDO_LABEL_KD \
    --coefficient 1000 \
    --initial_per_weight 0.2 \
    --merge_type task_vector \
    --learn_type task_wise \
    --learning_rate 0.001 \
    --seed 42
