#!/bin/bash

# This is an example of:
# - merging recformer_base models
# - with ties merge type
# - and layer_wise merging coefficient
# You can change `model_type` to recformer_large

python merge_train.py \
    --model_kwargs ckpt_path checkpoints/pretrained/recformer_base_pretrained.pt \
    --model_type recformer_base \
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
    --item_embeddings_paths \
      checkpoints/recformer_base/Arts/item_embedding.pt \
      checkpoints/recformer_base/Beauty/item_embedding.pt \
      checkpoints/recformer_base/Instruments/item_embedding.pt \
      checkpoints/recformer_base/Office/item_embedding.pt \
      checkpoints/recformer_base/Pantry/item_embedding.pt \
      checkpoints/recformer_base/Scientific/item_embedding.pt \
      checkpoints/recformer_base/Sports/item_embedding.pt \
      checkpoints/recformer_base/Toys/item_embedding.pt \
    --sequence_embeddings_paths \
      checkpoints/recformer_base/Arts/item_embedding.pt \
      checkpoints/recformer_base/Beauty/item_embedding.pt \
      checkpoints/recformer_base/Instruments/item_embedding.pt \
      checkpoints/recformer_base/Office/item_embedding.pt \
      checkpoints/recformer_base/Pantry/item_embedding.pt \
      checkpoints/recformer_base/Scientific/item_embedding.pt \
      checkpoints/recformer_base/Sports/item_embedding.pt \
      checkpoints/recformer_base/Toys/item_embedding.pt \
    --batch_size 16 \
    --train_data_split item \
    --test_data_split test \
    --sample_method random \
    --max_steps 500 \
    --loss_type SINGLE_PSEUDO_LABEL_KD \
    --initial_per_weight 1 \
    --coefficient 1000 \
    --merge_type ties \
    --learn_type layer_wise \
    --learning_rate 0.001 \
    --seed 42
