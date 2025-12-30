#!/bin/bash

# The script will extract item embeddings from a model checkpoint.
# The item embedding will be saved to the same directory as the checkpoint.

MODEL_PATH=$1

python extract.py $MODEL_PATH
